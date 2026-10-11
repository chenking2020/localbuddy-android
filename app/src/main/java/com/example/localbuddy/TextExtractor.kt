package com.example.localbuddy

/**
 * **抽取式关键句选择器**（extractive key-sentence extractor，纯算法、零 LLM）。
 *
 * 存在的理由：端侧 0.6B 单次 prompt ≤ 1024 token，长文必须"减"才能进模型。
 * 之前所有方案都是**按块截断**（滚动式/逐段式），每轮模型只看得到一块——这是
 * "只保留最前边段"的根因。这里换成**全文打分筛选**：
 *
 * 1. 切句 → 按信息量打分（词权重 + 位置权重 + 数字/决策信号词奖励 + 长度惩罚）；
 * 2. **分位配额**：把全文按位置分成若干桶，先每桶取一句，强制头/中/尾都有代表
 *    ——覆盖性由配额保证，而不是靠模型拼；
 * 3. **MMR 去冗余**：后续名额按 `score - λ·maxSim(候选, 已选)` 贪心，避免选出一堆近义句；
 * 4. 结果按**原文顺序**输出（读起来仍是一段连贯的摘录），并按 token 预算裁剪。
 *
 * 产物两个用途：① 生成中作为**即时预览**（毫秒级可见、零推理）；
 * ② 作为 LLM 的**事实锚点**——人名/数字/决定都已在 prompt 里，模型不需要"回忆"，
 * 既省去多轮生成，也显著降低长文总结的遗漏与编造。
 */
internal object TextExtractor {

    /** 一句候选句及其打分（[index] 为原文句序，输出时按它排序）。 */
    internal data class Candidate(val text: String, val index: Int, val score: Double)

    /** 切句分隔符：中英文句末标点 + 换行（换行也视为句边界，字幕/转写天然一行一句）。 */
    private val SENT_SPLIT = Regex("[。！？!?；;\\n]+")

    /** 虚词字：其构成的 bigram 无信息量，不计入词权重（避免"的/了/我们"刷分）。 */
    private val STOP_CHARS = setOf(
        '的', '了', '是', '我', '你', '他', '她', '它', '们', '这', '那', '就', '都', '也', '很',
        '吧', '啊', '呢', '吗', '哦', '嗯', '个', '有', '在', '对', '说', '和', '与', '把', '被'
    )

    /** 决策/事实信号词：命中代表"信息密度高"（结论、数字、待办、时间点是总结的骨架）。 */
    private val SIGNAL_WORDS = listOf(
        "决定", "结论", "待办", "计划", "安排", "预算", "成本", "截止", " deadline", "预计", "必须",
        "需要", "要求", "目标", "风险", "问题", "原因", "方案", "负责人", "下周", "明天", "今天",
        "同意", "确认", "通过", "拒绝", "上线", "发布", "修复", "优化", "提升", "下降", "增长"
    )
    private val NUMBER = Regex("\\d")

    /** 默认最多摘录句数。 */
    internal const val DEFAULT_MAX_SENTENCES = 14

    /** 文档问答的检索句数上限：端上 prompt 只有 ~1k token，塞太多句反而稀释问题。 */
    internal const val DEFAULT_RETRIEVE_SENTENCES = 10
    /**
     * 配额阶段判定「与已选句重复」的 bigram Jaccard 阈值。
     * 刻意取很高（0.85 ≈ 几乎逐字重复）：配额的目的是**覆盖**，口语转录里大量句子句式相近
     * （bigram 重叠天然偏高），阈值一松就会把中部/尾部的名额判成"重复"而丢掉覆盖。
     * 真正的去冗余交给 MMR 阶段的连续相似度惩罚。
     */
    private const val DUP_THRESHOLD = 0.85

    /**
     * 选出关键句。
     *
     * @param tokenCounter token 计数（真 tokenizer 或按字符的估算），用于按预算裁剪
     * @param tokenBudget  摘录允许占用的 token 上限（prompt 预算减去 system/指令后的余额）
     * @return 按原文顺序排列的关键句；文本过短则原样返回（不筛，信息最全）
     */
    internal fun extract(
        text: String,
        tokenBudget: Int,
        tokenCounter: (String) -> Int = { it.length },
        maxSentences: Int = DEFAULT_MAX_SENTENCES
    ): List<String> {
        val sents = splitSentences(text)
        if (sents.isEmpty()) return emptyList()
        // 短文本无需筛选：全给模型，信息最全，也省掉打分开销
        if (sents.size <= maxSentences &&
            tokenCounter(sents.joinToString("\n")) <= tokenBudget) return sents

        val scored = score(sents)
        val picked = select(scored, maxSentences)
        return fitBudget(picked, tokenBudget, tokenCounter)
    }

    /**
     * 面向**文档问答**的检索式选句：只按「与问题的字面重合度」取 top-N。
     *
     * 与 [extract] 的分歧点：[extract] 走分位配额（头/中/尾各取一句）→ 适合"总结"这种要覆盖
     * 全文的任务；问答要的是**相关性**，配额会把与问题无关的头尾句硬塞进 prompt，反而稀释答案、
     * 让 0.6B 小模型跑题。这里纯按重合度排序取 top-N，再按原文顺序输出（读起来仍连贯）。
     *
     * 重合度两路相加：bigram Jaccard（词组级）+ 字符命中率（兜底短问句，如"多少钱"——
     * 其 bigram 在原文里可能一个都组不上）。
     */
    internal fun retrieve(
        text: String,
        query: String,
        tokenBudget: Int,
        tokenCounter: (String) -> Int = { it.length },
        maxSentences: Int = DEFAULT_RETRIEVE_SENTENCES
    ): List<String> {
        val sents = splitSentences(text)
        if (sents.isEmpty()) return emptyList()
        // 空问题 / 无可用 bigram：退回覆盖式抽取（至少要给模型一段有代表性的内容）
        val qb = bigrams(query)
        val qChars = query.toSet().filter { it !in STOP_CHARS && !it.isWhitespace() }
        if (query.isBlank() || (qb.isEmpty() && qChars.isEmpty())) {
            return extract(text, tokenBudget, tokenCounter, maxSentences)
        }

        val ranked = sents.mapIndexed { i, s ->
            val sb = bigrams(s)
            val overlap = if (sb.isEmpty()) 0.0 else jaccard(sb, qb)
            val charHit = qChars.count { it in s }.toDouble() / qChars.size
            Candidate(s, i, overlap * 10.0 + charHit)
        }
        val hit = ranked.filter { it.score > 0.0 }
            .sortedByDescending { it.score }
            .take(maxSentences)
            .sortedBy { it.index }
        // 一句都没命中（问法与原文用词完全不同）：仍取最相关的若干句，别让 prompt 空着
        val kept = if (hit.isEmpty())
            ranked.sortedByDescending { it.score }.take(maxSentences).sortedBy { it.index }
        else hit
        return fitBudget(kept, tokenBudget, tokenCounter)
    }

    /** 切句：按标点/换行切，去掉空白与过短碎片。 */
    internal fun splitSentences(text: String): List<String> =
        text.split(SENT_SPLIT)
            .map { it.trim() }
            .filter { it.length >= 4 }

    /** 打分：词权重 + 位置 + 信号词 + 长度惩罚。 */
    internal fun score(sents: List<String>): List<Candidate> {
        if (sents.isEmpty()) return emptyList()
        // 词权重：字符 bigram 的词频（无分词器，中文 bigram 近似词）
        val df = HashMap<String, Int>()
        sents.forEach { s -> bigrams(s).forEach { df[it] = (df[it] ?: 0) + 1 } }
        val n = sents.size
        return sents.mapIndexed { i, s ->
            var sc = 0.0
            for (bg in bigrams(s)) sc += 1.0 + kotlin.math.ln((df[bg] ?: 1).toDouble())
            sc *= positionWeight(i, n)
            val signals = SIGNAL_WORDS.count { s.contains(it) } +
                NUMBER.findAll(s).count().coerceAtMost(3) * 0.5
            sc += signals.coerceAtMost(4.0)
            sc *= lengthPenalty(s)
            Candidate(s, i, sc)
        }
    }

    /**
     * 选择：先分位配额（头/中/尾各取一句，强制覆盖）→ 再 MMR 贪心补满名额。
     * 位置配额是"全面"的关键：纯按分数取会让头部高分句占满名额，后段直接消失。
     */
    internal fun select(scored: List<Candidate>, maxSentences: Int): List<Candidate> {
        if (maxSentences <= 0) return emptyList()
        val picked = ArrayList<Candidate>(maxSentences)
        val taken = HashSet<Int>()
        val buckets = 3
        // 1) 配额阶段：把序列三等分，每桶取一句。桶内优先挑与已选句不重复的——
        //    整桶都在复述同一句时宁可放弃这一桶名额，把名额留给 MMR 阶段的新信息
        for (b in 0 until buckets) {
            val from = scored.size * b / buckets
            val to = (scored.size * (b + 1) / buckets).coerceAtMost(scored.size)
            val cands = scored.subList(from, to)
                .filter { it.index !in taken }
                .sortedByDescending { it.score }
            val best = cands.firstOrNull { c -> picked.none { p -> isDup(c.text, p.text) } }
            if (best != null && picked.size < maxSentences) { picked.add(best); taken.add(best.index) }
        }
        // 2) MMR 阶段：剩余名额在"相关且不与已选重复"之间权衡
        while (picked.size < maxSentences) {
            var best: Candidate? = null
            var bestMmr = Double.NEGATIVE_INFINITY
            for (c in scored) {
                if (c.index in taken) continue
                var maxSim = 0.0
                val cbg = bigrams(c.text)
                for (p in picked) maxSim = maxOf(maxSim, jaccard(cbg, bigrams(p.text)))

                val mmr = c.score - 1.6 * maxSim * c.score   // λ·sim·score：分数越高越怕重复
                if (mmr > bestMmr) { bestMmr = mmr; best = c }
            }
            best ?: break
            picked.add(best); taken.add(best.index)
        }
        return picked.sortedBy { it.index }
    }

    /** 按 token 预算裁剪：超额时从**分数最低**的句子开始丢（保留信息量最高的）。 */
    private fun fitBudget(
        picked: List<Candidate>, tokenBudget: Int, tokenCounter: (String) -> Int
    ): List<String> {
        val kept = picked.toMutableList()
        var total = kept.sumOf { tokenCounter(it.text) } + (kept.size - 1).coerceAtLeast(0)
        while (kept.isNotEmpty() && total > tokenBudget) {
            val worst = kept.minByOrNull { it.score } ?: break
            kept.remove(worst)
            total = kept.sumOf { tokenCounter(it.text) } + (kept.size - 1).coerceAtLeast(0)
        }
        return kept.sortedBy { it.index }.map { it.text }
    }

    /** 位置权重：首句/尾句通常是主题与结论，中段权重最低。 */
    private fun positionWeight(i: Int, n: Int): Double = when {
        i == 0 -> 1.35
        i == n - 1 -> 1.2
        i < n * 0.15 -> 1.15
        i > n * 0.85 -> 1.1
        else -> 1.0
    }

    private fun lengthPenalty(s: String): Double = when {
        s.length < 8 -> 0.5
        s.length > 80 -> 0.8
        else -> 1.0
    }

    /** 字符 bigram（跳过两端皆为虚词的组合），用于词频与相似度。 */
    private fun bigrams(s: String): Set<String> {
        val out = HashSet<String>()
        for (i in 0 until s.length - 1) {
            val a = s[i]; val b = s[i + 1]
            if (a in STOP_CHARS && b in STOP_CHARS) continue
            out.add("$a$b")
        }
        return out
    }

    /** 与已选句重复判定（bigram Jaccard 阈值）：配额与 MMR 阶段共用的去重口径。 */
    private fun isDup(a: String, b: String): Boolean = jaccard(bigrams(a), bigrams(b)) > DUP_THRESHOLD

    private fun jaccard(a: Set<String>, b: Set<String>): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        var inter = 0
        for (x in a) if (x in b) inter++
        return inter.toDouble() / (a.size + b.size - inter)
    }
}
