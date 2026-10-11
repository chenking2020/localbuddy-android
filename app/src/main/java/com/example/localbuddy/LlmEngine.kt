package com.example.localbuddy

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * 纪要整理 LLM 推理引擎：assets/llm —— Qwen3-0.6B（q4 量化 ONNX，877MB）。
 *
 * 图结构与 MT 同构（复用 runMt 的 KV cache 自回归模式）：
 *  - 算子：MatMulNBits(196) + 融合 GroupQueryAttention(28) + SimplifiedLayerNormalization / RotaryEmbedding；
 *  - 输入：input_ids / attention_mask / position_ids / past_key_values.{0..27}.{key,value}，KV 为
 *    FLOAT32 [1, 8, past_seq, 128]（8 KV 头 × head_dim 128）；
 *  - 输出：logits [1, seq, 151936] + present.{i}.{key,value}。
 *
 * 分词器复用 [HfBpeTokenizer]（Qwen HF tokenizer.json，addPrefixSpace=false）；
 * 聊天模板为 Qwen 标准 im_start/im_end 格式。
 *
 * ⚠️ 思考模式（Qwen3 混合思考）默认**关闭**：Qwen3 的 chat_template 在
 * enable_thinking=False 时会在 assistant 开头渲染一个空思考块 `<think>\n\n</think>\n\n`，
 * 模型随即跳过推理链直接输出正文（等价于 API 侧的 enable_thinking=False）。本项目手写
 * prompt，必须自己拼这段，见 [runGenerate]。输出另有 [stripThink] 兜底剥离。
 */
class LlmEngine private constructor(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    @Volatile private var session: OrtSession? = null
    @Volatile private var tokenizer: HfBpeTokenizer? = null
    @Volatile private var warmed: Boolean = false
    private val warmUpMutex = Mutex()
    /** LLM 推理串行锁：OrtSession 非线程安全。 */
    private val inferMutex = Mutex()

    val ready: Boolean get() = session != null && tokenizer != null
    val isLoaded: Boolean get() = warmed

    /** 异步加载模型（幂等；失败置 ready=false，上层显示错误）。 */
    suspend fun warmUp() = withContext(Dispatchers.IO) {
        if (warmed) return@withContext
        warmUpMutex.withLock {
            if (warmed) return@withLock
            runCatching {
                tokenizer = HfBpeTokenizer(readAssetText("llm/tokenizer.json"), addPrefixSpace = false)
                Log.i(TAG, "LLM tokenizer loaded (im_start=${tokenizer!!.imStartId}, im_end=${tokenizer!!.imEndId})")
            }.onFailure { Log.w(TAG, "LLM tokenizer load failed", it) }
            runCatching {
                val dir = extractAssetDir("llm")
                val so = sessionOpts()
                session = env.createSession("$dir/model_q4.onnx", so)
                Log.i(TAG, "LLM ready: $dir/model_q4.onnx")
            }.onFailure { Log.w(TAG, "LLM model load failed", it) }
            // 加载成功才置 warmed：会话/分词器任一为 null（如内存不足加载失败）则保持 false，
            // 下次 summarize 会重新 warmUp 重试，而非把失败永久固化（旧实现 warmed 恒为 true，
            // 一旦加载失败后续总结永远拿到 session=null、输出空、却显示"模型未就绪"且无法恢复）。
            warmed = ready
        }
    }

    /**
     * 流式生成。按 Qwen 聊天模板组装 system/user 消息，贪心解码 + 重复惩罚，
     * 每生成一个 token 以「当前已生成文本」回调 [onPartial]（回调在推理线程，StateFlow 更新线程安全）。
     *
     * @param enableThinking 是否开启 Qwen3 思考链。**默认 false（不思考）**：端上 0.6B 小模型
     *   的思考链对摘要质量基本无益，却会先吐几百个思考 token 才出正文——长文本分层总结时每个
     *   分块都要付这份代价，整体慢数倍；且思考期间 [stripThink] 返回空串，界面长时间空白。
     * @return 完整生成文本（已剥离 <think> 块）；模型未就绪返回空串。
     */
    suspend fun generate(
        user: String,
        system: String = DEFAULT_SYSTEM,
        maxNew: Int = 512,
        enableThinking: Boolean = DEFAULT_ENABLE_THINKING,
        onPartial: ((String) -> Unit)? = null
    ): String = withContext(Dispatchers.Default) {
        val tok = tokenizer ?: return@withContext ""
        val sess = session ?: return@withContext ""
        inferMutex.withLock {
            runCatching { runGenerate(tok, sess, user, system, maxNew, enableThinking, { !isActive }, onPartial) }
                .getOrElse { e ->
                    Log.e(TAG, "LLM generate failed", e)
                    // OOM 单独上抛：分步 prefill 后单步 logits 拷贝有界（≈78MB），但仍可能因
                    // KV cache 原生内存超限（约 229KB/token，past+present 双份）触发。原实现被
                    // runCatching 吞成空串、UI 误报"模型未就绪"，这里转成明确错误让上层提示。
                    if (e is OutOfMemoryError) {
                        throw RuntimeException(
                            "内存不足：字幕过长，单次推理所需显存超出设备上限，请缩短字幕文本或分段后再试", e
                        )
                    }
                    ""
                }
        }
    }

    private fun runGenerate(
        tok: HfBpeTokenizer, sess: OrtSession,
        user: String, system: String, maxNew: Int,
        enableThinking: Boolean,
        isCancelled: () -> Boolean,
        onPartial: ((String) -> Unit)?
    ): String = runDecode(
        tok, sess, buildSingleTurnSeq(tok, system, user, enableThinking),
        maxNew, isCancelled, null, onPartial
    )

    /**
     * 单轮 prompt：[system][user] 两段 + assistant 开头（含思考开关）。
     *
     * ⚠️ 输入长度护栏：KV cache FLOAT32 约 229KB/token（28 层 × 8 头 × 128 维 × K/V），上限同时
     * 约束原生内存；超限截断 user 内容（调用方应优先走 [summarizeLong] 分层总结而非截断）。
     */
    private fun buildSingleTurnSeq(
        tok: HfBpeTokenizer, system: String, user: String, enableThinking: Boolean
    ): List<Long> {
        val seq = ArrayList<Long>(256)
        fun addText(t: String) = tok.encode(t).forEach { seq.add(it.toLong()) }
        fun addSpecial(id: Int) = seq.add(id.toLong())
        addSpecial(tok.imStartId); addText("system\n"); addText(system); addSpecial(tok.imEndId); addText("\n")
        addSpecial(tok.imStartId); addText("user\n")
        val templateOverhead = seq.size + 6   // 模板固定 token + user 尾部（im_end/换行/assistant 头）
        val userBudget = (MAX_PROMPT_TOKENS - templateOverhead).coerceAtLeast(64)
        // 思考开关由模板统一处理 → 先摘掉调用方可能拼在 user 末尾的软开关（/no_think、<arg_key:6124c78e>），
        // 免得与下面的空思考块重复/冲突。
        val userText = stripThinkSwitch(user)
        val userIds = tok.encode(userText)
        if (userIds.size > userBudget) {
            Log.w(TAG, "user text truncated: ${userIds.size} -> $userBudget tokens (seq cap $MAX_PROMPT_TOKENS)")
            addText(tok.decode(userIds.take(userBudget)) + "\n（后文过长已截断）")
        } else {
            addText(userText)
        }
        addSpecial(tok.imEndId); addText("\n")
        return appendAssistantHead(seq, tok, enableThinking)
    }

    /** 补上 assistant 段头（含关闭思考时的空思考块），返回同一 seq。 */
    private fun appendAssistantHead(
        seq: ArrayList<Long>, tok: HfBpeTokenizer, enableThinking: Boolean
    ): ArrayList<Long> {
        fun addText(t: String) = tok.encode(t).forEach { seq.add(it.toLong()) }
        fun addSpecial(id: Int) = seq.add(id.toLong())
        addSpecial(tok.imStartId); addText("assistant\n")
        if (!enableThinking && tok.thinkStartId != 0 && tok.thinkEndId != 0) {
            // Qwen3 关闭思考的**硬开关**（等价于 API 侧 enable_thinking=False）：官方 chat_template
            // 在 enable_thinking=False 时会在 assistant 开头渲染一个空思考块，模型看到已闭合的空
            // <think> 便直接输出正文。
            // ⚠️ 必须按 **id** 拼：encode() 不查 added_tokens，写进文本的 "<think>" 会被 BPE 切碎，
            // 模型认不出边界 → 开关失效。此前只在 user 末尾拼 "/no_think" 同样无效（既非官方软开关，
            // 也会被切碎）→ 思考一直是开着的：长文本每个分块都先吐几百个思考 token，慢数倍。
            addSpecial(tok.thinkStartId); addText("\n\n")
            addSpecial(tok.thinkEndId); addText("\n\n")
        }
        return seq
    }

    /**
     * prefill（[PREFILL_STEP] 分步）+ 逐 token decode，共用给单轮 [generate] 与多轮 [chat]。
     *
     * @param onThink 深度思考模式下**思考过程**的增量回调（完整思考文本，会重复下发整段；
     *                思考块闭合后停止回调）。不思考模式恒不回调。
     */
    private fun runDecode(
        tok: HfBpeTokenizer, sess: OrtSession,
        seq: List<Long>, maxNew: Int,
        isCancelled: () -> Boolean,
        onThink: ((String) -> Unit)?,
        onPartial: ((String) -> Unit)?
    ): String {
        val eosIds = intArrayOf(tok.imEndId, 151643)   // <|im_end|> / <|endoftext|>
        val generated = ArrayList<Int>()
        var prevResult: OrtSession.Result? = null
        var past: List<Pair<OnnxTensor, OnnxTensor>> = emptyList()
        try {
            // ── 分步 prefill（chunked prefill）：按 [PREFILL_STEP] 逐段送入，KV cache 逐段累积。
            // 只有最后一段读 logits——中间段的 logits 是整张 [1,step,V]（每 token ~608KB）的
            // Java 堆拷贝，读出来即扔；跳过它们后峰值堆内存从 seq×608KB 降到 step×608KB（≈78MB），
            // 整段一次性 prefill 的 219MB 级拷贝 + GC 风暴消失，长上下文分块总结不再卡顿/OOM。
            var logits: FloatArray? = null
            var consumed = 0
            while (consumed < seq.size) {
                if (isCancelled()) return ""
                val end = minOf(consumed + PREFILL_STEP, seq.size)
                val step = runStep(sess, seq.subList(consumed, end), past, consumed.toLong(), end == seq.size)
                prevResult?.close()
                prevResult = step.result
                past = step.past
                if (step.logits != null) logits = step.logits
                consumed = end
            }

            // ── decode：每步 1 token（首个 token 直接用 prefill 最后一段的 logits）
            var nextInput: List<Long> = emptyList()
            var posStart = seq.size.toLong()
            var thinkClosed = false
            while (generated.size < maxNew) {
                if (isCancelled()) break   // 协作式取消：正在跑的单次推理完成后立即停
                if (nextInput.isNotEmpty()) {
                    val step = runStep(sess, nextInput, past, posStart, needLogits = true)
                    prevResult?.close()
                    prevResult = step.result
                    past = step.past
                    logits = step.logits
                    posStart += nextInput.size
                }
                val lg = logits ?: break
                if (generated.isNotEmpty()) applyRepPenaltyInPlace(lg, generated)
                val nextId = argmax(lg)
                if (nextId in eosIds) break
                generated.add(nextId)
                if (onThink != null && !thinkClosed) {
                    // 思考块闭合后不再回调（正文阶段不该再刷思考区）
                    thinkText(tok, generated)?.let { runCatching { onThink(it) } }
                    if (generated.contains(tok.thinkEndId)) thinkClosed = true
                }
                if (onPartial != null) {
                    runCatching {
                        onPartial(stripThink(tok.decode(
                            stripThinkTokens(generated, tok.thinkStartId, tok.thinkEndId)
                        )))
                    }
                }
                nextInput = listOf(nextId.toLong())
            }
        } finally {
            prevResult?.close()
        }
        return stripThink(tok.decode(stripThinkTokens(generated, tok.thinkStartId, tok.thinkEndId))).trim()
    }

    /**
     * 生成过程中的思考内容（id 层取，[HfBpeTokenizer.decode] 会跳过 `<think>` 这类 added token，
     * 文本层找不到边界）：未出现 `<think>` 返回 null；已闭合返回完整思考；未闭合返回当前已生成部分。
     */
    private fun thinkText(tok: HfBpeTokenizer, ids: List<Int>): String? {
        val s = tok.thinkStartId
        val e = tok.thinkEndId
        if (s == 0 || e == 0) return null
        val ts = ids.indexOf(s)
        if (ts < 0) return null
        val te = ids.indexOf(e)
        val end = if (te >= 0) te else ids.size
        if (end <= ts + 1) return null
        return tok.decode(ids.subList(ts + 1, end)).trim()
    }

    /** 单次 sess.run：输入 [stepTokens]（带上 [past] KV），返回该步 logits（可选）与新 KV。
     *  result 由调用方负责关闭（其 present KV 张量生命周期挂在其上，需跑完下一步再关）。 */
    private class StepResult(
        val logits: FloatArray?,
        val past: List<Pair<OnnxTensor, OnnxTensor>>,
        val result: OrtSession.Result
    )

    private fun runStep(
        sess: OrtSession,
        stepTokens: List<Long>,
        past: List<Pair<OnnxTensor, OnnxTensor>>,
        posStart: Long,
        needLogits: Boolean
    ): StepResult {
        val curLen = stepTokens.size
        val inputs = HashMap<String, OnnxTensor>()
        val owned = ArrayList<OnnxTensor>()
        inputs["input_ids"] = longTensor(stepTokens.toLongArray()).also { owned.add(it) }
        val total = posStart + curLen
        inputs["attention_mask"] = longTensor(LongArray(total.toInt()) { 1L }).also { owned.add(it) }
        // Qwen3 图显式接收 position_ids（prefill 分步与 decode 统一：本步 token 的绝对位置）
        inputs["position_ids"] = longTensor(LongArray(curLen) { posStart + it }).also { owned.add(it) }
        if (past.isEmpty()) {
            for (i in 0 until N_LAYERS) {
                val k = emptyKv(); val v = emptyKv()
                inputs["past_key_values.$i.key"] = k; inputs["past_key_values.$i.value"] = v
                owned.add(k); owned.add(v)
            }
        } else {
            past.forEachIndexed { i, (k, v) ->
                inputs["past_key_values.$i.key"] = k
                inputs["past_key_values.$i.value"] = v
            }
        }

        val result = sess.run(inputs)
        owned.forEach { runCatching { it.close() } }
        val logits = if (needLogits) readLastLogits(result.get("logits").get() as OnnxTensor, curLen) else null
        val newPast = (0 until N_LAYERS).map {
            (result.get("present.$it.key").get() as OnnxTensor) to
                (result.get("present.$it.value").get() as OnnxTensor)
        }
        return StepResult(logits, newPast, result)
    }

    /**
     * **多轮对话**生成（文本对话 / 语音通话智能体的底座）。
     *
     * 与 [generate] 的区别只在 prompt 组装：这里把完整对话历史按 Qwen 模板逐轮展开
     * （system + user/assistant 交替 + 待生成的 assistant），让模型看得到上文；
     * 推理路径（分步 prefill + 贪心 decode + 重复惩罚）完全复用。
     *
     * ⚠️ 上下文预算仍是 [MAX_PROMPT_TOKENS]：超预算时**从最旧的对话轮次开始丢弃**
     * （system 与最后一轮 user 恒保留，保证"人设"和"当前问题"不丢）；
     * 最后一轮自身超预算时截断其中部。端上 0.6B 的可用上下文本就有限，
     * 上层应只保留最近若干轮（见 TextChatAgent 的 [CHAT_HISTORY_TURNS]）。
     *
     * @param messages 完整对话序列：首条建议为 system，其余 user/assistant 交替，以 user 结尾。
     * @param onThink  深度思考模式下的思考过程回调（思考块闭合后停止）；不思考模式恒不回调。
     */
    suspend fun chat(
        messages: List<ChatTurn>,
        maxNew: Int = 512,
        enableThinking: Boolean = false,
        onThink: ((String) -> Unit)? = null,
        onPartial: ((String) -> Unit)? = null
    ): String = withContext(Dispatchers.Default) {
        val tok = tokenizer ?: return@withContext ""
        val sess = session ?: return@withContext ""
        inferMutex.withLock {
            runCatching {
                runDecode(
                    tok, sess, buildChatSeq(tok, messages, enableThinking),
                    maxNew, { !isActive }, onThink, onPartial
                )
            }.getOrElse { e ->
                Log.e(TAG, "LLM chat failed", e)
                if (e is OutOfMemoryError) {
                    throw RuntimeException(
                        "内存不足：对话上下文过长，请新建会话或缩短输入后再试", e
                    )
                }
                ""
            }
        }
    }

    /** 组装多轮 prompt：尾部优先保留，超预算丢最旧的轮次。 */
    private fun buildChatSeq(
        tok: HfBpeTokenizer, messages: List<ChatTurn>, enableThinking: Boolean
    ): List<Long> {
        val system = messages.firstOrNull { it.role == ROLE_SYSTEM }?.content.orEmpty()
        val turns = messages.filter { it.role != ROLE_SYSTEM && it.content.isNotBlank() }

        val head = ArrayList<Long>()
        fun addText(seq: MutableList<Long>, t: String) = tok.encode(t).forEach { seq.add(it.toLong()) }
        addText(head, "system\n"); addText(head, system)

        val budget = MAX_PROMPT_TOKENS - 8 - (if (enableThinking) 0 else 4)
        val kept = ArrayList<ChatTurn>()
        var used = head.size + 8      // system 段头/段尾 + assistant 段头 的固定开销
        for (i in turns.lastIndex downTo 0) {
            val t = turns[i]
            val ids = tok.encode(stripThinkSwitch(t.content))
            if (used + ids.size + 4 > budget) {
                // 连最后一轮都装不下：截断保留（至少有问题的一部分，不至于空 prompt）
                if (kept.isEmpty()) {
                    val room = (budget - used - 8).coerceAtLeast(64)
                    kept.add(ChatTurn(t.role,
                        tok.decode(ids.take(room)) + "\n（后文过长已截断）"))
                }
                break
            }
            used += ids.size + 4
            kept.add(0, t)
        }
        if (kept.size < turns.size) {
            Log.w(TAG, "chat history trimmed: ${turns.size} -> ${kept.size} turns (cap $MAX_PROMPT_TOKENS)")
        }

        val seq = ArrayList<Long>(256)
        seq.add(tok.imStartId.toLong()); seq.addAll(head)
        seq.add(tok.imEndId.toLong()); addText(seq, "\n")
        for (t in kept) {
            seq.add(tok.imStartId.toLong()); addText(seq, "${t.role}\n")
            addText(seq, stripThinkSwitch(t.content))
            seq.add(tok.imEndId.toLong()); addText(seq, "\n")
        }
        return appendAssistantHead(seq, tok, enableThinking)
    }

    /**
     * 超长文本的 **逐段独立总结**（segmented summarization）。
     *
     * 演进史：map-reduce 要全部跑完才出结果；滚动式（`要点 = f(要点, 新片段)`）虽然即时可见，
     * 但每轮都要重写整份要点（输出长 → 每轮慢），且 0.6B 小模型整合时会不断稀释、丢失后段信息。
     *
     * 现在回归最简单也最快的模式：**每段完全独立**地总结出「要点 + 待办」，
     * 结果直接**流式追加**到输出——没有任何跨段推理（无需携带上下文 → prompt 更短、
     * 输出只有本段要点 → 每轮生成 token 大减），也不做收尾汇总（少一整轮推理）。
     * 每段输出约定用「【待办】」行分隔两部分，[splitPointsTodo] 实时切开后
     * 分别通过 [onPartial]（要点增量）与 [onTodo]（待办增量）追加给调用方。
     *
     * 端上内存约束不变：单次 prompt ≤ [MAX_PROMPT_TOKENS]，KV cache 有界；
     * 无累积要点占预算 → 片段预算更大 → 段数更少 → 更快。
     *
     * @param contentDesc 内容描述（如「视频字幕」「录音转写」），用于提示词措辞
     * @param onPartial  **要点增量**回调（append 语义，调用方自行拼接）
     * @param onTodo     **待办增量**回调（append 语义）；短文本路径不回调
     * @param onProgress 段进度 (第 i 段, 共 n 段)；短文本不回调
     * @return 完整总结（「【要点】…【待办】…」两节），供持久化
     */
    suspend fun summarizeLong(
        content: String,
        system: String = DEFAULT_SYSTEM,
        contentDesc: String = "长文本",
        maxNew: Int = 512,
        enableThinking: Boolean = DEFAULT_ENABLE_THINKING,
        onPartial: ((String) -> Unit)? = null,
        onTodo: ((String) -> Unit)? = null,
        onProgress: ((Int, Int) -> Unit)? = null,
        userTemplate: SummaryUserTemplate = SummaryUserTemplate()
    ): String = withContext(Dispatchers.Default) {
        val tok = tokenizer ?: return@withContext ""
        if (content.isBlank()) return@withContext ""

        // 短文本：一次总结即可（全量流式，把累计 partial 转成增量保持回调语义一致）
        val contentTokens = tok.encode(content).size
        if (contentTokens <= CHUNK_TOKENS) {
            var emitted = 0
            val text = generate(
                userTemplate.wholeDoc(contentDesc, content), system, maxNew,
                enableThinking = enableThinking,
                onPartial = { p -> emitDelta(p, emitted) { d, len -> onPartial?.invoke(d); emitted = len } }
            )
            return@withContext text
        }

        // 片段预算：MAX_PROMPT_TOKENS 减去 system + 模板开销（无累积要点，预算全给内容）
        val overhead = tok.encode(system).size +
            tok.encode(userTemplate.segment(contentDesc, "", 1, 1)).size + 16
        val chunkBudget = (MAX_PROMPT_TOKENS - overhead).coerceAtLeast(256)
        val chunks = packChunks(tok, content, chunkBudget)
        Log.i(TAG, "summarize segmented: $contentTokens tokens -> ${chunks.size} chunks (budget $chunkBudget)")

        val points = StringBuilder()
        val todos = StringBuilder()
        chunks.forEachIndexed { i, chunk ->
            onProgress?.invoke(i + 1, chunks.size)
            val t0 = System.currentTimeMillis()
            // 每段独立：流式 partial 实时切分为「要点 / 待办」两个追加流
            var lastPointsLen = 0
            var lastTodosLen = 0
            var pointsSep = false   // 本段要点前是否已发过分隔换行
            var todosSep = false
            generate(
                userTemplate.segment(contentDesc, chunk, i + 1, chunks.size),
                system, maxNew = SEG_MAX_NEW,
                enableThinking = enableThinking,
                onPartial = { partial ->
                    val (p, t) = splitPointsTodo(partial)
                    if (p.length > lastPointsLen) {
                        if (!pointsSep && points.isNotEmpty()) { onPartial?.invoke("\n"); points.append('\n') }
                        pointsSep = true
                        onPartial?.invoke(p.substring(lastPointsLen))
                        points.append(p, lastPointsLen, p.length)
                        lastPointsLen = p.length
                    }
                    if (t.length > lastTodosLen) {
                        if (!todosSep && todos.isNotEmpty()) { onTodo?.invoke("\n"); todos.append('\n') }
                        todosSep = true
                        onTodo?.invoke(t.substring(lastTodosLen))
                        todos.append(t, lastTodosLen, t.length)
                        lastTodosLen = t.length
                    }
                }
            )
            Log.i(TAG, "segment ${i + 1}/${chunks.size}: ${System.currentTimeMillis() - t0}ms, " +
                "points ${points.length} chars, todos ${todos.length} chars")
        }
        combinePointsTodo(points.toString(), todos.toString())
    }

    /** 累计 partial → 增量：生成中 UI 只需 append，不必整段重绘。 */
    private inline fun emitDelta(partial: String, emittedLen: Int, onDelta: (String, Int) -> Unit) {
        if (partial.length > emittedLen) onDelta(partial.substring(emittedLen), partial.length)
    }

    /**
     * **抽取预热 → 大纲先行 → 单次流式生成**（Extract-then-Generate，长文总结的主路径）。
     *
     * 为什么换掉「逐段独立总结」：它快但**没有全局视野**（每段只看一块 → 无主题/结论、段间重复），
     * 而且总生成 token = 段数 × 每段输出，才是慢的根源。这里把"读全文"从 LLM 里拿出来：
     *
     * ```
     * ① 抽取（零推理，毫秒级）：全文打分选关键句 → [onPreview] 立刻可见（分位配额保证头/中/尾覆盖）
     * ② 单次生成（一次 prefill + 一次 decode）：关键句已在 prompt 里当事实锚点
     *     输出 = 【大纲】话题词 → 【总结】主题+要点 → 【待办】
     *     三段实时切分后分别增量推给 [onOutline] / [onPartial] / [onTodo]
     * ```
     *
     * 于是：t≈0s 有预览、t≈1s 有话题大纲（可见的"思考过程"）、随后流式出正文；
     * 总生成 token 从「段数 × 每段输出」降到「一次 ~300」，且覆盖由抽取配额保证（不靠拼）。
     *
     * 极长文（抽取后仍装不进预算，或摘录过少不足以成文）自动降级 [summarizeLong] 逐段模式。
     *
     * @return 完整总结（「【要点】…【待办】…」），供持久化；大纲只作过程展示，不入存档。
     */
    suspend fun summarizeSmart(
        content: String,
        system: String = DEFAULT_SYSTEM,
        contentDesc: String = "长文本",
        maxNew: Int = 512,
        enableThinking: Boolean = DEFAULT_ENABLE_THINKING,
        onPreview: ((List<String>) -> Unit)? = null,
        onOutline: ((String) -> Unit)? = null,
        onPartial: ((String) -> Unit)? = null,
        onTodo: ((String) -> Unit)? = null,
        onProgress: ((Int, Int) -> Unit)? = null,
        userTemplate: SummaryUserTemplate = SummaryUserTemplate()
    ): String = withContext(Dispatchers.Default) {
        val tok = tokenizer ?: return@withContext ""
        if (content.isBlank()) return@withContext ""

        val contentTokens = tok.encode(content).size
        // 短文本：一次总结即可，无需抽取（全量进 prompt 信息最全）
        if (contentTokens <= CHUNK_TOKENS) {
            var emitted = 0
            return@withContext generate(
                userTemplate.wholeDoc(contentDesc, content), system, maxNew,
                enableThinking = enableThinking,
                onPartial = { p -> emitDelta(p, emitted) { d, len -> onPartial?.invoke(d); emitted = len } }
            )
        }

        // ① 抽取：预算 = MAX_PROMPT_TOKENS 减去 system + 模板开销
        val overhead = tok.encode(system).size +
            tok.encode(userTemplate.extractive(contentDesc, emptyList())).size + 16
        val excerptBudget = (MAX_PROMPT_TOKENS - overhead).coerceAtLeast(256)
        val excerpt = TextExtractor.extract(
            content, excerptBudget, tokenCounter = { tok.encode(it).size }
        )
        Log.i(TAG, "summarizeSmart: $contentTokens tokens -> excerpt ${excerpt.size} sentences")

        // 摘录太少：说明抽取没能概括全文（如整篇都是超长句），退回逐段模式兜底
        if (excerpt.size < 3) {
            Log.i(TAG, "summarizeSmart: excerpt too small (${excerpt.size}), fallback to segmented")
            return@withContext summarizeLong(
                content, system, contentDesc, maxNew, enableThinking,
                onPartial = onPartial, onTodo = onTodo, onProgress = onProgress,
                userTemplate = userTemplate
            )
        }
        onPreview?.invoke(excerpt)

        // ② 单次生成：大纲 → 总结 → 待办，三段流式切分
        val outline = StringBuilder()
        val points = StringBuilder()
        val todos = StringBuilder()
        var lastOutlineLen = 0
        var lastPointsLen = 0
        var lastTodosLen = 0
        var bodyStarted = false
        generate(
            userTemplate.extractive(contentDesc, excerpt), system, maxNew,
            enableThinking = enableThinking,
            onPartial = { partial ->
                val (head, rest) = splitAtMarker(partial, "【总结】")
                if (head.length > lastOutlineLen) {
                    onOutline?.invoke(head.substring(lastOutlineLen))
                    outline.append(head, lastOutlineLen, head.length)
                    lastOutlineLen = head.length
                }
                if (rest != null) {
                    bodyStarted = true
                    val (p, t) = splitPointsTodo(rest)
                    if (p.length > lastPointsLen) {
                        onPartial?.invoke(p.substring(lastPointsLen))
                        points.append(p, lastPointsLen, p.length)
                        lastPointsLen = p.length
                    }
                    if (t.length > lastTodosLen) {
                        onTodo?.invoke(t.substring(lastTodosLen))
                        todos.append(t, lastTodosLen, t.length)
                        lastTodosLen = t.length
                    }
                }
            }
        )
        Log.i(TAG, "summarizeSmart done: outline ${outline.length}, points ${points.length}, todos ${todos.length}")
        // 模型完全没按格式输出（没有【总结】锚点）：把正文整体当要点，避免结果空白
        if (!bodyStarted && outline.isNotBlank() && points.isBlank()) {
            val text = outline.toString()
            points.setLength(0); points.append(text)
            onPartial?.invoke(text)
        }
        combinePointsTodo(points.toString(), todos.toString())
    }

    /** 按行把内容打包成 ≤[maxTokens] 的块（行=字幕条/转写段，保持语义单元完整）。 */
    private fun packChunks(tok: HfBpeTokenizer, content: String, maxTokens: Int): List<String> {
        val chunks = ArrayList<String>()
        val cur = StringBuilder()
        var curTokens = 0
        for (line in content.split("\n")) {
            val ids = tok.encode(line)
            val t = ids.size + 1   // +1 换行
            if (t > maxTokens) {
                // 单行超预算（异常长行）：按 token 硬切
                if (cur.isNotEmpty()) { chunks.add(cur.toString()); cur.setLength(0); curTokens = 0 }
                ids.chunked(maxTokens).forEach { c -> chunks.add(tok.decode(c)) }
                continue
            }
            if (curTokens + t > maxTokens && cur.isNotEmpty()) {
                chunks.add(cur.toString())
                cur.setLength(0)
                curTokens = 0
            }
            if (cur.isNotEmpty()) cur.append('\n')
            cur.append(line)
            curTokens += t
        }
        if (cur.isNotEmpty()) chunks.add(cur.toString())
        return chunks
    }

    /** 摘掉调用方可能拼在 user 末尾的思考软开关（`/no_think` / `<arg_key:6124c78e>`）：开关由模板统一控制。 */
    private fun stripThinkSwitch(s: String): String =
        s.replace(Regex("[\\s　]*/?no_think[\\s　]*$"), "")

    /** 剥离 Qwen3 思考块：完整 <think>..</think> 取其后；未闭合（生成中）返回空串。 */
    private fun stripThink(s: String): String {
        val end = s.lastIndexOf("</think>")
        if (end >= 0) return s.substring(end + "</think>".length)
        if (s.contains("<think>")) return ""
        return s
    }

    private fun applyRepPenaltyInPlace(logits: FloatArray, ids: List<Int>, penalty: Float = 1.1f) {
        val seen = HashSet<Int>()
        for (id in ids) {
            if (id in 0 until logits.size && seen.add(id)) {
                val v = logits[id]
                logits[id] = if (v > 0) v / penalty else v * penalty
            }
        }
    }

    private fun argmax(a: FloatArray): Int {
        var bi = 0; var bv = Float.NEGATIVE_INFINITY
        for (i in a.indices) if (a[i] > bv) { bv = a[i]; bi = i }
        return bi
    }

    /** logits [1, seqLen, V] 最后一行（V 个 float），不整张物化。 */
    private fun readLastLogits(tensor: OnnxTensor, seqLen: Int): FloatArray {
        val fb = tensor.getFloatBuffer()
        val total = fb.remaining()
        if (seqLen <= 0 || total == 0) return FloatArray(0)
        val v = total / seqLen
        fb.position((seqLen - 1) * v)
        val row = FloatArray(v)
        fb.get(row)
        return row
    }

    private fun longTensor(arr: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, LongBuffer.wrap(arr), longArrayOf(1, arr.size.toLong()))

    /** 空 KV cache [1,8,0,128]，FLOAT32。 */
    private fun emptyKv(): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1L, KV_HEADS.toLong(), 0L, HEAD_DIM.toLong()))

    private fun sessionOpts(): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(4, 8))
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        // 与 MT 同族算子（MatMulNBits/GQA），XNNPACK 在本机未复现 MTE 崩溃问题
        runCatching { addXnnpack(mapOf("intra_op_num_threads" to Runtime.getRuntime().availableProcessors().coerceIn(4, 8).toString())) }
            .onFailure { Log.w(TAG, "XNNPACK EP unavailable", it) }
    }

    private fun readAssetText(name: String): String =
        context.assets.open(name).bufferedReader(Charsets.UTF_8).use { it.readText() }

    /** assets/<dir> 全量解包到 filesDir/<dir>（按未压缩大小校验缓存），返回目录绝对路径。 */
    private fun extractAssetDir(dir: String): String {
        val outDir = File(context.filesDir, dir)
        val names = context.assets.list(dir) ?: emptyArray()
        if (!outDir.exists()) outDir.mkdirs()
        val nameSet = names.toSet()
        outDir.listFiles()?.forEach { f ->
            if (f.isFile && f.name !in nameSet) runCatching { f.delete() }
        }
        for (name in names) {
            val cached = File(outDir, name)
            val assetSize = try { context.assets.openFd("$dir/$name").use { it.length } } catch (e: Exception) { -1L }
            if (cached.exists() && cached.length() > 0L &&
                (assetSize < 0L || cached.length() == assetSize)) continue
            Log.i(TAG, "extracting $dir/$name")
            context.assets.open("$dir/$name").use { input ->
                FileOutputStream(cached).use { output -> input.copyTo(output) }
            }
        }
        return outDir.absolutePath
    }

    companion object {
        private const val TAG = "LlmEngine"
        /** Qwen3-0.6B：28 层、8 KV 头、head_dim 128（与图中 past_key_values 形状一致）。 */
        private const val N_LAYERS = 28
        private const val KV_HEADS = 8
        private const val HEAD_DIM = 128
        /** 单次推理 prompt token 硬上限。
         *  分步 prefill（[PREFILL_STEP]）后，logits 的 Java 堆拷贝峰值 = 一步 × 608KB ≈ 78MB
         *  （2026-09-12 复现过整段 prefill 一次性物化 logits 在 seq≈592 时 OOM，已修复）。
         *  现在的约束变成 KV cache 原生内存：FLOAT32 约 229KB/token（28 层 × 8 头 × 128 维 × K/V），
         *  运行瞬间 past+present 双份共存，1024 token ≈ 470MB 原生内存瞬时峰值（现代机型可承受）。
         *  超限截断（长文请走 [summarizeLong] 分层）。 */
        private const val MAX_PROMPT_TOKENS = 1024
        /** 分步 prefill 的步长：每步 logits 拷贝 = step × 608KB，128 → ≈78MB，堆内安全。 */
        private const val PREFILL_STEP = 128
        /** 分层总结的单块内容 token 预算（加 system/模板后单次 prefill ≤ MAX_PROMPT_TOKENS）。
         *  从 360 提到 640：同样内容块数减半，map 阶段串行推理次数近乎减半，长上下文总结明显提速。 */
        private const val CHUNK_TOKENS = 640
        /** 逐段总结每段的生成上限：只输出本段要点 + 待办（远短于跨段整合的输出 → 每段更快）。 */
        private const val SEG_MAX_NEW = 256
        /** 默认**不思考**：端上小模型的思考链对摘要无益，长文本分层总结却要为每个分块付数倍耗时。 */
        private const val DEFAULT_ENABLE_THINKING = false
        /** 纪要整理系统提示：强总结性（非逐句复述）、要点列表化、同语言输出。 */
        private const val DEFAULT_SYSTEM =
            "你是一个专业的会议纪要助手。请对用户提供的录音转写内容做高度提炼的总结，要求：\n" +
                "1. 用简洁的自然语言概括，不要逐句复述原文，不要罗列流水账；\n" +
                "2. 总结正文开头用一两句话说明这段录音的整体主题和背景；\n" +
                "3. 多个要点用列表形式（- 开头）逐条列出，每条是一个完整、信息密度高的句子，可合并同话题的零散内容；\n" +
                "4. 有结论、决定或待办事项时，单独列出（如「结论」「待办」小节）；\n" +
                "5. 忽略口语中的寒暄、重复和无意义内容；\n" +
                "6. 使用与录音内容相同的语言输出，直接给出纪要正文，不要任何额外解释或前缀；\n" +
                SUMMARY_FORMAT_RULE

        @Volatile private var instance: LlmEngine? = null
        fun get(context: Context): LlmEngine =
            instance ?: synchronized(this) {
                instance ?: LlmEngine(context.applicationContext).also { instance = it }
            }
    }
}

/** [LlmEngine.chat] 的对话角色常量（Qwen 模板的 im_start 角色名）。 */
internal const val ROLE_SYSTEM = "system"
internal const val ROLE_USER = "user"
internal const val ROLE_ASSISTANT = "assistant"

/**
 * 多轮对话的一轮。
 *
 * @param role [ROLE_SYSTEM] / [ROLE_USER] / [ROLE_ASSISTANT]
 * @param content 该轮文本。assistant 的历史轮次应存**已剥离思考块**的正文
 *   （[LlmEngine.chat] 返回的就是正文），否则思考内容会混进上文、把后续轮次带偏。
 */
data class ChatTurn(val role: String, val content: String)

/**
 * 在 **token id 层面** 剥离 Qwen3 思考块。
 *
 * 必须在 id 层做：[HfBpeTokenizer.decode] 会跳过全部 added token（含 `<think>` / `</think>`），
 * 解码后的文本里根本找不到 `</think>` → 纯字符串剥离（[LlmEngine] 的 stripThink）形同虚设，
 * 思考内容会整段混进"正文"。
 *
 * @return 已闭合（出现 `</think>`）返回其后正文；思考中（只有 `<think>` 未闭合）返回空；
 *         无思考标记原样返回；未配置思考标记 id（传 0，如非 Qwen 词表）也原样返回。
 */
internal fun stripThinkTokens(ids: List<Int>, thinkStart: Int, thinkEnd: Int): List<Int> {
    if (thinkEnd != 0) {
        val end = ids.lastIndexOf(thinkEnd)
        if (end >= 0) return ids.subList(end + 1, ids.size)
    }
    if (thinkStart != 0 && ids.indexOf(thinkStart) >= 0) return emptyList()
    return ids
}

/**
 * 系统提示的统一尾条：**格式要求归用户消息**。
 *
 * 抽取/逐段路径都靠【总结】【待办】这类小节锚点做流式切分；系统提示里任何笼统的
 * "用列表输出""开头写一段话"都会被小模型当成格式，从而改写或吞掉锚点 → 切分失效。
 * 所以各套 system 都必须以它收尾。
 */
internal const val SUMMARY_FORMAT_RULE =
    "7. 若用户消息指定了输出格式（如小节标题与顺序），严格按该格式输出，" +
        "不要自行调整顺序或改写小节标题。"

/** [SUMMARY_FORMAT_RULE] 的英文版：指令语言切成英文时配套使用（口径一致，只是换语言）。 */
internal const val SUMMARY_FORMAT_RULE_EN =
    "7. If the user message specifies an output format (section titles and order), follow it exactly; " +
        "do not reorder or rename the sections."

/**
 * 跨语沟通的**双语纪要** system：与普通纪要唯一的区别是**强制输出语种**——
 * 同一场对话要分别整理成 A 方语种版和 B 方语种版，不能像默认提示那样"跟随原文语言"。
 *
 * **指令语言随目标语种走**：目标是中文就用中文下指令（[zhInstruction]），其余一律用英文下指令。
 * 端侧小模型对"用哪种语言回答"极敏感——整段中文指令 + 一句"请用英语输出"，实测仍会顺着
 * 中文指令输出中文；把指令整体换成英文、只有内容是非中文，语种才守得住。
 *
 * @param targetLangName 目标输出语种名（如「英语」，用于中文指令版）
 * @param targetLangCode 目标输出语种码（如 `en`，决定指令语言与英文名）
 */
internal fun bilingualSummarySystem(targetLangName: String, targetLangCode: String): String =
    if (zhInstruction(targetLangCode)) {
        "你是一个专业的跨语沟通纪要助手。请对用户提供的双方对话内容整理成沟通纪要，要求：\n" +
            "1. 全程必须使用${targetLangName}输出（含小节标题、要点、待办），" +
            "即使对话里出现其他语种也不要混用；\n" +
            "2. 用简洁的自然语言概括，不要逐句复述原文，不要罗列流水账；\n" +
            "3. 总结正文开头用一两句话说明这场沟通的整体主题与背景；\n" +
            "4. 多个要点用列表形式（- 开头）逐条列出，每条是一个完整、信息密度高的句子，" +
            "可合并同话题的零散内容；\n" +
            "5. 保留关键信息：说话人、事件、数字、时间、结论、决定与待办事项；\n" +
            "6. 忽略口语中的寒暄、重复和无意义内容，直接给出纪要正文，不要任何额外解释或前缀；\n" +
            SUMMARY_FORMAT_RULE
    } else {
        "You are a professional assistant that turns cross-language conversations into meeting minutes.\n" +
            "1. Write EVERYTHING in ${langEnName(targetLangCode)} — headings, bullet points and to-dos " +
            "included. Never switch to another language, even when the transcript contains one.\n" +
            "2. Summarize in concise natural language: never transcribe sentence by sentence, " +
            "never produce a running log of who said what.\n" +
            "3. Open with one or two sentences describing the topic and background of the conversation.\n" +
            "4. Then list the key points as \"- \" bullets; each bullet is one complete, " +
            "information-dense sentence, merging scattered remarks on the same topic.\n" +
            "5. Keep the key facts: speakers, events, numbers, dates, conclusions, decisions and action items.\n" +
            "6. Skip greetings, repetitions and filler; output only the minutes, no preamble or explanation.\n" +
            SUMMARY_FORMAT_RULE_EN
    }

/** 目标语种是中文（含粤语等方言）时用中文下指令，其余用英文。 */
internal fun zhInstruction(code: String): Boolean =
    code.startsWith("zh") || code == "yue"

/** 语种英文名的兜底映射：英文指令里 "write in English" 比 "write in 英语" 稳得多。 */
private val LANG_EN_NAMES = mapOf(
    "zh" to "Chinese", "yue" to "Chinese (Cantonese)", "en" to "English", "ja" to "Japanese",
    "ko" to "Korean", "fr" to "French", "de" to "German", "es" to "Spanish", "pt" to "Portuguese",
    "ru" to "Russian", "it" to "Italian", "ar" to "Arabic", "th" to "Thai", "vi" to "Vietnamese",
    "tr" to "Turkish", "hi" to "Hindi", "id" to "Indonesian", "ms" to "Malay", "nl" to "Dutch",
    "pl" to "Polish", "uk" to "Ukrainian", "fa" to "Persian", "he" to "Hebrew", "sv" to "Swedish"
)

internal fun langEnName(code: String): String = LANG_EN_NAMES[code.substringBefore("-")] ?: code

/** 双语纪要的 contentDesc：跟随指令语言，别让一句中文描述把输出带偏。 */
internal fun bilingualContentDesc(langCode: String, langName: String): String =
    if (zhInstruction(langCode)) "${langName}对话记录" else "${langEnName(langCode)} conversation transcript"

// ── 双语纪要的 user 提示口径 ──
//
// 不能用默认那三条：wholeDocSummaryUser / extractiveSummaryUser 的末端都写着
// "使用与原文/摘录相同的语言"，而跨语场景里两个语种的内容池是**同一场对话的两种重述**，
// 那句等于让模型自由发挥 → 两份纪要一起倒向占比高的语种（实测都成中文）。
// 这里把目标语种钉死在 user 里（system 之外再压一道），并明确"要概括、不要复述"。
// 三段锚点【大纲】【总结】【待办】必须保留：流式切分靠它们，改动会让 UI 三栏失效。

internal fun bilingualUserTemplate(langCode: String): SummaryUserTemplate =
    if (zhInstruction(langCode)) {
        SummaryUserTemplate(
            wholeDoc = { desc, content -> bilingualWholeDocZh(desc, content) },
            extractive = { desc, ex -> bilingualExtractiveZh(desc, ex) },
            segment = { desc, chunk, i, n -> bilingualSegmentZh(desc, chunk, i, n) }
        )
    } else {
        val lang = langEnName(langCode)
        SummaryUserTemplate(
            wholeDoc = { desc, content -> bilingualWholeDocEn(desc, content, lang) },
            extractive = { desc, ex -> bilingualExtractiveEn(desc, ex, lang) },
            segment = { desc, chunk, i, n -> bilingualSegmentEn(desc, chunk, i, n, lang) }
        )
    }

private fun bilingualWholeDocZh(desc: String, content: String): String =
    "以下是${desc}的完整内容：\n\n$content\n\n" +
        "请把它整理成沟通纪要：\n" +
        "1. 全文（含开头概述、要点、待办）只能用中文书写，遇到其他语种一律译成中文；\n" +
        "2. 要概括，不要逐句复述、不要照抄对话原话、不要记流水账；\n" +
        "3. 先用一两句话说明这场沟通的主题与背景；\n" +
        "4. 再用 - 开头的要点列出关键内容，合并同一话题的零散发言，" +
        "保留说话人、数字、时间、结论与决定；\n" +
        "5. 最后列出待办事项，没有就写「无」。\n" +
        "直接给出纪要正文。"

private fun bilingualWholeDocEn(desc: String, content: String, lang: String): String =
    "Below is a $desc:\n\n$content\n\n" +
        "Turn it into meeting minutes:\n" +
        "1. Write EVERYTHING in $lang — the opening summary, the bullet points and the action items. " +
        "Translate anything in another language into $lang.\n" +
        "2. Summarize: do not transcribe sentence by sentence, do not copy the dialogue verbatim, " +
        "do not write a running log.\n" +
        "3. Open with one or two sentences describing the topic and background of the conversation.\n" +
        "4. Then list the key points as \"- \" bullets, merging scattered remarks on the same topic; " +
        "keep speakers, numbers, dates, conclusions and decisions.\n" +
        "5. Finish with the action items, or write \"None\" if there are none.\n" +
        "Output only the minutes."

private fun bilingualExtractiveZh(desc: String, excerpt: List<String>): String {
    val body = excerpt.mapIndexed { i, s -> "${i + 1}. $s" }.joinToString("\n")
    return "以下是${desc}的关键句摘录（共 ${excerpt.size} 句，按原文顺序，覆盖开头到结尾）：\n" +
        "$body\n\n" +
        "请基于这些关键句整理成沟通纪要，严格按下面三段格式输出：\n" +
        "【大纲】\n" +
        "先用最多 $OUTLINE_MAX_TOPICS 个话题词概括主要话题，用顿号分隔，一行写完。\n" +
        "【总结】\n" +
        "先用一两句话概括主题与背景，再用 - 开头的要点展开关键内容：合并同类话题，" +
        "保留人名、数字、时间、结论；要概括，不要逐句复述或照抄原话。这一段全部用中文书写。\n" +
        "【待办】\n" +
        "列出需要跟进的行动项，每条以 - 开头；没有就写「无」。同样全部用中文书写。\n" +
        "不要复述上面的指令，也不要混用其他语种。"
}

private fun bilingualExtractiveEn(desc: String, excerpt: List<String>, lang: String): String {
    val body = excerpt.mapIndexed { i, s -> "${i + 1}. $s" }.joinToString("\n")
    return "Below are key excerpts from a $desc (${excerpt.size} sentences, in original order, " +
        "covering the beginning to the end):\n" +
        "$body\n\n" +
        "Based on these excerpts, output the minutes strictly in these three sections:\n" +
        "【大纲】\n" +
        "At most $OUTLINE_MAX_TOPICS topic words covering the main topics, separated by \"、\", on one line.\n" +
        "【总结】\n" +
        "One or two sentences on the topic and background, then \"- \" bullets with the key content: " +
        "merge remarks on the same topic, keep names, numbers, dates and conclusions. " +
        "Summarize — do not transcribe or copy the excerpts verbatim. Write this section entirely in $lang.\n" +
        "【待办】\n" +
        "Action items, each starting with \"-\"; write \"None\" if there are none. " +
        "Also entirely in $lang.\n" +
        "Do not repeat these instructions and do not mix in any other language."
}

private fun bilingualSegmentZh(desc: String, chunk: String, index: Int, total: Int): String =
    "以下是${desc}的第 $index/$total 段。请单独概括这一段，只输出两部分：\n" +
        "1. 要点：最多 $SEG_MAX_BULLETS 条，每条以 - 开头，概括本段关键信息（人物、事件、数字、结论），" +
        "忽略寒暄与重复，不要照抄原话；\n" +
        "2. 另起一行写「【待办】」，在其下列出本段的行动项（每条以 - 开头）；没有就写「无」。\n" +
        "两部分都只能用中文书写，不要输出其他任何内容。内容如下：\n\n$chunk"

private fun bilingualSegmentEn(desc: String, chunk: String, index: Int, total: Int, lang: String): String =
    "Below is part $index/$total of a $desc. Summarize this part alone and output only two sections:\n" +
        "1. Key points: at most $SEG_MAX_BULLETS bullets, each starting with \"-\", covering the key " +
        "information (people, events, numbers, conclusions); skip greetings and repetition, " +
        "do not copy the text verbatim.\n" +
        "2. On a new line write \"【待办】\", then list this part's action items (each starting with \"-\"); " +
        "write \"None\" if there are none.\n" +
        "Write both sections entirely in $lang. Output nothing else. The part:\n\n$chunk"

/**
 * 总结 user 提示的**构造口径**：三条分别对应「全量短文本 / 抽取预热 / 逐段」三种路径。
 *
 * 默认口径的末端都锚定"与原文（或摘录）相同的语言"——单语纪要这样最省心，
 * 但跨语双语纪要必须换成 [bilingualUserTemplate]：那句会让两份纪要一起倒向
 * 内容里占比更高的语种，等于把"分别生成两个语种"这件事直接废掉。
 */
data class SummaryUserTemplate(
    val wholeDoc: (contentDesc: String, content: String) -> String = ::wholeDocSummaryUser,
    val extractive: (contentDesc: String, excerpt: List<String>) -> String = ::extractiveSummaryUser,
    val segment: (contentDesc: String, chunk: String, index: Int, total: Int) -> String = ::segmentSummaryUser
)

/** 短文本（单次推理可覆盖）的一次性总结提示。 */
internal fun wholeDocSummaryUser(contentDesc: String, content: String): String =
    "以下是${contentDesc}的完整文本。请对其做高度提炼的总结：\n\n$content"

/** 逐段总结的每段要点条数上限：压住单段输出长度（逐段独立模式下速度的主要来源）。 */
internal const val SEG_MAX_BULLETS = 6

/** 大纲话题数上限：够覆盖全文话题，又只花几十个 token（可见但便宜的"思考过程"）。 */
internal const val OUTLINE_MAX_TOPICS = 6

/**
 * 抽取预热路径的提示：**关键句摘录 → 单次的「大纲 + 总结 + 待办」**。
 *
 * 摘录按原文顺序给出、覆盖开头到结尾，等于把"读全文"这件事从 LLM 身上卸掉：
 * 模型只需重组与表达，不必回忆（遗漏与编造都大幅减少），因此**一次推理**就够。
 * 三段锚点（【大纲】/【总结】/【待办】）供流式切分，UI 才能分栏实时展示。
 */
internal fun extractiveSummaryUser(contentDesc: String, excerpt: List<String>): String {
    val body = excerpt.mapIndexed { i, s -> "${i + 1}. $s" }.joinToString("\n")
    return "以下是${contentDesc}全文的关键句摘录（共 ${excerpt.size} 句，按原文顺序，覆盖开头到结尾）：\n" +
        "$body\n\n" +
        "请基于这些关键句输出总结，严格按下面三段格式，不要输出其他内容：\n" +
        "【大纲】\n" +
        "先用最多 $OUTLINE_MAX_TOPICS 个话题词概括全文涉及的主要话题，用顿号分隔，一行写完。\n" +
        "【总结】\n" +
        "先用一两句话概括整体主题与背景；再用 - 开头的要点列表展开关键内容（合并同类话题，" +
        "保留人名、数字、时间、结论等关键信息）。\n" +
        "【待办】\n" +
        "列出需要跟进的行动项，每条以 - 开头；没有待办就只写「无」。\n" +
        "使用与摘录相同的语言，不要复述上面的指令。"
}

/**
 * 逐段独立总结的单段提示：只看本段、只输出「要点 + 【待办】」两部分。
 * 没有任何跨段上下文 → prompt 最短、输出最短，每段推理最快；【待办】是流式切分锚点，措辞固定。
 */
internal fun segmentSummaryUser(contentDesc: String, chunk: String, index: Int, total: Int): String =
    "以下是${contentDesc}的第 $index/$total 段。请单独总结这一段，只输出两部分：\n" +
        "1. 要点：最多 $SEG_MAX_BULLETS 条，每条以 - 开头，概括本段关键信息（人物、事件、数字、结论），" +
        "忽略寒暄与重复；\n" +
        "2. 另起一行写「【待办】」，在其下列出本段提到的待办事项或需要跟进的行动（每条以 - 开头）；" +
        "没有待办就写「【待办】无」。\n" +
        "不要输出其他任何内容。内容如下：\n\n$chunk"

/**
 * 把单段生成的流式文本切成（要点, 待办）两部分：以「【待办】」行为锚点。
 *
 * 两个保证（增量分发依赖）：
 * 1. 无锚点时把**结尾处未写完的锚点前缀**从要点里剥掉——否则「…- A\n【待」会被当要点发出去，
 *    下一 tick 锚点闭合后要点反而变短，增量流倒退。剥掉后要点单调不减。
 * 2. 模型没按格式输出锚点（如漏写、写成「待办：」）时自然退化：全部归要点，待办为空，不丢内容。
 */
internal fun splitPointsTodo(text: String): Pair<String, String> {
    val (points, rest) = splitAtMarker(text, "【待办】")
    if (rest == null) return points to ""
    val todos = if (rest == "无" || rest == "无待办") "" else rest
    return points to todos
}

/**
 * 把流式文本切成「锚点前 / 锚点后」两部分（三段式输出的通用切分）。
 *
 * 两个保证（增量分发依赖）：
 * 1. 无锚点时把**结尾处未写完的锚点前缀**从前半剥掉——否则「…话题A\n【总」会被当前半段发出去，
 *    下一 tick 锚点闭合后前半反而变短，增量流倒退。剥掉后前半单调不减。
 * 2. 模型没按格式输出锚点时自然退化：[rest] 为 null，调用方按自己的策略兜底，不丢内容。
 *
 * @return (锚点前文本, 锚点后文本)；锚点尚未出现时后半为 null。
 */
internal fun splitAtMarker(text: String, marker: String): Pair<String, String?> {
    val mi = text.indexOf(marker)
    if (mi >= 0) {
        return stripHeadLabel(text.substring(0, mi)).trimEnd() to
            text.substring(mi + marker.length).trimStart()
    }
    // 锚点尚未出现：剥掉结尾可能正在生成的锚点前缀（"【"、"【待"…），保证前半单调不减
    var head = text
    for (len in marker.length - 1 downTo 1) {
        if (head.endsWith(marker.substring(0, len))) { head = head.dropLast(len); break }
    }
    return stripHeadLabel(head).trimEnd() to null
}

/** 剥掉模型偶尔加在正文前的标头（「要点：」「【大纲】」之类）。 */
private fun stripHeadLabel(s: String): String =
    s.trimStart()
        .removePrefix("【要点】").removePrefix("【大纲】")
        .trimStart()
        .removePrefix("要点：").removePrefix("要点:").removePrefix("大纲：").removePrefix("大纲:")
        .trimStart()

/** 汇总持久化格式：要点节 + （有内容时的）待办节。 */
internal fun combinePointsTodo(points: String, todos: String): String {
    val p = points.trim()
    if (p.isEmpty()) return ""
    val sb = StringBuilder("【要点】\n").append(p)
    val t = todos.trim()
    if (t.isNotEmpty()) sb.append("\n\n【待办】\n").append(t)
    return sb.toString()
}
