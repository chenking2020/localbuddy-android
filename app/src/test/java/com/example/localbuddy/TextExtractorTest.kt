package com.example.localbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 抽取器的核心不变量：
 * 1. **覆盖性**：分位配额强制头/中/尾都有句子——这正是"只保留最前边段"的对症药；
 * 2. **去冗余**：MMR 不会选出一批近义句；
 * 3. **预算**：摘录必须落在 token 预算内（否则 prompt 会超 1024 被截断）；
 * 4. **短文本不筛**：预算够就全给，信息最全。
 */
class TextExtractorTest {

    private val counter: (String) -> Int = { it.length }   // 单测用字符数近似 token

    private fun longDoc(): String = buildString {
        repeat(12) { i ->
            append("第${i + 1}段讨论的是模块$i 的实现细节，涉及接口定义与联调顺序。")
            append('\n')
        }
    }

    @Test
    fun shortText_keptVerbatim_noScreening() {
        val doc = "今天只说了两句话。\n第二句是预算 20 万。"
        val out = TextExtractor.extract(doc, tokenBudget = 500, tokenCounter = counter)
        assertEquals(TextExtractor.splitSentences(doc), out)
    }

    @Test
    fun coversHeadMiddleAndTail_notOnlyTheFront() {
        val doc = longDoc()
        val all = TextExtractor.splitSentences(doc)
        val picked = TextExtractor.select(TextExtractor.score(all), 6)
        val idx = picked.map { it.index }
        val last = all.size - 1
        assertTrue("头部缺失: $idx", idx.any { it <= 1 })
        assertTrue("中部缺失: $idx", idx.any { it in 2 until last - 1 })
        assertTrue("尾部缺失: $idx", idx.any { it >= last - 1 })
        // 输出按原文顺序
        assertEquals(idx.sorted(), idx)
    }

    @Test
    fun mmr_suppressesNearDuplicateSentences() {
        val doc = buildString {
            repeat(8) { append("服务器下周三到期，需要尽快申请续费。\n") }   // 8 句完全重复
            append("预算审批由李总负责，金额 20 万。\n")
            append("下个迭代重点是端上推理提速。")
        }
        val picked = TextExtractor.extract(doc, tokenBudget = 400, tokenCounter = counter, maxSentences = 4)
        val dup = picked.count { it.contains("服务器下周三到期") }
        assertTrue("重复句被选了 $dup 次", dup <= 2)
        assertTrue(picked.size <= 4)
    }

    @Test
    fun respectsTokenBudget() {
        val doc = longDoc()
        val out = TextExtractor.extract(doc, tokenBudget = 120, tokenCounter = counter, maxSentences = 14)
        val used = out.sumOf { it.length } + (out.size - 1).coerceAtLeast(0)
        assertTrue("超预算: $used > 120", used <= 120)
        assertTrue(out.isNotEmpty())
    }

    @Test
    fun respectsMaxSentences() {
        val out = TextExtractor.extract(longDoc(), tokenBudget = 10000, tokenCounter = counter, maxSentences = 3)
        assertEquals(3, out.size)
    }

    @Test
    fun emptyAndDegenerateInput_isSafe() {
        assertEquals(emptyList<String>(), TextExtractor.extract("", 100, counter))
        assertEquals(emptyList<String>(), TextExtractor.extract("。！！？？", 100, counter))
        // 碎片（<4 字）被切句过滤
        assertEquals(emptyList<String>(), TextExtractor.splitSentences("啊啊。哦哦。"))
    }

    @Test
    fun signalWordsAndNumbers_scoreHigher() {
        val sents = listOf(
            "今天天气不错大家聊了聊。",                 // 无信息
            "决定下周三上线，预算 20 万，李总负责。"    // 决策 + 数字 + 责任人
        )
        val scored = TextExtractor.score(sents)
        assertTrue("信号句应显著高于寒暄句: ${scored.map { it.score }}", scored[1].score > scored[0].score)
    }
}
