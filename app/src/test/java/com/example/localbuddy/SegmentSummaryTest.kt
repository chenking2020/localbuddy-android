package com.example.localbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 逐段独立总结（segmented summarization）的提示词与流式切分验证。
 *
 * 关键不变量：
 * 1. 每段提示**不携带**任何跨段上下文（速度与低漂移的来源）；
 * 2. 「【待办】」是流式切分锚点——生成中途就要正确切分（待办行只写了一半时不能误当要点）；
 * 3. 要点流**单调不减**：增量分发（delta = 新 - 旧）依赖这一点，倒退会导致 UI 丢字或重复。
 */
class SegmentSummaryTest {

    private val chunk = "张工说服务器周三到期。\n李总确认预算 20 万。"

    @Test
    fun segmentPrompt_isSelfContained() {
        val p = segmentSummaryUser("录音转写", chunk, 3, 9)
        assertTrue(p.contains(chunk))
        assertTrue(p.contains("第 3/9 段"))
        assertTrue(p.contains("【待办】"))
        assertTrue(p.contains("$SEG_MAX_BULLETS 条"))
        // 不携带跨段上下文：不得出现「已整理/要点：」之类的累积态措辞
        assertTrue(!p.contains("已整理"))
    }

    @Test
    fun split_pointsBeforeMarkerTodosAfter() {
        val (p, t) = splitPointsTodo("- 服务器周三到期\n- 预算 20 万\n【待办】\n- 申请续费")
        assertEquals("- 服务器周三到期\n- 预算 20 万", p)
        assertEquals("- 申请续费", t)
    }

    @Test
    fun split_noTodoWord_yieldsEmptyTodo() {
        val (p, t) = splitPointsTodo("- 要点一\n【待办】无")
        assertEquals("- 要点一", p)
        assertEquals("", t)
    }

    @Test
    fun split_missingMarker_degradesToAllPoints() {
        val (p, t) = splitPointsTodo("- 只有要点，模型漏了标记")
        assertEquals("- 只有要点，模型漏了标记", p)
        assertEquals("", t)
    }

    @Test
    fun split_stripsPointsLabelHeader() {
        val (p, t) = splitPointsTodo("要点：\n- A\n【待办】\n- B")
        assertEquals("- A", p)
        assertEquals("- B", t)
    }

    @Test
    fun split_halfWrittenMarker_neverLeaksIntoPoints_andPointsStayMonotonic() {
        // 模拟流式 tick 序列：锚点逐字生成中 → 闭合 → 待办增长
        val ticks = listOf(
            "- A\n",
            "- A\n【",
            "- A\n【待",
            "- A\n【待办",
            "- A\n【待办】",
            "- A\n【待办】\n",
            "- A\n【待办】\n- T"
        )
        var lastPointsLen = 0
        val seenPoints = StringBuilder()
        val seenTodos = StringBuilder()
        var lastTodosLen = 0
        ticks.forEach { text ->
            val (p, t) = splitPointsTodo(text)
            // 要点单调不减，且增量恰好无重复无丢失地拼回原文
            if (p.length > lastPointsLen) {
                seenPoints.append(p, lastPointsLen, p.length)
                lastPointsLen = p.length
            }
            if (t.length > lastTodosLen) {
                seenTodos.append(t, lastTodosLen, t.length)
                lastTodosLen = t.length
            }
        }
        // 锚点半成品绝不能漏进要点流
        assertTrue("锚点片段泄漏进要点: $seenPoints", !seenPoints.contains("【"))
        assertEquals("- A", seenPoints.toString())
        assertEquals("- T", seenTodos.toString())
    }

    @Test
    fun extractivePrompt_carriesWholeExcerpt_andThreeMarkers() {
        val excerpt = listOf("第一句讲预算", "中间句讲排期", "结尾句讲风险")
        val p = extractiveSummaryUser("录音转写", excerpt)
        excerpt.forEach { assertTrue(p.contains(it)) }
        assertTrue(p.contains("【大纲】") && p.contains("【总结】") && p.contains("【待办】"))
        assertTrue(p.contains("覆盖开头到结尾"))
    }

    @Test
    fun splitAtMarker_headBeforeTailAfter_andStripsHeadLabel() {
        val (head, rest) = splitAtMarker("【大纲】话题A、话题B\n【总结】正文", "【总结】")
        assertEquals("话题A、话题B", head)
        assertEquals("正文", rest)
    }

    @Test
    fun splitAtMarker_halfWrittenMarker_notInHead() {
        val (head, rest) = splitAtMarker("话题A、话题B\n【总", "【总结】")
        assertEquals("话题A、话题B", head)
        assertEquals(null, rest)
    }

    @Test
    fun threeStageStream_outlinePointsTodosSplitCleanly_andMonotonic() {
        // 模拟一次生成里的流式 tick：大纲 → 【总结】 → 要点 → 【待办】 → 待办，锚点逐字出现
        val ticks = listOf(
            "【大纲】话",
            "【大纲】话题A、话题B\n",
            "【大纲】话题A、话题B\n【总",
            "【大纲】话题A、话题B\n【总结】",
            "【大纲】话题A、话题B\n【总结】主题是一件事\n- 要点1",
            "【大纲】话题A、话题B\n【总结】主题是一件事\n- 要点1\n【待办",
            "【大纲】话题A、话题B\n【总结】主题是一件事\n- 要点1\n【待办】\n- 待办1"
        )
        val outline = StringBuilder(); val points = StringBuilder(); val todos = StringBuilder()
        var lo = 0; var lp = 0; var lt = 0
        ticks.forEach { text ->
            val (head, rest) = splitAtMarker(text, "【总结】")
            if (head.length > lo) { outline.append(head, lo, head.length); lo = head.length }
            rest ?: return@forEach
            val (p, t) = splitPointsTodo(rest)
            if (p.length > lp) { points.append(p, lp, p.length); lp = p.length }
            if (t.length > lt) { todos.append(t, lt, t.length); lt = t.length }
        }
        assertEquals("话题A、话题B", outline.toString())
        assertEquals("主题是一件事\n- 要点1", points.toString())
        assertEquals("- 待办1", todos.toString())
        assertTrue("锚点泄漏: ${outline}/${points}", !outline.contains("【") && !points.contains("【"))
    }

    @Test
    fun combine_persistsPointsAndTodoSections() {
        assertEquals(
            "【要点】\n- A\n\n【待办】\n- B",
            combinePointsTodo("- A\n", "- B")
        )
        assertEquals("【要点】\n- A", combinePointsTodo(" - A ", ""))
        assertEquals("", combinePointsTodo("", "- B"))
    }
}
