package com.example.localbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Qwen3 思考模式开关（对应 API 侧 enable_thinking=False）的验证。
 *
 * 两件事必须成立，否则"关闭思考"是空谈：
 * 1. `<think>`/`</think>` 是 **added token**（151667/151668），不在 base vocab 里——
 *    [HfBpeTokenizer.encode] 不查 added_tokens，把它们当文本写进 prompt 只会被 BPE 切成碎片，
 *    模型认不出思考块边界（旧实现在 user 末尾拼 "/no_think" 正是如此，完全无效 → 思考一直开着，
 *    长文本每个分块都先吐几百个思考 token，慢数倍）。
 * 2. 生成内容要在 **token id 层** 剥离思考——[HfBpeTokenizer.decode] 跳过全部 added token，
 *    解码文本里根本找不到 `</think>`，纯字符串剥离形同虚设（见 [stripThinkTokens]）。
 *
 * 注：单测不加载真实 tokenizer——main 代码用的 org.json 在 JVM 上是 android.jar 的桩（返回 null），
 * 这里改为直接校验 assets 里的 added_tokens/vocab 事实 + 纯函数行为。
 */
class LlmThinkingTest {

    private fun asset(name: String): File = listOf(
        File("src/main/assets/llm/$name"), File("app/src/main/assets/llm/$name")
    ).firstOrNull { it.exists() } ?: error("llm/$name not found")

    /** added_tokens.json 形如 {"</think>": 151668, "<think>": 151667, ...}。 */
    private fun addedTokens(): Map<String, Int> =
        Regex("\"([^\"]+)\"\\s*:\\s*(\\d+)")
            .findAll(asset("added_tokens.json").readText())
            .associate { it.groupValues[1] to it.groupValues[2].toInt() }

    @Test
    fun thinkMarkersAreAddedTokens_notInBaseVocab() {
        val added = addedTokens()
        assertEquals(151667, added["<think>"])
        assertEquals(151668, added["</think>"])
        // base vocab（vocab.json）里没有它们 → 只能按 id 拼，写进文本必然被 BPE 切碎
        // （不能用 tokenizer.json 判断：它的 added_tokens 段里也有这两个标记）
        assertFalse(asset("vocab.json").readText().contains("\"<think>\""))
    }

    @Test
    fun legacyNoThinkSuffixIsNotAToken() {
        val added = addedTokens()
        assertFalse("'/no_think' 不是词表 token：旧软开关写法无效", added.containsKey("/no_think"))
        assertFalse(added.containsKey("<arg_key:6124c78e>"))
    }

    @Test
    fun chatTemplateRendersEmptyThinkBlockWhenThinkingDisabled() {
        // 官方模板在 enable_thinking=False 时于 assistant 开头渲染空思考块——本项目的手写 prompt 必须对齐
        val tpl = asset("chat_template.jinja").readText()
        assertTrue(tpl.contains("enable_thinking is defined and enable_thinking is false"))
        assertTrue(tpl.contains("'<think>\\n\\n</think>\\n\\n'"))
    }

    @Test
    fun stripThinkTokens_keepsOnlyBodyAfterClosedBlock() {
        assertEquals(
            listOf(20, 21),
            stripThinkTokens(listOf(151667, 10, 11, 151668, 20, 21), 151667, 151668)
        )
    }

    @Test
    fun stripThinkTokens_emptyWhileStillThinking() {
        assertTrue(stripThinkTokens(listOf(151667, 10, 11), 151667, 151668).isEmpty())
    }

    @Test
    fun stripThinkTokens_passthroughWithoutThinkMarkers() {
        val ids = listOf(1, 2, 3)
        assertEquals(ids, stripThinkTokens(ids, 151667, 151668))
        assertEquals(ids, stripThinkTokens(ids, 0, 0))   // 非 Qwen 词表：不剥离
    }
}
