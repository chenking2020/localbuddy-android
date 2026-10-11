package com.example.localbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 跨语沟通**双语纪要**的核心：同一场对话按语种侧各自重述一遍，
 * 才能分别整理出 A 方语种与 B 方语种两份纪要。
 *
 * 这里锁住的是输入端重述的正确性——输出端由 [bilingualSummarySystem] 强制语种，
 * 两侧结果再按槽位互不覆盖地并存（[mergeCrossSummary]）。
 */
class CrossLangSummaryTest {

    private fun turn(
        id: String, speakerId: String, name: String, src: String, transcript: String, translation: String?
    ) = CrossTurn(
        id = id, sessionId = "s1", createdAt = 1000L * id.length,
        speakerId = speakerId, speakerName = name,
        srcLangCode = src, tgtLangCode = if (src == "zh") "en" else "zh",
        transcript = transcript, translation = translation
    )

    private val dialogue = listOf(
        turn("1", "a", "张总", "zh", "这个方案预算二十万，下周三前给我答复。", "The budget is 200k, reply before next Wednesday."),
        turn("2", "b", "Lisa", "en", "I need the delivery date confirmed first.", "我需要先确认交付日期。")
    )

    @Test
    fun buildsChineseSide_fromNativeChineseAndTranslatedEnglish() {
        val content = buildLangContent(dialogue, "zh")
        // A 用自己的中文原文，B 用其中文译文 → 整篇都是中文
        assertTrue(content.contains("张总：这个方案预算二十万，下周三前给我答复。"))
        assertTrue(content.contains("Lisa：我需要先确认交付日期。"))
        // 不能把对方英文原文混进来（否则小模型会串语种）
        assertTrue("英文原文混入了中文池: $content", !content.contains("The budget is 200k"))
        assertTrue(!content.contains("I need the delivery date"))
    }

    @Test
    fun buildsEnglishSide_fromNativeEnglishAndTranslatedChinese() {
        val content = buildLangContent(dialogue, "en")
        assertTrue(content.contains("Lisa：I need the delivery date confirmed first."))
        assertTrue(content.contains("Speaker A：The budget is 200k, reply before next Wednesday."))
        assertTrue("中文原文混入了英文池: $content", !content.contains("这个方案预算二十万"))
    }

    @Test
    fun eachTurnAppearsExactlyOnce_inBothLanguages() {
        listOf("zh", "en").forEach { lang ->
            val lines = buildLangContent(dialogue, lang).split("\n")
            assertEquals("语种 $lang 应覆盖全部发言", 2, lines.size)
            assertTrue(lines.all { it.contains("：") })   // 都带说话人前缀
        }
    }

    @Test
    fun missingTranslation_isSkipped_keepsLanguagePure() {
        val noTrans = listOf(
            turn("1", "a", "张总", "zh", "预算二十万。", null),
            turn("2", "b", "Lisa", "en", "Confirm the date.", null)
        )
        // 译文缺失时**不再退化用原文**：中文原句混进英文池会把整篇英文纪要拉回中文
        val en = buildLangContent(noTrans, "en")
        assertTrue("缺译文的轮次混进了别的语种: $en", !en.contains("预算二十万"))
        assertTrue(en.contains("Confirm the date."))
    }

    @Test
    fun speakerLabel_followsTargetLanguage() {
        // 英文池里不能出现中文标签：它会把输出语言带偏
        val en = buildLangContent(dialogue, "en")
        assertTrue("中文说话人标签混进了英文池: $en", !en.contains("张总"))
        assertTrue(en.startsWith("Speaker A"))
        val zh = buildLangContent(dialogue, "zh")
        assertTrue(zh.startsWith("张总："))
    }

    @Test
    fun emptyTranscriptAndTranslation_isSkipped() {
        val blank = listOf(turn("1", "a", "张总", "zh", "", null))
        assertEquals("", buildLangContent(blank, "zh"))
    }

    @Test
    fun unknownSpeaker_getsFallbackLabel() {
        val unnamed = listOf(turn("1", "", "", "zh", "先对齐口径。", "Align first."))
        assertTrue(buildLangContent(unnamed, "zh").startsWith("说话人?："))
    }

    @Test
    fun bilingualSystem_forcesTargetLanguage_zhTargetGetsChineseInstructions() {
        val sys = bilingualSummarySystem("中文", "zh")
        assertTrue("未指定输出语种", sys.contains("全程必须使用中文输出"))
        assertTrue(sys.contains(SUMMARY_FORMAT_RULE))
        // 不能残留"跟随原文语言"的默认要求，否则双语纪约会跟语种跑
        assertTrue("残留同语言输出要求", !sys.contains("使用与录音内容相同的语言"))
    }

    @Test
    fun bilingualSystem_nonZhTarget_switchesInstructionLanguageToEnglish() {
        // 目标非中文时指令整体转英文：整段中文指令 + 一句"请用英语输出"，实测仍会输出中文
        val sys = bilingualSummarySystem("英语", "en")
        assertTrue("未指定输出语种", sys.contains("Write EVERYTHING in English"))
        assertTrue("指令语言没切到英文", !sys.contains("你是一个"))
        assertTrue(!sys.contains("使用与录音内容相同的语言"))
    }

    @Test
    fun bilingualUserTemplate_pinsTargetLanguage_andKeepsSectionAnchors() {
        val tmpl = bilingualUserTemplate("en")
        val whole = tmpl.wholeDoc("English conversation transcript", "Speaker A: hello")
        assertTrue("未钉死目标语种", whole.contains("Write EVERYTHING in English"))
        // 默认口径那句"与原文相同的语言"必须消失：它才是两份纪要都变中文的元凶
        assertTrue("残留同语言输出要求", !whole.contains("使用与录音内容相同的语言"))
        // 要概括不要复述
        assertTrue(whole.contains("do not transcribe"))

        // 三段锚点是流式切分的依据，改口径也不能丢
        val ex = tmpl.extractive("English conversation transcript", listOf("a", "b"))
        assertTrue("丢了【大纲】锚点", ex.contains("【大纲】"))
        assertTrue("丢了【总结】锚点", ex.contains("【总结】"))
        assertTrue("丢了【待办】锚点", ex.contains("【待办】"))
        val seg = tmpl.segment("English conversation transcript", "chunk", 1, 2)
        assertTrue("逐段路径丢了【待办】锚点", seg.contains("【待办】"))
    }

    @Test
    fun defaultTemplate_unchanged_stillFollowsSourceLanguage() {
        // 单语纪要口径不受影响
        assertTrue(extractiveSummaryUser("录音转写", listOf("a")).contains("使用与摘录相同的语言"))
    }

    @Test
    fun mergeCrossSummary_twoLanguagesDoNotOverwriteEachOther() {
        val afterA = mergeCrossSummary(null, "s1", 0, "中文纪要", listOf("句1", "句2"), 100L)
        assertEquals("中文纪要", afterA.textA)
        assertEquals("句1\n句2", afterA.excerptA)
        assertEquals(null, afterA.textB)

        val afterB = mergeCrossSummary(afterA, "s1", 1, "English summary", listOf("s1"), 200L)
        assertEquals("生成 B 语纪要时覆盖了 A 语", "中文纪要", afterB.textA)
        assertEquals("句1\n句2", afterB.excerptA)
        assertEquals("English summary", afterB.textB)
        assertEquals("s1", afterB.excerptB)
        assertEquals(200L, afterB.updatedAt)

        // 重新生成 A 语不影响 B 语
        val reA = mergeCrossSummary(afterB, "s1", 0, "新中文纪要", null, 300L)
        assertEquals("新中文纪要", reA.textA)
        assertEquals(null, reA.excerptA)
        assertEquals("English summary", reA.textB)
    }

    @Test
    fun mergeCrossSummary_excerptEmptyListBecomesNull() {
        val s = mergeCrossSummary(null, "s1", 0, "x", emptyList(), 1L)
        assertEquals(null, s.excerptA)
    }
}
