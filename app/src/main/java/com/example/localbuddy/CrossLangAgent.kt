package com.example.localbuddy

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

// ── 品牌色 ──
private val CrossTeal = Color(0xFF0D9488)   // 跨语沟通智能体主色
private val SPK_A = Color(0xFFF97316)       // 说话人A（橙）
private val SPK_B = Color(0xFF0EA5E9)       // 说话人B（青）
private val RecRed = Color(0xFFEF4444)      // 录音中红点

/**
 * 官网支持语种（19 种，见 [supportedLangs]）：跨语沟通双方语种配置的可选集。
 * TTS 跨语种音色克隆已实测：音色参考帧仅决定音色，克隆自说话人本人的录音，
 * 中文音色的说话人可直接说出其他 18 种语种。
 */

/** 沟通一方：名字 + 语种（播报音色自动克隆自其本人的说话语音，无需配置）。 */
data class Party(
    val id: String,
    val name: String,
    val langCode: String
)

private fun defaultParties() = listOf(
    Party("a", "说话人A", "zh"),
    Party("b", "说话人B", "en")
)

/** 一段发言的处理阶段（live 转写/翻译/合成/播报；持久化记录恒为 DONE）。 */
enum class TurnPhase { TRANSCRIBING, TRANSLATING, SYNTHESIZING, PLAYING, DONE, FAILED }

/** 一段发言：说话语音 + 转写 + 译文（对方语种）+ 播报语音。 */
data class CrossTurn(
    val id: String,
    val sessionId: String,
    val createdAt: Long,
    val speakerId: String,          // ASR 语种判定归属（转写完成前为 ""）
    val speakerName: String,        // 归属判定后回填
    val srcLangCode: String,
    val tgtLangCode: String,
    val transcript: String,
    val translation: String? = null,
    val speechPath: String? = null,    // 原音 16k mono wav
    val broadcastPath: String? = null, // 播报 48k stereo wav
    val phase: TurnPhase = TurnPhase.DONE
) {
    val timeLabel: String
        get() = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(createdAt))
}

/** 会话（一次连续沟通的多段发言）。 */
data class CrossSession(
    val id: String,
    val startedAt: Long,
    val turns: List<CrossTurn>
)

/**
 * 一次会话的**双语纪要**：同一场对话分别整理成 A 方语种与 B 方语种两份，
 * 每份附关键句摘录（[excerptA]/[excerptB]，换行分隔）便于核对来源。
 *
 * 两语种**分别存储**：生成也分两次跑（各自语种独立 prompt），
 * 所以可以先出 A 语版给用户看，B 语版稍后补上，互不覆盖。
 */
data class CrossSessionSummary(
    val sessionId: String,
    val textA: String? = null,
    val excerptA: String? = null,
    val textB: String? = null,
    val excerptB: String? = null,
    val updatedAt: Long = 0L
)

/** 纪要目标：只生成 A 语种 / 只生成 B 语种 / 两种都生成。 */
enum class SummaryTarget { A, B, BOTH }

/** 单语纪要的 UI 状态：生成中的三股流式内容 + 已完成存档。 */
data class LangSummaryState(
    val langCode: String = "",
    val langName: String = "",
    val outline: String = "",               // 流式大纲（可见的"思考过程"）
    val points: String = "",                // 流式要点
    val todos: String = "",                 // 流式待办
    val excerpt: List<String> = emptyList(),// 抽取式关键句（即时预览 → 完成后折叠附件）
    val saved: String? = null,              // 已持久化的纪要正文
    val savedExcerpt: List<String> = emptyList()
) {
    /** 生成中是否已产出内容（决定 UI 显示流还是存档）。 */
    val streaming: Boolean get() = outline.isNotBlank() || points.isNotBlank() || todos.isNotBlank()
}

/**
 * 说话人前缀：**跟随目标语种**——英文池里出现「张总：」这种中文标签，等于往纯净的
 * 英文内容里掺中文，小模型会顺着它把纪要写回中文。
 */
private fun CrossTurn.speakerLabel(langCode: String): String {
    val who = speakerId.takeIf { it.isNotBlank() }?.uppercase() ?: "?"
    val named = speakerName.takeIf { it.isNotBlank() }
    return if (zhInstruction(langCode)) {
        named ?: "说话人$who"
    } else {
        named?.takeIf { it.all { c -> c.code < 0x2E80 } } ?: "Speaker $who"
    }
}

/**
 * 该轮发言在**指定语种**下的文本：谁的原文就是该语种 → 取原文；否则取其译文（对方听到的版本）。
 *
 * 译文缺失时**直接跳过这一轮**，不再退化用原文：语种纯净优先于信息完整——
 * 一句中文混进英文池，端侧小模型就会把整篇英文纪要写回中文。
 */
private fun CrossTurn.langVersion(langCode: String): String? =
    if (srcLangCode == langCode) transcript.takeIf { it.isNotBlank() }
    else translation?.takeIf { it.isNotBlank() }

/**
 * 按目标语种把整场会话**重述成一篇该语种的对话文本**（供总结模型阅读）。
 *
 * 这是双语纪要的关键：跨语沟通里 A 说中文、B 说英文，若直接把混合原文丢给模型，
 * 小模型既读不懂也容易串语种；这里按语种"翻译"一遍输入端，模型只需输出同一语种，
 * 两语种各跑一次即得两份各自语种的纪要。结果带说话人前缀，纪要才能区分是谁说的。
 */
internal fun buildLangContent(turns: List<CrossTurn>, langCode: String): String =
    turns.mapNotNull { t ->
        val text = t.langVersion(langCode) ?: return@mapNotNull null
        "${t.speakerLabel(langCode)}：$text"
    }.joinToString("\n")

/** 把某一语种的纪要并进会话档案，另一语种保持原样（两次生成互不覆盖）。 */
internal fun mergeCrossSummary(
    old: CrossSessionSummary?,
    sessionId: String,
    slot: Int,                 // 0 = A 方语种，1 = B 方语种
    text: String,
    excerpt: List<String>?,
    now: Long
): CrossSessionSummary {
    val base = old ?: CrossSessionSummary(sessionId = sessionId)
    val ex = excerpt?.joinToString("\n")?.takeIf { it.isNotBlank() }
    return base.copy(
        textA = if (slot == 0) text else base.textA,
        excerptA = if (slot == 0) ex else base.excerptA,
        textB = if (slot == 1) text else base.textB,
        excerptB = if (slot == 1) ex else base.excerptB,
        updatedAt = now
    )
}

enum class TalkStage { IDLE, RECORDING, PROCESSING, PLAYING }

data class CrossUiState(
    val parties: List<Party> = defaultParties(),
    val turns: List<CrossTurn> = emptyList(),     // 当前会话发言（时间正序）
    val stage: TalkStage = TalkStage.IDLE,
    val recMs: Long = 0L,                          // 录音计时（live）
    val statusText: String = "",
    val error: String? = null,
    val configOpen: Boolean = false,
    val historyOpen: Boolean = false,
    val loadingTts: Boolean = false,
    val ttsFrames: Int = 0,                         // 合成进度（已生成帧数）
    // ── 历史内「会话详情 / 双语纪要」两个标签页 ──
    val historyDetailId: String? = null,            // 历史里展开的会话；纪要就针对它生成
    val historyTab: Int = 0,                        // 0 = 会话详情，1 = 双语纪要
    val summaryTab: Int = 0,                        // 纪要面板内：0 = A 方语种，1 = B 方语种
    val summarizing: Boolean = false,
    val summaryLoadingLlm: Boolean = false,
    val summaryStatus: String = "",                 // 纪要进度文案（与录音 statusText 分开，互不冲）
    val summaryA: LangSummaryState = LangSummaryState(),
    val summaryB: LangSummaryState = LangSummaryState(),
    val summaries: Map<String, CrossSessionSummary> = emptyMap()   // sessionId → 双语纪要存档
)

// ──────────────────────────────────────────────────────────────────
// 持久化：发言记录 JSON + 音频文件（filesDir/cross_talk/）
// ──────────────────────────────────────────────────────────────────

private class CrossTalkStore(private val context: Context) {
    val dir: File get() = File(context.filesDir, "cross_talk").apply { mkdirs() }
    private val historyFile: File get() = File(context.filesDir, "cross_talk_history.json")
    private val configFile: File get() = File(context.filesDir, "cross_talk_config.json")
    private val summaryFile: File get() = File(context.filesDir, "cross_talk_summaries.json")

    fun loadConfig(): List<Party>? = runCatching {
        if (!configFile.exists()) return null
        val arr = JSONArray(configFile.readText())
        (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            // 兼容旧配置：gender 字段已废弃（音色改为克隆本人语音）
            Party(
                id = o.getString("id"),
                name = o.getString("name"),
                langCode = o.getString("langCode")
            )
        }.ifEmpty { null }
    }.getOrNull()

    fun saveConfig(parties: List<Party>) {
        val arr = JSONArray()
        parties.forEach {
            arr.put(JSONObject()
                .put("id", it.id)
                .put("name", it.name)
                .put("langCode", it.langCode))
        }
        configFile.writeText(arr.toString())
    }

    fun loadHistory(): List<CrossTurn> = runCatching {
        if (!historyFile.exists()) return emptyList()
        val arr = JSONArray(historyFile.readText())
        (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            CrossTurn(
                id = o.getString("id"),
                sessionId = o.getString("sessionId"),
                createdAt = o.getLong("createdAt"),
                speakerId = o.getString("speakerId"),
                speakerName = o.optString("speakerName", ""),
                srcLangCode = o.getString("srcLangCode"),
                tgtLangCode = o.getString("tgtLangCode"),
                transcript = o.getString("transcript"),
                translation = o.optString("translation").takeIf { it.isNotEmpty() },
                speechPath = o.optString("speechPath").takeIf { it.isNotEmpty() },
                broadcastPath = o.optString("broadcastPath").takeIf { it.isNotEmpty() }
            )
        }
    }.getOrElse { emptyList() }

    fun saveHistory(turns: List<CrossTurn>) {
        val arr = JSONArray()
        turns.forEach { t ->
            arr.put(JSONObject()
                .put("id", t.id)
                .put("sessionId", t.sessionId)
                .put("createdAt", t.createdAt)
                .put("speakerId", t.speakerId)
                .put("speakerName", t.speakerName)
                .put("srcLangCode", t.srcLangCode)
                .put("tgtLangCode", t.tgtLangCode)
                .put("transcript", t.transcript)
                .put("translation", t.translation ?: "")
                .put("speechPath", t.speechPath ?: "")
                .put("broadcastPath", t.broadcastPath ?: ""))
        }
        historyFile.writeText(arr.toString())
    }

    /** 双语纪要档案：sessionId → CrossSessionSummary。 */
    fun loadSummaries(): Map<String, CrossSessionSummary> = runCatching {
        if (!summaryFile.exists()) return emptyMap()
        val arr = JSONArray(summaryFile.readText())
        (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val sid = o.getString("sessionId")
            sid to CrossSessionSummary(
                sessionId = sid,
                textA = o.optString("textA").takeIf { it.isNotEmpty() },
                excerptA = o.optString("excerptA").takeIf { it.isNotEmpty() },
                textB = o.optString("textB").takeIf { it.isNotEmpty() },
                excerptB = o.optString("excerptB").takeIf { it.isNotEmpty() },
                updatedAt = o.optLong("updatedAt", 0L)
            )
        }.toMap()
    }.getOrElse { emptyMap() }

    fun saveSummaries(map: Map<String, CrossSessionSummary>) {
        val arr = JSONArray()
        map.values.forEach { s ->
            arr.put(JSONObject()
                .put("sessionId", s.sessionId)
                .put("textA", s.textA ?: "")
                .put("excerptA", s.excerptA ?: "")
                .put("textB", s.textB ?: "")
                .put("excerptB", s.excerptB ?: "")
                .put("updatedAt", s.updatedAt))
        }
        summaryFile.writeText(arr.toString())
    }
}

// ──────────────────────────────────────────────────────────────────
// ViewModel：开始-停止 分段录音 → ASR（语种判定归属）→ 译成对方语种 → TTS 播报
// ──────────────────────────────────────────────────────────────────

class CrossLangViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = SpeechMTEngine.get(app)
    private val tts = TtsEngine.get(app)
    private val llm = LlmEngine.get(app)
    private val store = CrossTalkStore(app)

    // 长停顿不切分（点击「停止」才算一段），8s 连续说话强制切块后拼接（VAD 静音剔除利于 ASR）
    private val recorder = MicRecorder(pauseMs = 60_000, maxSegMs = 8_000)

    private val _state = MutableStateFlow(CrossUiState())
    val state: StateFlow<CrossUiState> = _state.asStateFlow()

    private var history: MutableList<CrossTurn> = mutableListOf()
    private val summaries = LinkedHashMap<String, CrossSessionSummary>()

    private var recJob: Job? = null
    private var timerJob: Job? = null
    private var summaryJob: Job? = null
    private val recBuf = ArrayList<FloatArray>()
    private var player: MediaPlayer? = null

    init {
        viewModelScope.launch { engine.warmUp() }
        store.loadConfig()?.let { _state.update { s -> s.copy(parties = it) } }
        history = store.loadHistory().toMutableList()
        summaries.putAll(store.loadSummaries())
        _state.update { it.copy(summaries = summaries.toMap()) }
    }

    fun toggleConfig() = _state.update {
        it.copy(configOpen = !it.configOpen, historyOpen = false, historyDetailId = null, historyTab = 0)
    }

    /** 关闭历史面板时顺手停掉纪要生成：面板不可见了，没必要继续占着端侧算力。 */
    fun toggleHistory() {
        val open = !_state.value.historyOpen
        if (!open) {
            summaryJob?.cancel()
            summaryJob = null
            _state.update { it.copy(summarizing = false, summaryStatus = "") }
        }
        _state.update {
            it.copy(
                historyOpen = open, configOpen = false,
                historyDetailId = null, historyTab = 0, summaryTab = 0
            )
        }
    }

    /** 展开某个历史会话；[tab] = 0 会话详情 / 1 双语纪要。 */
    fun openSession(sessionId: String, tab: Int = 0) {
        val saved = summaries[sessionId]
        val parties = _state.value.parties
        _state.update { s ->
            s.copy(
                historyOpen = true, configOpen = false, historyDetailId = sessionId,
                historyTab = tab, summaryTab = 0, error = null,
                // 重新展开时丢掉上一次未完成的流式内容，回落到已存档结果
                summaryA = langSummaryState(parties.getOrNull(0), saved?.textA, saved?.excerptA),
                summaryB = langSummaryState(parties.getOrNull(1), saved?.textB, saved?.excerptB)
            )
        }
    }

    /** 会话详情 ↔ 双语纪要 两个标签页切换。 */
    fun selectHistoryTab(i: Int) = _state.update { it.copy(historyTab = i) }

    /** 返回历史列表（不中断生成：切回纪要页进度还在）。 */
    fun closeSession() = _state.update { it.copy(historyDetailId = null, historyTab = 0, summaryTab = 0) }

    fun selectSummaryTab(i: Int) = _state.update { it.copy(summaryTab = i) }

    fun updateParty(id: String, name: String? = null, langCode: String? = null) {
        _state.update { s ->
            s.copy(parties = s.parties.map {
                if (it.id == id) it.copy(
                    name = name ?: it.name,
                    langCode = langCode ?: it.langCode
                ) else it
            })
        }
        store.saveConfig(_state.value.parties)
    }

    /** 历史会话（按 sessionId 聚合，倒序）。 */
    fun historySessions(): List<CrossSession> =
        history.groupBy { it.sessionId }
            .map { (sid, turns) -> CrossSession(sid, turns.minOf { it.createdAt }, turns.sortedBy { it.createdAt }) }
            .sortedByDescending { it.startedAt }

    fun deleteSession(sessionId: String) {
        val doomed = history.filter { it.sessionId == sessionId }
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                doomed.forEach { t ->
                    t.speechPath?.let { runCatching { File(it).delete() } }
                    t.broadcastPath?.let { runCatching { File(it).delete() } }
                }
            }
            history.removeAll { it.sessionId == sessionId }
            store.saveHistory(history)
            summaries.remove(sessionId)
            store.saveSummaries(summaries)
            // 当前屏若正显示该会话发言/纪要，一并清掉
            _state.update { s ->
                s.copy(
                    turns = s.turns.filterNot { it.sessionId == sessionId },
                    summaries = summaries.toMap(),
                    historyDetailId = s.historyDetailId?.takeIf { it != sessionId }
                )
            }
        }
    }

    // ── 双语纪要 ──

    /** 会话的全部发言：纪要只针对**该历史会话**记录下来的内容，与当前屏无关。 */
    private fun sessionTurns(sessionId: String): List<CrossTurn> =
        history.filter { it.sessionId == sessionId }.sortedBy { it.createdAt }

    private fun langSummaryState(party: Party?, text: String?, excerpt: String?) =
        LangSummaryState(
            langCode = party?.langCode ?: "",
            langName = langName(party?.langCode ?: ""),
            saved = text,
            savedExcerpt = excerptLines(excerpt)
        )

    private inline fun mutateSlot(slot: Int, f: (LangSummaryState) -> LangSummaryState) =
        _state.update { s ->
            if (slot == 0) s.copy(summaryA = f(s.summaryA)) else s.copy(summaryB = f(s.summaryB))
        }

    /**
     * 生成历史里这场沟通的纪要：[target] 选 A 语种 / B 语种 / 双语都生成。
     *
     * 双语是**两次串行推理**（端侧单 session 不能并行）：先把目标语种的内容池（见 [buildLangContent]）
     * 交给抽取 + 单次流式生成（两天前定的那条主路径），A 语种先整份出完再跑 B 语种，
     * 用户能一路看着文字长出来，而不是干等两个语种一起出。
     */
    fun summarizeSession(target: SummaryTarget) {
        val sid = _state.value.historyDetailId ?: return
        val turns = sessionTurns(sid)
        if (turns.isEmpty()) {
            _state.update { it.copy(error = "该会话暂无可总结的发言") }
            return
        }
        summaryJob?.cancel()
        summaryJob = viewModelScope.launch {
            try {
                _state.update { it.copy(summarizing = true, error = null, summaryStatus = "纪要生成中…") }
                if (!llm.isLoaded) {
                    _state.update {
                        it.copy(summaryLoadingLlm = true, summaryStatus = "正在加载纪要模型…")
                    }
                    llm.warmUp()
                    _state.update { it.copy(summaryLoadingLlm = false) }
                }
                when (target) {
                    SummaryTarget.A -> summarizeSlot(sid, 0, turns)
                    SummaryTarget.B -> summarizeSlot(sid, 1, turns)
                    SummaryTarget.BOTH -> { summarizeSlot(sid, 0, turns); summarizeSlot(sid, 1, turns) }
                }
                _state.update { it.copy(summarizing = false, summaryStatus = "") }
            } catch (ce: CancellationException) {
                _state.update { it.copy(summaryStatus = "") }
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "生成跨语纪要失败", t)
                _state.update {
                    it.copy(summarizing = false, summaryStatus = "", error = "纪要生成失败：${t.message}")
                }
            }
        }
    }

    /** 生成单个语种的纪要（0 = A 方语种，1 = B 方语种）。 */
    private suspend fun summarizeSlot(sessionId: String, slot: Int, turns: List<CrossTurn>) {
        val party = _state.value.parties.getOrNull(slot) ?: return
        val name = langName(party.langCode).ifBlank { party.langCode }
        val content = buildLangContent(turns, party.langCode)
        if (content.isBlank()) {
            // 该语种下没有可用文本（对方的译文缺失）→ 宁可不生成，也不产出语种不对的纪要
            Log.w(TAG, "语种 ${party.langCode} 无可总结内容（译文缺失），跳过该语种纪要")
            return
        }
        // 重新生成：先清掉该语种的流式缓冲（不影响另一语种）
        mutateSlot(slot) { it.copy(outline = "", points = "", todos = "", excerpt = emptyList()) }
        _state.update { it.copy(summaryTab = slot, summaryStatus = "正在整理${name}纪要…") }
        val text = llm.summarizeSmart(
            content,
            system = bilingualSummarySystem(name, party.langCode),
            contentDesc = bilingualContentDesc(party.langCode, name),
            userTemplate = bilingualUserTemplate(party.langCode),
            maxNew = 768,
            onPreview = { sent -> mutateSlot(slot) { it.copy(excerpt = sent) } },
            onOutline = { d -> mutateSlot(slot) { it.copy(outline = it.outline + d) } },
            onPartial = { d -> mutateSlot(slot) { it.copy(points = it.points + d) } },
            onTodo = { d -> mutateSlot(slot) { it.copy(todos = it.todos + d) } }
        )
        if (text.isNotBlank()) saveSummary(sessionId, slot, text)
    }

    private fun saveSummary(sessionId: String, slot: Int, text: String) {
        val st = if (slot == 0) _state.value.summaryA else _state.value.summaryB
        val merged = mergeCrossSummary(
            summaries[sessionId], sessionId, slot, text, st.excerpt, System.currentTimeMillis()
        )
        summaries[sessionId] = merged
        store.saveSummaries(summaries)
        _state.update { it.copy(summaries = summaries.toMap()) }
        mutateSlot(slot) { it.copy(saved = text, savedExcerpt = it.excerpt) }
    }

    /** 停止生成：已产出的部分（含只生成完一侧的情况）照常存档。 */
    fun stopSummary() {
        summaryJob?.cancel()
        summaryJob = null
        val sid = _state.value.historyDetailId
        if (sid != null) {
            listOf(0, 1).forEach { slot ->
                val st = if (slot == 0) _state.value.summaryA else _state.value.summaryB
                val partial = combinePointsTodo(st.points, st.todos).trim()
                if (partial.isNotBlank()) saveSummary(sid, slot, partial)
            }
        }
        _state.update { it.copy(summarizing = false, summaryStatus = "") }
    }

    /** 开始录音（一段）。 */
    fun start() {
        val ctx = getApplication<Application>()
        if (!ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO).let { it == PackageManager.PERMISSION_GRANTED }) {
            _state.update { it.copy(error = "缺少录音权限") }
            return
        }
        if (_state.value.stage != TalkStage.IDLE) return
        stopPlayback()
        recBuf.clear()
        _state.update { it.copy(stage = TalkStage.RECORDING, recMs = 0, statusText = "录音中，说完点「停止」", error = null) }
        timerJob = viewModelScope.launch {
            val t0 = System.currentTimeMillis()
            while (isActive && _state.value.stage == TalkStage.RECORDING) {
                _state.update { it.copy(recMs = System.currentTimeMillis() - t0) }
                delay(200)
            }
        }
        recJob = viewModelScope.launch(Dispatchers.IO) {
            runCatching { recorder.stream().collect { chunk -> synchronized(recBuf) { recBuf.add(chunk) } } }
                .onFailure { Log.e(TAG, "录音异常", it) }
            timerJob?.cancel()
            processSegment()
        }
    }

    /** 停止录音 → 触发本段转写/翻译/合成/播报。 */
    fun stop() {
        if (_state.value.stage != TalkStage.RECORDING) return
        _state.update { it.copy(stage = TalkStage.PROCESSING, statusText = "处理中…") }
        recorder.stop()   // 采集流退出 → recJob 进入 processSegment
    }

    /** 一段语音的完整处理：ASR → 语种判定归属 → 译成对方语种 → TTS → 自动播报。 */
    private suspend fun processSegment() = withContext(Dispatchers.IO) {
        val pcm = synchronized(recBuf) {
            val total = recBuf.sumOf { it.size }
            val out = FloatArray(total)
            var off = 0
            recBuf.forEach { System.arraycopy(it, 0, out, off, it.size); off += it.size }
            recBuf.clear()
            out
        }
        if (pcm.size < 16_000 * 3 / 10) {
            _state.update { it.copy(stage = TalkStage.IDLE, statusText = "",
                error = "未录到语音：说话时间太短或麦克风初始化失败") }
            return@withContext
        }
        if (!engine.isLoaded) {
            _state.update { it.copy(stage = TalkStage.PROCESSING, statusText = "模型加载中…") }
            engine.warmUp()
        }

        val turnId = UUID.randomUUID().toString()
        // 新会话：无历史发言或距上段超 30 分钟
        val lastAt = history.maxOfOrNull { it.createdAt } ?: 0L
        val sessionId = if (System.currentTimeMillis() - lastAt > 30 * 60_000L) "s${System.currentTimeMillis()}" else null
        val sid = sessionId ?: history.lastOrNull()?.sessionId
            ?: _state.value.turns.lastOrNull()?.sessionId
            ?: "s${System.currentTimeMillis()}"

        val speechFile = File(store.dir, "seg_$turnId.wav")
        WavIo.writeWav(speechFile.absolutePath, pcm)

        _state.update {
            it.copy(
                stage = TalkStage.PROCESSING, statusText = "转写中…", error = null,
                turns = it.turns + CrossTurn(
                    id = turnId, sessionId = sid, createdAt = System.currentTimeMillis(),
                    speakerId = "", speakerName = "", srcLangCode = "", tgtLangCode = "",
                    transcript = "", speechPath = speechFile.absolutePath,
                    phase = TurnPhase.TRANSCRIBING
                )
            )
        }
        fun patch(f: (CrossTurn) -> CrossTurn) = _state.update { s ->
            s.copy(turns = s.turns.map { if (it.id == turnId) f(it) else it })
        }

        // 1) ASR：自动检测语种（partial 实时上屏）
        val asr = engine.asrDetect(pcm) { partial ->
            if (partial.isNotBlank()) patch { it.copy(transcript = partial) }
        }
        if (asr.text.isBlank()) {
            _state.update { s -> s.copy(turns = s.turns.filterNot { it.id == turnId }) }
            _state.update { it.copy(stage = TalkStage.IDLE, statusText = "", error = "未识别到语音内容") }
            return@withContext
        }
        patch { it.copy(transcript = asr.text) }

        // 2) 语种判定归属：识别语种对上谁的配置语种，就说的是谁的话 → 译成对方语种
        val parties = _state.value.parties
        val speaker = detectSpeaker(asr.text, asr.language, parties)
        val other = parties.firstOrNull { it.id != speaker.id } ?: speaker
        val srcLang = speaker.langCode
        val tgtLang = other.langCode
        patch { it.copy(speakerId = speaker.id, speakerName = speaker.name, srcLangCode = srcLang, tgtLangCode = tgtLang) }

        if (srcLang == tgtLang) {
            // 双方语种相同：无翻译，转写即内容
            patch { it.copy(translation = asr.text, phase = TurnPhase.SYNTHESIZING) }
        } else {
            // 3) 翻译成对方语种（MT 自检源语种）
            _state.update { it.copy(statusText = "翻译成${langName(tgtLang)}…") }
            patch { it.copy(phase = TurnPhase.TRANSLATING) }
            val mt = engine.translate(asr.text, srcLang, tgtLang)
            if (mt.isBlank()) {
                patch { it.copy(phase = TurnPhase.FAILED) }
                _state.update { it.copy(stage = TalkStage.IDLE, statusText = "", error = "翻译失败") }
                return@withContext
            }
            patch { it.copy(translation = mt) }
        }

        // 4) TTS：音色克隆自说话人本段语音，合成对方语种的播报
        val ttsText = _state.value.turns.firstOrNull { it.id == turnId }?.translation ?: asr.text
        if (!tts.isLoaded) {
            _state.update { it.copy(loadingTts = true, statusText = "语音合成模型加载中…") }
            tts.warmUp()
            _state.update { it.copy(loadingTts = false) }
            if (!tts.isLoaded) {
                patch { it.copy(phase = TurnPhase.FAILED) }
                persistTurn(turnId)
                _state.update {
                    it.copy(stage = TalkStage.IDLE, statusText = "",
                        error = "语音合成模型未就绪（详见 logcat Tag=TtsEngine）")
                }
                return@withContext
            }
        }
        _state.update { it.copy(statusText = "语音合成中…", ttsFrames = 0) }
        patch { it.copy(phase = TurnPhase.SYNTHESIZING) }
        val audio = try {
            tts.synthesizeCloned(ttsText, pcm) { frames ->
                _state.update { it.copy(ttsFrames = frames) }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "tts failed", e); null
        }
        if (audio == null) {
            patch { it.copy(phase = TurnPhase.FAILED) }
            persistTurn(turnId)
            _state.update { it.copy(stage = TalkStage.IDLE, statusText = "", error = "语音合成失败") }
            return@withContext
        }
        val broadcastFile = File(store.dir, "tts_$turnId.wav")
        TtsEngine.writeWavStereo(broadcastFile.absolutePath, audio.pcmInterleaved)
        patch { it.copy(broadcastPath = broadcastFile.absolutePath) }
        persistTurn(turnId)

        // 5) 自动播报
        _state.update { it.copy(stage = TalkStage.PLAYING, statusText = "播报中…") }
        patch { it.copy(phase = TurnPhase.PLAYING) }
        playBroadcast(broadcastFile.absolutePath, turnId)
    }

    /** 自动播报（VM 持有播放器；完成回调置回 DONE/IDLE）。 */
    private fun playBroadcast(path: String, turnId: String) {
        stopPlayback()
        runCatching {
            val mp = MediaPlayer()
            mp.setDataSource(path)
            mp.setOnPreparedListener { it.start() }
            mp.setOnCompletionListener {
                releasePlayer()
                _state.update { s ->
                    s.copy(
                        stage = TalkStage.IDLE, statusText = "",
                        turns = s.turns.map { if (it.id == turnId) it.copy(phase = TurnPhase.DONE) else it }
                    )
                }
            }
            mp.setOnErrorListener { _, _, _ ->
                releasePlayer()
                _state.update { s ->
                    s.copy(stage = TalkStage.IDLE,
                        turns = s.turns.map { if (it.id == turnId) it.copy(phase = TurnPhase.DONE) else it })
                }
                true
            }
            mp.prepareAsync()
            player = mp
        }.onFailure {
            Log.e(TAG, "broadcast play failed", it)
            releasePlayer()
            _state.update { s ->
                s.copy(stage = TalkStage.IDLE,
                    turns = s.turns.map { t -> if (t.id == turnId) t.copy(phase = TurnPhase.DONE) else t })
            }
        }
    }

    /** 停止播报（开始下一段录音 / 用户手动回放前调用）；播放中的发言置回 DONE。 */
    fun stopPlayback() {
        player?.let { mp ->
            runCatching { mp.stop() }
            releasePlayer()
        }
        _state.update { s ->
            val base = if (s.stage == TalkStage.PLAYING) s.copy(stage = TalkStage.IDLE, statusText = "") else s
            base.copy(turns = base.turns.map {
                if (it.phase == TurnPhase.PLAYING) it.copy(phase = TurnPhase.DONE) else it
            })
        }
    }

    private fun releasePlayer() {
        runCatching { player?.release() }
        player = null
    }

    private fun persistTurn(turnId: String) {
        val turn = _state.value.turns.firstOrNull { it.id == turnId } ?: return
        val done = turn.copy(phase = TurnPhase.DONE)
        val idx = history.indexOfFirst { it.id == turnId }
        if (idx >= 0) history[idx] = done else history.add(done)
        store.saveHistory(history)
    }

    fun clearError() = _state.update { it.copy(error = null) }

    override fun onCleared() {
        recorder.stop()
        recJob?.cancel()
        timerJob?.cancel()
        summaryJob?.cancel()
        releasePlayer()
    }

    companion object {
        private const val TAG = "CrossLangVM"
        private fun langName(code: String): String = langOf(code)?.name ?: code
    }

    /**
     * 说话人归属判定：说的是谁的配置语种。当前 ASR 模型（<asr_text> 纯转写模式）不输出
     * 语言标签（AsrResult.language 恒 null），故按可靠性依次：
     *  1) ASR 语言标签（未来模型若输出则直接用）；
     *  2) 字符集判定——韩文/假名/汉字/西里尔/希腊/阿拉伯字母可无歧义定位；双方分属不同
     *     文字类别时（如 中文↔拉丁语种）瞬间判定、零开销；
     *  3) MT 判别——双方同字符集（如 英↔德）时分别译到双方语种：把文本译回「源语种本身」
     *     输出≈原文，相似度最高者即源语种。
     *  仍无法判定（如双方同语种）时默认归 A。
     */
    private suspend fun detectSpeaker(text: String, asrLang: String?, parties: List<Party>): Party {
        asrLang?.let { l -> parties.firstOrNull { it.langCode == l }?.let { return it } }
        textScript(text)?.let { script ->
            val candidates = parties.filter { scriptMatches(it.langCode, script) }
            if (candidates.size == 1) return candidates[0]
        }
        if (parties.size == 2 && parties[0].langCode != parties[1].langCode) {
            val a = parties[0]
            val b = parties[1]
            _state.update { it.copy(statusText = "识别语种中…") }
            val toA = engine.translate(text, asrLang ?: "zh", a.langCode)
            val toB = engine.translate(text, asrLang ?: "zh", b.langCode)
            val simA = textSimilarity(text, toA)
            val simB = textSimilarity(text, toB)
            Log.i(TAG, "lang detect: to ${a.langCode}=$toA (sim=$simA), to ${b.langCode}=$toB (sim=$simB)")
            if (simA > simB + 0.05f) return a
            if (simB > simA + 0.05f) return b
        }
        return parties.first()
    }

    /** 文本字符集 → 文字类别；无有效字符返回 null。 */
    private fun textScript(text: String): String? {
        var han = false; var kana = false; var hangul = false; var cyrillic = false
        var greek = false; var arabic = false; var latin = false
        for (c in text) when {
            c in '가'..'힣' -> hangul = true   // 韩文
            c in '぀'..'ヿ' -> kana = true      // 平假名/片假名
            c in '一'..'鿿' -> han = true       // CJK 汉字
            c in 'Ѐ'..'ӿ' -> cyrillic = true  // 西里尔（俄语）
            c in 'Ͱ'..'Ͽ' -> greek = true     // 希腊
            c in '؀'..'ۿ' || c in 'ﭐ'..'ﻼ' -> arabic = true  // 阿拉伯字母（阿/波）
            c.isLetter() && c.code < 0x0250 -> latin = true  // 拉丁字母
        }
        return when {
            hangul -> "ko"
            kana -> "ja"          // 有假名必是日语
            han -> "han"          // 纯汉字：中文或日语汉字词
            cyrillic -> "cyrillic"
            greek -> "greek"
            arabic -> "arabic"
            latin -> "latin"
            else -> null
        }
    }

    /** 语种码可书写的文字类别（zh→han、ja→ja：日语以假名为主要特征）。 */
    private fun langScript(code: String): String = when (code) {
        "zh" -> "han"; "ja" -> "ja"; "ko" -> "ko"
        "ru" -> "cyrillic"; "el" -> "greek"
        "ar", "fa" -> "arabic"
        else -> "latin"
    }

    /** 文字类别与语种码是否兼容（纯汉字可属中文或日语；阿拉伯字母可属阿/波斯）。 */
    private fun scriptMatches(langCode: String, script: String): Boolean = when (script) {
        "ja" -> langCode == "ja"                          // 有假名只可能是日语
        "han" -> langScript(langCode) == "han" || langCode == "ja"
        "arabic" -> langScript(langCode) == "arabic"
        else -> langScript(langCode) == script
    }

    /** 文本相似度（0..1）：小写化、去空白标点后的编辑距离比。 */
    private fun textSimilarity(a: String, b: String): Float {
        val x = a.lowercase().filter { it.isLetterOrDigit() }
        val y = b.lowercase().filter { it.isLetterOrDigit() }
        if (x.isEmpty() && y.isEmpty()) return 1f
        if (x.isEmpty() || y.isEmpty()) return 0f
        if (x == y) return 1f
        val dp = IntArray(y.length + 1) { it }
        for (i in 1..x.length) {
            var prev = dp[0]
            dp[0] = i
            for (j in 1..y.length) {
                val tmp = dp[j]
                dp[j] = minOf(dp[j] + 1, dp[j - 1] + 1, prev + if (x[i - 1] == y[j - 1]) 0 else 1)
                prev = tmp
            }
        }
        return 1f - dp[y.length].toFloat() / maxOf(x.length, y.length)
    }
}

// ──────────────────────────────────────────────────────────────────
// 页面
// ──────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CrossLangScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val vm: CrossLangViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance((context.applicationContext as Application))
    )
    val state by vm.state.collectAsState()
    val player = rememberTurnPlayer(vm::stopPlayback)

    val micPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) vm.start() }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            CrossHeader(
                state = state,
                onBack = onBack,
                onToggleConfig = vm::toggleConfig,
                onToggleHistory = vm::toggleHistory,
                onStartMic = {
                    if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                        == PackageManager.PERMISSION_GRANTED) vm.start()
                    else micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                },
                onStopMic = vm::stop
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // ── 配置抽屉 ──
            AnimatedVisibility(
                visible = state.configOpen,
                enter = expandVertically(
                    expandFrom = Alignment.Top, animationSpec = tween(220)
                ) + fadeIn(tween(220)),
                exit = shrinkVertically(
                    shrinkTowards = Alignment.Top, animationSpec = tween(180)
                ) + fadeOut(tween(180))
            ) {
                ConfigDrawer(
                    parties = state.parties,
                    enabled = state.stage == TalkStage.IDLE,
                    onUpdate = vm::updateParty
                )
            }

            // ── 历史记录抽屉 ──
            AnimatedVisibility(
                visible = state.historyOpen,
                enter = expandVertically(
                    expandFrom = Alignment.Top, animationSpec = tween(220)
                ) + fadeIn(tween(220)),
                exit = shrinkVertically(
                    shrinkTowards = Alignment.Top, animationSpec = tween(180)
                ) + fadeOut(tween(180))
            ) {
                HistoryPanel(
                    vm = vm,
                    state = state,
                    player = player,
                    modifier = Modifier.fillMaxWidth().weight(1f)
                )
            }

            // ── 主区：发言流 ──
            if (!state.configOpen && !state.historyOpen) {
                TalkMainArea(state, vm, player, Modifier.fillMaxWidth().weight(1f))
            }
        }
    }
}

// ── 自定义头部：返回 + 仅标题（可换行）+ 右侧三操作块（配置/历史/开始-停止）──
@Composable
private fun CrossHeader(
    state: CrossUiState,
    onBack: () -> Unit,
    onToggleConfig: () -> Unit,
    onToggleHistory: () -> Unit,
    onStartMic: () -> Unit,
    onStopMic: () -> Unit
) {
    val gradient = Brush.linearGradient(listOf(CrossTeal, lerp(CrossTeal, Color.White, 0.22f)))
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(gradient)
            .windowInsetsPadding(WindowInsets.statusBars)
            .padding(start = 12.dp, end = 12.dp, top = 10.dp, bottom = 14.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(38.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color.White.copy(alpha = 0.2f))
                    .clickable(onClick = onBack),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "返回",
                    tint = Color.White,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "跨语沟通智能体",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                modifier = Modifier.weight(1f)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                HeaderAction(icon = Icons.Default.Tune, label = "配置",
                    highlighted = state.configOpen, onClick = onToggleConfig)
                HeaderAction(icon = Icons.Default.History, label = "历史",
                    highlighted = state.historyOpen, onClick = onToggleHistory)
                when (state.stage) {
                    TalkStage.RECORDING -> HeaderAction(
                        icon = Icons.Default.Stop, label = "停止", highlighted = true, onClick = onStopMic)
                    TalkStage.PROCESSING -> HeaderAction(
                        icon = null, label = "处理中", enabled = false, onClick = {})
                    TalkStage.PLAYING -> HeaderAction(
                        icon = Icons.Default.VolumeUp, label = "播报中", enabled = false, onClick = {})
                    TalkStage.IDLE -> HeaderAction(
                        icon = Icons.Default.Mic, label = "开始", onClick = onStartMic)
                }
            }
        }
    }
}

/** 头部单个操作块：上图标（半透明白底圆角块）下名称。 */
@Composable
private fun HeaderAction(
    icon: androidx.compose.ui.graphics.vector.ImageVector?,
    label: String,
    highlighted: Boolean = false,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 4.dp, vertical = 2.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Box(
            modifier = Modifier
                .size(38.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(
                    when {
                        !enabled -> Color.White.copy(alpha = 0.12f)
                        highlighted -> Color.White.copy(alpha = 0.45f)
                        else -> Color.White.copy(alpha = 0.22f)
                    }
                ),
            contentAlignment = Alignment.Center
        ) {
            if (icon != null) {
                Icon(imageVector = icon, contentDescription = label,
                    tint = Color.White, modifier = Modifier.size(20.dp))
            } else {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp), strokeWidth = 2.dp, color = Color.White)
            }
        }
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = Color.White.copy(alpha = if (enabled) 0.95f else 0.5f)
        )
    }
}

// ── 主区：录音状态条 + 发言卡片流 ──
@Composable
private fun TalkMainArea(
    state: CrossUiState,
    vm: CrossLangViewModel,
    player: TurnPlayer,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // 录音/状态指示条
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (state.stage == TalkStage.RECORDING) {
                val transition = rememberInfiniteTransition(label = "rec")
                val pulse by transition.animateFloat(
                    initialValue = 0.4f, targetValue = 1f,
                    animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse), label = "pulse")
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .alpha(pulse)
                        .clip(CircleShape)
                        .background(RecRed)
                )
                Text(
                    text = "录音中 ${state.recMs / 1000.0}s · 说完点「停止」",
                    style = MaterialTheme.typography.bodySmall,
                    color = RecRed
                )
            } else {
                Icon(
                    imageVector = Icons.Default.RecordVoiceOver,
                    contentDescription = null,
                    tint = CrossTeal, modifier = Modifier.size(18.dp))
                Text(
                    text = when {
                        state.stage == TalkStage.PLAYING -> state.statusText.ifBlank { "播报中…" }
                        state.stage == TalkStage.PROCESSING -> state.statusText.ifBlank { "处理中…" }
                        state.loadingTts -> "语音合成模型加载中…"
                        state.statusText.isNotEmpty() -> state.statusText
                        else -> "点「开始」说话，自动识别语种并译给对方"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1, overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                // 双方语种速览
                state.parties.forEach { p ->
                    Text(
                        text = "${p.name.take(3)} ${langOf(p.langCode)?.flag ?: ""}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }
        }

        // 错误提示（点 ✕ 清除）
        if (state.error != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.4f))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(Icons.Default.Info, contentDescription = null,
                    tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(16.dp))
                Text(
                    text = state.error!!,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.weight(1f)
                )
                Icon(Icons.Default.Close, contentDescription = "清除",
                    tint = MaterialTheme.colorScheme.error,
                    modifier = Modifier.size(16.dp).clickable { vm.clearError() })
            }
        }

        // 发言流
        ElevatedCard(
            modifier = Modifier.fillMaxWidth().weight(1f),
            shape = RoundedCornerShape(18.dp),
            elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
        ) {
            if (state.turns.isEmpty()) {
                Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
                    Text(
                        text = "配置好双方语种后点「开始」：\n谁在说话由语音自动判定，\n译文按对方语种合成播报",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.outline,
                        lineHeight = 22.sp
                    )
                }
            } else {
                val listState = rememberLazyListState()
                val last = state.turns.lastOrNull()
                LaunchedEffect(state.turns.size, last?.transcript, last?.translation, last?.phase) {
                    if (state.turns.isNotEmpty()) listState.animateScrollToItem(state.turns.lastIndex)
                }
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize().padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(state.turns, key = { it.id }) { turn ->
                        TurnCard(turn, state, vm, player)
                    }
                }
            }
        }
    }
}

// ── 单段发言卡片：说话人 + 原音 + 转写 + 译文 + 播报 ──
@Composable
private fun TurnCard(
    turn: CrossTurn,
    state: CrossUiState,
    vm: CrossLangViewModel,
    player: TurnPlayer
) {
    val speaker = state.parties.firstOrNull { it.id == turn.speakerId }
    val color = when {
        speaker != null -> if (speaker.id == "a") SPK_A else SPK_B
        turn.speakerId.isNotEmpty() -> SPK_A
        else -> MaterialTheme.colorScheme.outline   // 归属判定中
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .border(1.dp, color.copy(alpha = 0.45f), RoundedCornerShape(12.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        // 头行：说话人 + 语种方向 + 时间 + 阶段
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(
                modifier = Modifier.size(22.dp).clip(CircleShape).background(color),
                contentAlignment = Alignment.Center
            ) {
                Text((if (turn.speakerId.isEmpty()) "?" else speaker?.name?.take(1) ?: "?"),
                    color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            }
            Text(
                text = when {
                    turn.speakerId.isEmpty() -> "识别语种中…"
                    turn.srcLangCode.isNotEmpty() && turn.tgtLangCode.isNotEmpty() && turn.srcLangCode != turn.tgtLangCode ->
                        "${speaker?.name ?: ""} · ${langFlag(turn.srcLangCode)} → ${langFlag(turn.tgtLangCode)}"
                    else -> speaker?.name ?: ""
                },
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
                color = color
            )
            Spacer(modifier = Modifier.weight(1f))
            Text(turn.timeLabel, style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline)
        }
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

        // 原音播放
        turn.speechPath?.let { path ->
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                PlayButton(player, "${turn.id}-src", path, color)
                TagBadge("原音", color)
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
        }
        // 转写
        Row(verticalAlignment = Alignment.CenterVertically) {
            TagBadge("转写", MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = turn.transcript.ifBlank { "…" },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
        // 译文
        if (turn.translation != null) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                TagBadge("译文", Color(0xFFF97316))
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = turn.translation,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
        // 播报
        turn.broadcastPath?.let { path ->
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                PlayButton(player, "${turn.id}-tts", path, CrossTeal)
                TagBadge("播报", CrossTeal)
                Text(
                    text = langOf(turn.tgtLangCode)?.let { "${it.flag} ${it.name}" } ?: turn.tgtLangCode,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        // 阶段徽标
        when (turn.phase) {
            TurnPhase.TRANSCRIBING -> PhaseText("转写中…", color)
            TurnPhase.TRANSLATING -> PhaseText("翻译中…", color)
            TurnPhase.SYNTHESIZING -> PhaseText(
                if (state.ttsFrames > 0) "语音合成中…${state.ttsFrames} 帧" else "语音合成中…", color)
            TurnPhase.PLAYING -> PhaseText("播报中", CrossTeal)
            TurnPhase.FAILED -> PhaseText("处理失败", MaterialTheme.colorScheme.error)
            TurnPhase.DONE -> {}
        }
    }
}

@Composable
private fun PhaseText(text: String, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        CircularProgressIndicator(modifier = Modifier.size(12.dp), strokeWidth = 1.5.dp, color = color)
        Text(text, style = MaterialTheme.typography.labelSmall, color = color)
    }
}

/** 圆形播放按钮（原音/播报共用，同一时刻只播一个）。 */
@Composable
private fun PlayButton(player: TurnPlayer, key: String, path: String, color: Color) {
    val active = player.playingKey == key
    Box(
        modifier = Modifier
            .size(30.dp)
            .clip(CircleShape)
            .background(if (active) color else MaterialTheme.colorScheme.surface)
            .clickable { player.toggle(key, path) },
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = if (active && player.isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
            contentDescription = "播放",
            tint = if (active) Color.White else color,
            modifier = Modifier.size(16.dp)
        )
    }
}

/** 原音/播报回放播放器（与 VM 自动播报互斥：点回放时停掉 VM 播放）。 */
private class TurnPlayer(private val onStopVmPlayback: () -> Unit) {
    private var mp: MediaPlayer? = null
    var playingKey by mutableStateOf<String?>(null)
        private set
    var isPlaying by mutableStateOf(false)
        private set

    fun toggle(key: String, path: String) {
        onStopVmPlayback()
        if (playingKey == key) {
            val m = mp
            if (m != null && m.isPlaying) { m.pause(); isPlaying = false }
            else { m?.start(); isPlaying = true }
        } else {
            release()
            runCatching {
                val m = MediaPlayer()
                m.setDataSource(path)
                m.setOnPreparedListener { it.start(); isPlaying = true }
                m.setOnCompletionListener { isPlaying = false; playingKey = null }
                m.setOnErrorListener { _, _, _ -> isPlaying = false; playingKey = null; true }
                m.prepareAsync()
                mp = m
                playingKey = key
                isPlaying = false
            }.onFailure { release() }
        }
    }

    fun release() {
        runCatching { mp?.release() }
        mp = null; playingKey = null; isPlaying = false
    }
}

@Composable
private fun rememberTurnPlayer(onStopVmPlayback: () -> Unit = {}): TurnPlayer {
    val p = remember { TurnPlayer(onStopVmPlayback) }
    DisposableEffect(Unit) { onDispose { p.release() } }
    return p
}

// ── 配置抽屉：双方名字 / 语种 / 声音性别 ──
@Composable
private fun ConfigDrawer(
    parties: List<Party>,
    enabled: Boolean,
    onUpdate: (id: String, name: String?, langCode: String?) -> Unit
) {
    var langPickerFor by remember { mutableStateOf<Party?>(null) }

    ElevatedCard(
        modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle(icon = Icons.Default.Tune, text = "沟通双方配置", accent = CrossTeal)
            parties.forEach { party ->
                PartyConfigCard(party, enabled, onUpdate, onPickLang = { langPickerFor = party })
            }
            Text(
                text = "谁在说话由语音自动判定，译文自动译成对方语种；播报用说话人自己的音色合成" +
                    "（克隆自本段语音），19 种语种均支持。",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline,
                lineHeight = 18.sp
            )
        }
    }

    langPickerFor?.let { party ->
        LanguagePickerDialog(
            title = "${party.name} 的语种",
            selected = party.langCode,
            onPick = { onUpdate(party.id, null, it); langPickerFor = null },
            onDismiss = { langPickerFor = null }
        )
    }
}

@Composable
private fun PartyConfigCard(
    party: Party,
    enabled: Boolean,
    onUpdate: (id: String, name: String?, langCode: String?) -> Unit,
    onPickLang: () -> Unit
) {
    val color = if (party.id == "a") SPK_A else SPK_B
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(color.copy(alpha = 0.08f))
            .border(1.dp, color.copy(alpha = 0.4f), RoundedCornerShape(14.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(
                modifier = Modifier.size(26.dp).clip(CircleShape).background(color),
                contentAlignment = Alignment.Center
            ) {
                Text(party.name.take(1), color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            }
            OutlinedTextField(
                value = party.name,
                onValueChange = { if (enabled) onUpdate(party.id, it.take(8), null) },
                enabled = enabled, singleLine = true,
                textStyle = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f).heightIn(min = 52.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    unfocusedBorderColor = Color.Transparent,
                    focusedBorderColor = color.copy(alpha = 0.5f)
                )
            )
        }
        // 语种选择
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(MaterialTheme.colorScheme.surface)
                .clickable(enabled = enabled, onClick = onPickLang)
                .padding(horizontal = 10.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(langOf(party.langCode)?.flag ?: "", fontSize = 16.sp)
            Text(
                text = langOf(party.langCode)?.name ?: party.langCode,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.weight(1f)
            )
            Icon(Icons.Default.ExpandMore, contentDescription = null,
                tint = MaterialTheme.colorScheme.outline, modifier = Modifier.size(18.dp))
        }
    }
}

@Composable
private fun LanguagePickerDialog(
    title: String,
    selected: String,
    onPick: (String) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                supportedLangs.forEach { lang ->
                    val sel = lang.code == selected
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onPick(lang.code) }
                            .padding(horizontal = 8.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(lang.flag, fontSize = 18.sp)
                        Spacer(modifier = Modifier.width(10.dp))
                        Text(
                            text = lang.name,
                            modifier = Modifier.weight(1f),
                            fontWeight = if (sel) FontWeight.SemiBold else FontWeight.Normal
                        )
                        if (sel) Icon(Icons.Default.Check, contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary)
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } }
    )
}

// ── 双语纪要抽屉：A/B 双方语种各一份纪要 ──
/**
 * 双语纪要：**内嵌在历史记录里某条沟通之下**，与「会话详情」并列为两个标签页之一。
 * 生成只针对这条历史会话记录下来的发言，跟当前正在录的那场无关。
 *
 * 展示沿用总结方案 C 的三段结构（大纲 → 要点 → 待办），生成中三股内容各自流式增长，
 * 关键句摘录在生成时是即时预览、完成后变成可折叠附件。
 *
 * 操作区是**固定两行、按钮数量与文案都不随状态变化**：按钮增删/改文案会把整块挤变形
 * （生成中多一个"停止"、文案"双语纪要"变"生成中…"都会重排），
 * 生成期的反馈统一交给上方那条固定高度的进度行。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SummaryPanel(
    vm: CrossLangViewModel,
    state: CrossUiState,
    modifier: Modifier = Modifier
) {
    val cur = if (state.summaryTab == 0) state.summaryA else state.summaryB
    val scroll = rememberScrollState()
    LaunchedEffect(cur.outline.length, cur.points.length, cur.todos.length, cur.excerpt.size) {
        scroll.animateScrollTo(scroll.maxValue)
    }
    val busy = state.summarizing || state.summaryLoadingLlm

    Column(modifier = modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        // 语种 Tab：A 方语种 / B 方语种（各自一份独立纪要）
        TabRow(
            selectedTabIndex = state.summaryTab,
            containerColor = Color.Transparent,
            contentColor = CrossTeal
        ) {
            listOf(state.summaryA, state.summaryB).forEachIndexed { i, st ->
                Tab(
                    selected = state.summaryTab == i,
                    onClick = { vm.selectSummaryTab(i) },
                    text = {
                        Text(
                            text = if (st.langName.isNotBlank()) "${langFlag(st.langCode)} ${st.langName}纪要" else "语种 ${i + 1}",
                            style = MaterialTheme.typography.labelLarge,
                            maxLines = 1, overflow = TextOverflow.Ellipsis
                        )
                    }
                )
            }
        }

        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LangSummaryBody(st = cur, summarizing = state.summarizing, scroll = scroll)
        }

        // 进度行：固定高度占位——生成状态只改这一行，不牵动下方按钮
        Row(
            modifier = Modifier.fillMaxWidth().height(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            if (busy) {
                CircularProgressIndicator(
                    modifier = Modifier.size(14.dp), strokeWidth = 2.dp, color = CrossTeal
                )
                Text(
                    text = state.summaryStatus.ifBlank { "纪要生成中…" },
                    style = MaterialTheme.typography.labelSmall,
                    color = CrossTeal, maxLines = 1, overflow = TextOverflow.Ellipsis
                )
            }
        }

        // ── 操作区：两行固定结构，按钮数量与文案都不随生成状态变化 ──
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = { vm.summarizeSession(SummaryTarget.BOTH) },
                enabled = !busy,
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(4.dp))
                Text("双语纪要", maxLines = 1)
            }
            // 常驻（非生成期置灰）：按钮一多一少会把整行挤变形
            OutlinedButton(onClick = vm::stopSummary, enabled = busy) {
                Text(
                    "停止",
                    color = if (busy) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.outline,
                    maxLines = 1
                )
            }
        }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SummaryLangButton(state.summaryA, busy, Modifier.weight(1f)) {
                vm.summarizeSession(SummaryTarget.A)
            }
            SummaryLangButton(state.summaryB, busy, Modifier.weight(1f)) {
                vm.summarizeSession(SummaryTarget.B)
            }
        }
    }
}

/** 单语纪要按钮：flag + 语种名，宽度由 weight 平分（不随文案长短变）。 */
@Composable
private fun SummaryLangButton(
    st: LangSummaryState,
    busy: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    OutlinedButton(onClick = onClick, enabled = !busy, modifier = modifier) {
        Text(
            text = if (st.langName.isNotBlank()) "${langFlag(st.langCode)} ${st.langName}" else "单语",
            maxLines = 1, overflow = TextOverflow.Ellipsis
        )
    }
}

/** 单语纪要正文：生成中显示流式三段，否则显示存档（含可折叠关键句摘录）。 */
@Composable
private fun LangSummaryBody(st: LangSummaryState, summarizing: Boolean, scroll: ScrollState) {
    if (summarizing && st.streaming) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scroll)
                .padding(vertical = 4.dp)
        ) {
            if (st.outline.isNotBlank()) {
                Text(
                    text = "话题：" + st.outline,
                    style = MaterialTheme.typography.bodySmall,
                    color = CrossTeal
                )
                Spacer(modifier = Modifier.height(8.dp))
            }
            if (st.points.isNotBlank()) MarkdownText(markdown = st.points)
            if (st.todos.isNotBlank()) {
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = "【待办】",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = CrossTeal
                )
                MarkdownText(markdown = st.todos)
            }
            if (st.excerpt.isNotEmpty()) ExcerptCard(st.excerpt)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = "▍", style = MaterialTheme.typography.bodyMedium, color = CrossTeal)
        }
        return
    }
    if (st.saved.isNullOrBlank()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Icon(
                    Icons.Default.Description, contentDescription = null,
                    tint = MaterialTheme.colorScheme.outline, modifier = Modifier.size(32.dp)
                )
                Text(
                    text = "还没有${st.langName.ifBlank { "该语种" }}纪要，点下方按钮生成",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
        return
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scroll)
            .padding(vertical = 4.dp)
    ) {
        MarkdownText(markdown = st.saved)
        if (st.savedExcerpt.isNotEmpty()) ExcerptCard(st.savedExcerpt)
    }
}

// ── 历史记录抽屉：会话列表 → 会话详情（原音/转写/译文/播报）──
@Composable
@OptIn(ExperimentalMaterial3Api::class)
private fun HistoryPanel(
    vm: CrossLangViewModel,
    state: CrossUiState,
    player: TurnPlayer,
    modifier: Modifier = Modifier
) {
    var pendingDelete by remember { mutableStateOf<CrossSession?>(null) }
    // 不 remember：每次重组重取，删除会话后（state 变化触发重组）即为最新
    val sessions = vm.historySessions()
    val detail = state.historyDetailId?.let { id -> sessions.firstOrNull { it.id == id } }

    ElevatedCard(
        modifier = modifier.padding(top = 12.dp),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (detail != null) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.KeyboardArrowLeft,
                        contentDescription = null, tint = CrossTeal,
                        modifier = Modifier.size(22.dp).clickable { vm.closeSession() })
                }
                SectionTitle(
                    icon = Icons.Default.History,
                    text = if (detail != null) "沟通记录" else "历史记录",
                    accent = CrossTeal,
                    modifier = Modifier.weight(1f)
                )
                if (detail != null) {
                    Text(
                        text = "${detail.turns.size} 段发言",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }

            // 展开某条沟通记录后：会话详情 / 双语纪要 两个标签页
            if (detail != null) {
                TabRow(
                    selectedTabIndex = state.historyTab,
                    containerColor = Color.Transparent,
                    contentColor = CrossTeal
                ) {
                    Tab(
                        selected = state.historyTab == 0,
                        onClick = { vm.selectHistoryTab(0) },
                        text = { Text("会话详情", style = MaterialTheme.typography.labelLarge) }
                    )
                    Tab(
                        selected = state.historyTab == 1,
                        onClick = { vm.selectHistoryTab(1) },
                        text = { Text("双语纪要", style = MaterialTheme.typography.labelLarge) }
                    )
                }
            }
            HorizontalDivider()

            if (detail != null) {
                // 纪要只针对这条历史记录里的沟通内容（数据来自该会话落盘的发言）
                if (state.historyTab == 1) {
                    SummaryPanel(vm = vm, state = state, modifier = Modifier.fillMaxWidth().weight(1f))
                } else {
                    LazyColumn(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        items(detail.turns, key = { it.id }) { turn ->
                            // 历史记录无 live 状态，借用当前 parties 着色
                            TurnCard(turn, state, vm, player)
                        }
                    }
                }
            } else {
                if (sessions.isEmpty()) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text("暂无历史沟通记录", style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.outline)
                    }
                } else {
                    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        items(sessions, key = { it.id }) { s ->
                            SessionRow(
                                session = s,
                                summaryCount = listOfNotNull(
                                    state.summaries[s.id]?.textA, state.summaries[s.id]?.textB).size,
                                onClick = { vm.openSession(s.id) },
                                // 纪要从列表直达：展开该会话并落在「双语纪要」页
                                onSummary = { vm.openSession(s.id, 1) },
                                onDelete = { pendingDelete = s }
                            )
                        }
                    }
                }
            }
        }
    }

    pendingDelete?.let { s ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("删除会话") },
            text = { Text("删除 ${s.turns.size} 段沟通记录及其语音文件，无法恢复。") },
            confirmButton = {
                TextButton(onClick = {
                    vm.deleteSession(s.id)
                    pendingDelete = null
                    if (state.historyDetailId == s.id) vm.closeSession()
                }) { Text("删除", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { pendingDelete = null }) { Text("取消") } }
        )
    }
}

@Composable
private fun SessionRow(
    session: CrossSession,
    summaryCount: Int = 0,
    onClick: () -> Unit,
    onSummary: () -> Unit,
    onDelete: () -> Unit
) {
    val first = session.turns.firstOrNull()
    val time = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(session.startedAt))
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Icon(Icons.Default.Forum, contentDescription = null, tint = CrossTeal, modifier = Modifier.size(20.dp))
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                text = first?.transcript?.take(24)?.ifBlank { "（语音）" } ?: "（语音）",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1, overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "$time · ${session.turns.size} 段" +
                    if (summaryCount > 0) " · 纪要 $summaryCount/2" else "",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )
        }
        IconButton(onClick = onSummary, modifier = Modifier.size(28.dp)) {
            Icon(Icons.Default.Description, contentDescription = "双语纪要",
                tint = if (summaryCount > 0) CrossTeal else MaterialTheme.colorScheme.outline,
                modifier = Modifier.size(16.dp))
        }
        IconButton(onClick = onDelete, modifier = Modifier.size(28.dp)) {
            Icon(Icons.Default.Delete, contentDescription = "删除",
                tint = MaterialTheme.colorScheme.outline, modifier = Modifier.size(16.dp))
        }
    }
}

private fun langFlag(code: String): String = langOf(code)?.flag ?: code

@Composable
private fun TagBadge(text: String, color: Color) {
    Box(
        Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(color.copy(alpha = 0.15f))
            .padding(horizontal = 6.dp, vertical = 2.dp)
    ) {
        Text(text, style = MaterialTheme.typography.labelSmall, color = color,
            fontWeight = FontWeight.SemiBold)
    }
}
