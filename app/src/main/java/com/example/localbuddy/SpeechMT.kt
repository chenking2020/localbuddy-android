package com.example.localbuddy

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.TextSnippet
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.style.TextOverflow
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.sqrt
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

// ──────────────────────────────────────────────────────────────────
// 数据模型
// ──────────────────────────────────────────────────────────────────

/** 一段转写 + 可选翻译。 */
data class Segment(
    val id: String = UUID.randomUUID().toString(),
    val transcript: String,
    val translation: String? = null,
    val audioPath: String? = null,   // 该段独立 WAV（录音模式按段落盘；上传模式=整条）
    val audioSamples: Int = 0        // 该段 PCM 样本数（算时长用，16k -> samples/16000 秒）
)

/** 上传模式顶侧展示的整条音频概览：先让用户看到并试听原文件，再看下面的分段结果。 */
data class UploadFilePreview(
    val fileName: String,
    val uriString: String,   // 原始 content Uri：MediaPlayer 可直接播整条，不必等解码完成
    val durationMs: Long = 0L,
    val sizeBytes: Long = 0L
) {
    /** 时长文案；拿不到时长（0）时显示 --:--。 */
    val durationLabel: String
        get() = if (durationMs <= 0) "--:--" else {
            val total = durationMs / 1000
            "%d:%02d".format(total / 60, total % 60)
        }

    val sizeLabel: String
        get() = if (sizeBytes > 0) "%.1f MB".format(sizeBytes / 1024f / 1024f) else ""
}

/** 历史记录名：有纪要时取其提炼出的会议主题，否则退回首段转写。 */
private val TOPIC_MARKDOWN_RE = Regex("^[#>*\\-·\\s]+")
private val TOPIC_PREFIX_RE = Regex("^(?:会议主题|主题|标题|纪要标题|摘要|Topic|Summary)\\s*[:：\\-—]?\\s*", RegexOption.IGNORE_CASE)
private val SENTENCE_SPLIT_RE = Regex("[。！？!?]")

/**
 * 从纪要正文提炼「会议主题」：取首个非空行 → 去掉 Markdown 标题/列表符号与「会议主题：」之类前缀
 * → 截到第一个句末标点 → 限长 30 字。系统提示要求 LLM 开头一两句话点明主题，故首句即主题。
 */
private fun topicFromSummary(summary: String): String {
    val line = summary.lineSequence()
        .map { it.trim() }
        .map { it.replace(TOPIC_MARKDOWN_RE, "") }
        .map { it.replace(TOPIC_PREFIX_RE, "") }
        .firstOrNull { it.isNotBlank() } ?: return ""
    val firstSentence = line.split(SENTENCE_SPLIT_RE).firstOrNull()?.trim().orEmpty()
    val text = firstSentence.ifBlank { line }
    return if (text.length > 30) text.take(30) + "…" else text
}

/** 一条历史记录：可能由多段构成。 */
data class SpeechMTRecord(
    val id: String = UUID.randomUUID().toString(),
    val createdAt: Long,
    val source: SourceMode,
    val targetLang: String?,         // null 表示当时未开启翻译
    val segments: List<Segment>,
    val audioPath: String? = null,   // 完整拼接 WAV（录音模式）/ 整条 WAV（上传模式）
    val summary: String? = null,     // LLM 整理的纪要（有译文按译文生成，否则按转写）
    val excerpt: String? = null,     // 纪要依据的关键句摘录（换行分隔，可折叠附件供溯源）
    val titleOverride: String? = null // 上传音频：直接用源文件名（用户认文件名不认首句转写）
) {
    /** 记录名：上传音频用源文件名；有纪要时用会议主题；否则退回首段转写前 24 字。 */
    val title: String
        get() {
            titleOverride?.takeIf { it.isNotBlank() }?.let { return it }
            val topic = summary?.let { topicFromSummary(it) }?.takeIf { it.isNotBlank() }
            if (topic != null) return topic
            return segments.firstOrNull()?.transcript?.take(24)?.ifBlank { "（无文本）" }
                ?: "（无文本）"
        }

    val timeLabel: String
        get() = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(createdAt))
}

enum class SourceMode(val label: String) { Mic("录音"), Upload("上传音频") }

/** 历史详情的两个标签页：查看历史（音频/文字/译文）/ 纪要总结。 */
enum class HistoryDetailTab(val label: String) { CONTENT("查看历史"), SUMMARY("纪要总结") }

data class SpeechMTUiState(
    val sourceMode: SourceMode = SourceMode.Mic,
    val translateOn: Boolean = false,
    val targetLang: Language = languages[0],   // 默认中文
    val srcLangCode: String = LANG_AUTO,       // 源语种：auto=自动识别，选定后强制该语种转写
    val isBusy: Boolean = false,                // 录音中 / 处理中 / 排空中
    val isCapturing: Boolean = false,           // 正在采集麦克风（区别于停止后排空识别）
    val isLoadingModel: Boolean = false,        // 模型加载中（点击开始后若模型未就绪，结果区显示加载态）
    val statusText: String = "",                // 顶部状态提示
    val currentSegments: List<Segment> = emptyList(),
    val uploadPreview: UploadFilePreview? = null, // 上传模式：整条音频概览（钉在实时结果最顶侧）
    val viewingRecordId: String? = null,        // 非空时左侧选中某条历史，右下展示历史片段
    val history: List<SpeechMTRecord> = emptyList(),
    // ── 纪要整理 ──
    val detailTab: HistoryDetailTab = HistoryDetailTab.CONTENT, // 历史详情当前标签页
    val summarizing: Boolean = false,           // 纪要生成中（流式）
    val summaryStream: String = "",             // 流式纪要要点（生成中，增量追加）
    val todoStream: String = "",                // 流式待办事项（生成中，与要点分开展示）
    val outlineStream: String = "",             // 流式大纲（可见的"思考过程"）
    val excerpt: List<String> = emptyList(),    // 抽取式关键句（即时预览 → 完成后折叠附件）
    val loadingLlm: Boolean = false             // LLM 模型加载中（首次整理纪要）
)

// ──────────────────────────────────────────────────────────────────
// ViewModel
// ──────────────────────────────────────────────────────────────────
class SpeechMTViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = SpeechMTEngine.get(app)
    // 分段以语言表达完整性为主（区别于实时听音的细粒度切分）：句尾停顿 1s 才判句尾、
    // 单段上限 10s（< ASR 12s 截断上限），尽量把一整句/一小段完整语义切进同一段，
    // 减少碎句带来的转写/翻译上下文缺失
    private val recorder = MicRecorder(pauseMs = 1300, maxSegMs = 10_000)
    private val store = HistoryStore(app)
    /** 录音段/完整音频 WAV 落盘目录（filesDir/audio，随记录持久，删记录时清理）。 */
    private val audioDir = File(app.filesDir, "audio").apply { mkdirs() }.absolutePath

    /** 录音采集 -> 推理之间的传递单元：一段 PCM + 已落盘的 WAV 路径。 */
    private data class SegmentChunk(val pcm: FloatArray, val audioPath: String)

    private val _state = MutableStateFlow(SpeechMTUiState())
    val state: StateFlow<SpeechMTUiState> = _state.asStateFlow()

    private var recordingJob: Job? = null
    private var inferenceJob: Job? = null
    private var processingJob: Job? = null
    private var summarizeJob: Job? = null

    init {
        viewModelScope.launch { engine.warmUp() }
        viewModelScope.launch {
            val list = store.load()
            _state.update { it.copy(history = list) }
        }
    }

    fun setSourceMode(mode: SourceMode) {
        if (_state.value.isBusy) return
        _state.update { it.copy(sourceMode = mode) }
    }

    /** 开启翻译：源语种不支持翻译 / 与目标语种相同时拒绝开启（开关在 UI 上也是禁用的）。 */
    fun setTranslateOn(on: Boolean) = _state.update {
        if (on && translateBlockReason(it.srcLangCode, it.targetLang.code) != null) it
        else it.copy(translateOn = on)
    }

    /** 切换目标语种：与源语种冲突（同语种）时自动关闭翻译。 */
    fun setTargetLang(lang: Language) = _state.update {
        val blocked = translateBlockReason(it.srcLangCode, lang.code) != null
        it.copy(targetLang = lang, translateOn = if (blocked) false else it.translateOn)
    }

    /** 切换语音语种：该语种不支持翻译 / 与目标语种相同时自动关闭翻译。 */
    fun setSrcLang(code: String) = _state.update {
        val blocked = translateBlockReason(code, it.targetLang.code) != null
        it.copy(srcLangCode = code, translateOn = if (blocked) false else it.translateOn)
    }

    /**
     * 按当前源语种设置转写：auto → 自动检测语种（asrDetect）；选定语种 → 强制该语种
     * （模型 prompt 注入 language 指令，短语音不再依赖语种识别）。
     * 返回 (转写文本, 源语种码)。
     */
    private suspend fun transcribe(
        pcm: FloatArray,
        onPartial: ((String) -> Unit)?
    ): SpeechMTEngine.AsrResult {
        val src = _state.value.srcLangCode
        // 段内峰值归一化到 0.9：麦克风远场/轻声时 VAD 段电平偏低，低电平送 ASR 易被判空或误识。
        // 与视频字幕/实时听音路径（VideoSubtitle / ListenSubtitle）一致——本路径原先漏做，补齐。
        val norm = AudioDecoder.normalizePeak(pcm)
        return if (src == LANG_AUTO) {
            engine.asrDetect(norm, onPartial)
        } else {
            SpeechMTEngine.AsrResult(engine.asr(norm, src, onPartial), src)
        }
    }

    fun openHistory(id: String) {
        _state.update { it.copy(viewingRecordId = id, detailTab = HistoryDetailTab.CONTENT) }
    }

    fun closeHistory() {
        _state.update { it.copy(viewingRecordId = null) }
    }

    fun setDetailTab(tab: HistoryDetailTab) = _state.update { it.copy(detailTab = tab) }

    /**
     * 整理纪要：源内容取译文（有任何一段已翻译）否则取转写——与用户看到的主体文本一致；
     * LLM 流式生成，结果写回记录并持久化。
     */
    fun summarize(id: String) {
        if (_state.value.summarizing) return
        val rec = _state.value.history.firstOrNull { it.id == id } ?: return
        val hasTranslation = rec.segments.any { !it.translation.isNullOrBlank() }
        val content = rec.segments.mapNotNull { seg ->
            // 有译文取译文（个别段缺失时回退转写），否则取转写
            (if (hasTranslation) (seg.translation ?: seg.transcript) else seg.transcript)
                .takeUnless { it.isBlank() }
        }.joinToString("\n")
        if (content.isBlank()) {
            _state.update { it.copy(statusText = "记录无可用文本，无法整理纪要") }
            return
        }
        val llm = LlmEngine.get(getApplication())
        _state.update { it.copy(summarizing = true, summaryStream = "", todoStream = "", outlineStream = "",
            excerpt = emptyList(), loadingLlm = !llm.isLoaded,
            detailTab = HistoryDetailTab.SUMMARY, statusText = if (hasTranslation) "按译文整理纪要…" else "按转写整理纪要…") }
        summarizeJob?.cancel()
        summarizeJob = viewModelScope.launch {
            try {
                if (!llm.isLoaded) llm.warmUp()
                _state.update { it.copy(loadingLlm = false, statusText = "纪要生成中…") }
                // 抽取预热（毫秒级可见）→ 大纲 → 正文流式；超长转写内部自动降级逐段模式
                val summary = llm.summarizeSmart(
                    content, contentDesc = "录音转写",
                    onPreview = { s -> _state.update { it.copy(excerpt = s) } },
                    onOutline = { d -> _state.update { it.copy(outlineStream = it.outlineStream + d) } },
                    onPartial = { d -> _state.update { it.copy(summaryStream = it.summaryStream + d) } },
                    onTodo = { d -> _state.update { it.copy(todoStream = it.todoStream + d) } },
                    onProgress = { i, n ->
                        _state.update { it.copy(statusText = "纪要生成中…已梳理 $i/$n 段") }
                    }
                )
                if (summary.isNotBlank()) {
                    val updated = _state.value.history.map { r ->
                        if (r.id == id) r.copy(
                            summary = summary,
                            excerpt = _state.value.excerpt.joinToString("\n").takeIf { it.isNotBlank() }
                        ) else r
                    }
                    _state.update { it.copy(history = updated, statusText = "纪要已生成") }
                    store.save(updated)
                } else {
                    _state.update { it.copy(statusText = "纪要生成失败（模型已就绪但输出为空，详见 logcat Tag=LlmEngine）") }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                _state.update { it.copy(statusText = "已停止整理") }
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "summarize failed", e)
                _state.update { it.copy(statusText = "纪要生成失败：${e.message ?: "未知"}") }
            } finally {
                _state.update { it.copy(summarizing = false, loadingLlm = false) }
            }
        }
    }

    /** 停止纪要生成（保留已流式生成的部分：写回记录持久化）。 */
    fun stopSummarize() {
        val job = summarizeJob ?: return
        summarizeJob = null
        // 先取已生成的流式文本再取消：finally 里不再覆盖状态
        val partial = combinePointsTodo(_state.value.summaryStream, _state.value.todoStream)
        val recId = _state.value.viewingRecordId
        job.cancel()
        viewModelScope.launch {
            if (partial.isNotBlank() && recId != null) {
                val updated = _state.value.history.map { r ->
                    if (r.id == recId) r.copy(
                        summary = partial,
                        excerpt = _state.value.excerpt.joinToString("\n").takeIf { it.isNotBlank() }
                    ) else r
                }
                _state.update { it.copy(history = updated) }
                store.save(updated)
            }
        }
    }

    fun deleteHistory(id: String) {
        viewModelScope.launch {
            val rec = _state.value.history.firstOrNull { it.id == id }
            // 清理音频文件：完整 WAV + 各段 WAV（上传模式两者同文件，delete 幂等）
            rec?.audioPath?.let { runCatching { File(it).delete() } }
            rec?.segments?.forEach { seg ->
                seg.audioPath?.let { runCatching { File(it).delete() } }
            }
            val updated = _state.value.history.filterNot { it.id == id }
            _state.update {
                it.copy(
                    history = updated,
                    viewingRecordId = if (it.viewingRecordId == id) null else it.viewingRecordId
                )
            }
            store.save(updated)
        }
    }

    /**
     * 开始录音模式 → 实时转写/翻译。
     * 采集与推理解耦：录音协程只把 PCM 片段塞进有界 Channel（不阻塞麦克风），
     * 推理协程按序消费；推理跟不上时丢最旧片段（DROP_OLDEST）。首次需加载模型时
     * 先提示"模型加载中"，就绪后再开始采集。
     */
    fun startRecording() {
        val ctx = getApplication<Application>()
        if (!hasMicPermission(ctx)) {
            _state.update { it.copy(statusText = "缺少录音权限") }
            return
        }
        if (_state.value.isBusy) return
        val translateOn = _state.value.translateOn
        val targetLang = _state.value.targetLang

        val needLoad = !engine.isLoaded
        _state.update {
            it.copy(
                isBusy = true,
                isCapturing = true,
                isLoadingModel = needLoad,
                statusText = if (needLoad) "模型加载中…" else "正在录音…",
                currentSegments = emptyList(),
                uploadPreview = null,
                viewingRecordId = null
            )
        }

        val channel = Channel<SegmentChunk>(capacity = 8, onBufferOverflow = BufferOverflow.DROP_OLDEST)

        // 采集协程：只读麦克风、塞 Channel，推理快慢不影响录音流畅度
        recordingJob = viewModelScope.launch(Dispatchers.IO) {
            if (needLoad) {
                engine.warmUp()                      // 已在加载则等待完成，未加载则触发加载
                ensureActive()                       // 等待期间用户可能点了"停止录音"
                _state.update { it.copy(isLoadingModel = false, statusText = "正在录音…") }
            }
            try {
                runCatching {
                    recorder.stream().collect { pcm ->
                        // 每段 PCM 落盘成独立 WAV（供单段播放 + 停止后拼接完整音频）
                        val wavPath = File(audioDir, "seg_${UUID.randomUUID()}.wav").absolutePath
                        runCatching { WavIo.writeWav(wavPath, pcm) }
                            .onFailure { Log.e(TAG, "write seg wav failed", it) }
                        channel.trySend(SegmentChunk(pcm, wavPath))
                    }
                }.onFailure { Log.e(TAG, "recording error", it) }
            } finally {
                channel.close()                      // 采集结束（正常停止 / 取消）→ 让推理排空后退出
            }
        }

        // 推理协程：按序消费 PCM → ASR；翻译与 ASR 解耦——每段转写一出来就丢给
        // 独立子协程翻译，推理协程立刻去吃下一段 PCM，不被翻译拖住。所有翻译子协程
        // 在推理循环结束后统一 join，保证落库时译文已齐（ASR 与 MT 是不同 OrtSession，
        // 可安全并发）。
        inferenceJob = viewModelScope.launch(Dispatchers.IO) {
            var segCount = 0
            val translationJobs = mutableListOf<Job>()
            for (chunk in channel) {
                ensureActive()
                segCount++
                val segId = UUID.randomUUID().toString()
                // token 级流式：音频段一进来就先建空段（带音频路径，UI 立即可播放），decoder 每生成一个 token 就逐字刷新 transcript
                _state.update {
                    it.copy(
                        statusText = "识别中…(段 $segCount)",
                        currentSegments = it.currentSegments + Segment(
                            id = segId, transcript = "", translation = null,
                            audioPath = chunk.audioPath, audioSamples = chunk.pcm.size
                        )
                    )
                }
                // ASR：源语种按配置（auto=自动检测；选定语种=强制转写，短语音不依赖语种识别）
                val asrResult = transcribe(chunk.pcm) { partial ->
                    // partial 含 decoder 当前已生成文本（可能带 language 前缀，最终会清洗）
                    if (partial.isNotBlank()) {
                        _state.update { s ->
                            s.copy(
                                currentSegments = s.currentSegments.map {
                                    if (it.id == segId) it.copy(transcript = partial) else it
                                }
                            )
                        }
                    }
                }
                val asr = asrResult.text
                if (asr.isBlank()) {
                    // 静音/无语音：移除占位空段
                    _state.update { s ->
                        s.copy(currentSegments = s.currentSegments.filterNot { it.id == segId })
                    }
                    continue
                }
                // 用清洗后的最终文本覆盖（去掉 language 前缀等）
                _state.update { s ->
                    s.copy(
                        currentSegments = s.currentSegments.map {
                            if (it.id == segId) it.copy(transcript = asr) else it
                        }
                    )
                }
                // 源语种不支持翻译 / 与目标语种相同 → 不做翻译（自动检测模式下按检测到的语种判定）
                if (translateOn && needsTranslation(asrResult.language ?: "zh", targetLang.code)) {
                    // 翻译异步化：另起子协程，不阻塞下一段 ASR
                    // 源语种用 ASR 检测结果（未知回退 "zh"）；token 级流式回调边生成边刷新译文
                    val tgtLang = targetLang
                    val srcLangCode = asrResult.language ?: "zh"
                    translationJobs += launch {
                        val mt = engine.translate(asr, srcLangCode, tgtLang.code) { partial ->
                            if (partial.isNotBlank()) {
                                _state.update { s ->
                                    s.copy(
                                        currentSegments = s.currentSegments.map {
                                            if (it.id == segId) it.copy(translation = partial) else it
                                        }
                                    )
                                }
                            }
                        }
                        if (mt.isNotBlank()) {
                            _state.update { s ->
                                s.copy(
                                    currentSegments = s.currentSegments.map {
                                        if (it.id == segId) it.copy(translation = mt) else it
                                    }
                                )
                            }
                        }
                    }
                }
            }
            // Channel 已关闭且排空：等所有翻译子协程完成后，再落库（译文不丢）
            translationJobs.joinAll()
            persistCurrentAsRecord(SourceMode.Mic)
            _state.update { it.copy(isBusy = false, isCapturing = false, isLoadingModel = false, statusText = "") }
        }
    }

    /**
     * 停止录音：结束采集，让推理协程把缓冲中剩余片段处理完后再保存（不丢已采集音频）。
     * 重复点击为幂等（已停止则忽略）。isBusy 在推理排空后由推理协程清除。
     */
    fun stopRecording() {
        if (!_state.value.isCapturing) return
        _state.update { it.copy(isCapturing = false, isLoadingModel = false, statusText = "识别中…") }
        recorder.stop()   // 采集流退出 → channel.close() → 推理协程排空后 persist + 清忙态
    }

    /**
     * 上传音频模式 → 先在结果区最顶侧展示/预览整条音频（文件名、时长、整条播放），
     * 再**流式解码 + VAD 切分**，逐段 ASR（流式上屏）+ 翻译，最后入库。
     *
     * 为什么必须切分：ASR 单次输入有 ~12s 上限，整条音频直接送进去会被截断，
     * 且长音频整段解码持有 PCM（分钟级 ~MB~百MB）+ 长序列 prefill 会 OOM/崩溃。
     * 这里与视频字幕管线一致：流式解码（不持有整条 PCM）+ 能量门限 VAD 增量切段，
     * 段一出来就识别，单段失败不影响整条。
     */
    fun processUploadedAudio(uri: Uri) {
        if (_state.value.isBusy) return
        val translateOn = _state.value.translateOn
        val targetLang = _state.value.targetLang
        val ctx = getApplication<Application>()
        _state.update {
            it.copy(
                isBusy = true,
                statusText = "读取音频中…",
                currentSegments = emptyList(),
                uploadPreview = null,
                viewingRecordId = null
            )
        }

        // 翻译任务（viewModelScope 子协程）：声明在 runCatching 外，失败时可一并取消
        val translationJobs = mutableListOf<Job>()
        processingJob = viewModelScope.launch {
            runCatching {
                // ① 先取元信息并顶侧展示：不等解码，用户立刻能看到文件名/时长并可整条试听
                val meta = withContext(Dispatchers.IO) { probeAudioMeta(ctx, uri) }
                _state.update { it.copy(uploadPreview = meta) }

                if (!engine.isLoaded) {
                    _state.update { it.copy(isLoadingModel = true, statusText = "模型加载中…") }
                    engine.warmUp()
                    ensureActive()
                    _state.update { it.copy(isLoadingModel = false) }
                }

                // ② 第一遍解码：扫整条帧 RMS 的 10 分位作噪声底，定 VAD 门限
                //    （不同录音电平差异极大，固定门限要么漏切一大片，要么把噪声切成碎段）
                _state.update { it.copy(statusText = "分析音量…") }
                val threshold = withContext(Dispatchers.IO) { analyzeVoiceThreshold(ctx, uri) }
                Log.i(TAG, "upload vad threshold=$threshold")

                // ③ 第二遍解码：增量 VAD 切段，段一出来立即 ASR + 翻译（翻译异步，不阻塞下段识别）
                val vad = StreamingVadSegmenter(
                    EnergyVadDetector(512, threshold),
                    pauseMs = 1300, maxSegMs = 10_000
                )
                var segCount = 0
                _state.update { it.copy(statusText = "识别中…") }
                withContext(Dispatchers.IO) {
                    AudioDecoder.decodeStreamPcm16kMono(ctx, uri) { chunk ->
                        for (vseg in vad.feed(chunk)) {
                            ensureActive()
                            handleUploadSegment(vseg.pcm, segCount++, translateOn, targetLang, translationJobs)
                        }
                    }
                    // VAD 收尾：发出末尾未结句的语音段
                    for (vseg in vad.flush()) {
                        ensureActive()
                        handleUploadSegment(vseg.pcm, segCount++, translateOn, targetLang, translationJobs)
                    }
                }
                Log.i(TAG, "upload finished: vad segments=$segCount kept=${_state.value.currentSegments.size}")
                // 译文可能还在生成：等齐了再入库（保证历史记录里的译文不丢）
                translationJobs.joinAll()
                val kept = _state.value.currentSegments.size
                if (kept == 0) {
                    _state.update {
                        it.copy(isBusy = false, statusText = "未检测到语音（音频过安静或全是背景音）")
                    }
                } else {
                    // 上传音频的历史名直接用源文件名（去掉扩展名）：翻历史时一眼认出是哪条
                    val upName = _state.value.uploadPreview?.fileName
                        ?.substringBeforeLast('.', missingDelimiterValue = "")
                        ?.takeIf { it.isNotBlank() }
                    persistCurrentAsRecord(SourceMode.Upload, upName)
                    _state.update { it.copy(isBusy = false, statusText = "已完成 · 共 $kept 段") }
                }
            }.onFailure {
                Log.e(TAG, "process upload failed", it)
                // 翻译任务是 viewModelScope 子协程：整条任务失败/取消时一并停掉，避免残留推理继续改 UI
                translationJobs.forEach { runCatching { it.cancel() } }
                _state.update { s ->
                    s.copy(isBusy = false, isLoadingModel = false,
                        statusText = "处理失败：${it.message ?: "未知错误"}")
                }
            }
        }
    }

    /**
     * 单段上传音频处理：段 WAV 落盘 → ASR（token 级流式上屏）→ 翻译（异步 job）。
     * 单段异常被吞掉并丢弃该占位段，不影响整条音频后续段落。
     * @return true=该段有文本已保留；false=空段/失败段被丢弃。
     */
    private suspend fun handleUploadSegment(
        pcm: FloatArray,
        index: Int,
        translateOn: Boolean,
        targetLang: Language,
        translationJobs: MutableList<Job>
    ): Boolean {
        val segId = UUID.randomUUID().toString()
        // 段音频独立落盘（供单段回听；入库时拼成整条）
        val wavPath = File(audioDir, "upl_${UUID.randomUUID()}.wav").absolutePath
        withContext(Dispatchers.IO) {
            runCatching { WavIo.writeWav(wavPath, pcm) }
                .onFailure { Log.e(TAG, "write upload seg wav failed", it) }
        }
        _state.update {
            it.copy(
                statusText = "识别中…(段 ${index + 1})",
                currentSegments = it.currentSegments + Segment(
                    id = segId, transcript = "", translation = null,
                    audioPath = wavPath, audioSamples = pcm.size
                )
            )
        }
        val ok = runCatching {
            // ASR：源语种按配置（auto=自动检测；选定语种=强制转写）
            val asrResult = transcribe(pcm) { partial ->
                if (partial.isNotBlank()) {
                    _state.update { s ->
                        s.copy(currentSegments = s.currentSegments.map {
                            if (it.id == segId) it.copy(transcript = partial) else it
                        })
                    }
                }
            }
            val asr = asrResult.text
            if (asr.isBlank()) return@runCatching false     // 静音/噪声段：丢弃占位
            _state.update { s ->
                s.copy(currentSegments = s.currentSegments.map {
                    if (it.id == segId) it.copy(transcript = asr) else it
                })
            }
            // 源语种不支持翻译 / 与目标语种相同 → 不做翻译
            if (translateOn && needsTranslation(asrResult.language ?: "zh", targetLang.code)) {
                val srcLangCode = asrResult.language ?: "zh"
                translationJobs += viewModelScope.launch {
                    val mt = engine.translate(asr, srcLangCode, targetLang.code) { partial ->
                        if (partial.isNotBlank()) {
                            _state.update { s ->
                                s.copy(currentSegments = s.currentSegments.map {
                                    if (it.id == segId) it.copy(translation = partial) else it
                                })
                            }
                        }
                    }
                    if (mt.isNotBlank()) {
                        _state.update { s ->
                            s.copy(currentSegments = s.currentSegments.map {
                                if (it.id == segId) it.copy(translation = mt) else it
                            })
                        }
                    }
                }
            }
            true
        }.getOrElse { e ->
            if (e is kotlinx.coroutines.CancellationException) throw e   // 取消不属于"单段失败"
            Log.e(TAG, "upload seg $index failed", e)
            false
        }
        if (!ok) {
            _state.update { s -> s.copy(currentSegments = s.currentSegments.filterNot { it.id == segId }) }
        }
        return ok
    }

    /** 读取上传音频的展示元信息（文件名/时长/大小）：只用系统解码器读元信息，不解码整条 PCM。 */
    private fun probeAudioMeta(ctx: Context, uri: Uri): UploadFilePreview {
        val retriever = android.media.MediaMetadataRetriever()
        var durationMs = 0L
        try {
            retriever.setDataSource(ctx, uri)
            durationMs = retriever.extractMetadata(
                android.media.MediaMetadataRetriever.METADATA_KEY_DURATION
            )?.toLongOrNull() ?: 0L
        } catch (e: Exception) {
            Log.w(TAG, "probe duration failed", e)
        } finally {
            runCatching { retriever.release() }
        }
        var name: String? = null
        var size = 0L
        runCatching {
            ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    val nIdx = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    val sIdx = c.getColumnIndex(android.provider.OpenableColumns.SIZE)
                    if (nIdx >= 0) name = c.getString(nIdx)
                    if (sIdx >= 0) size = c.getLong(sIdx)
                }
            }
        }.onFailure { Log.w(TAG, "query display name failed", it) }
        return UploadFilePreview(
            fileName = name ?: uri.lastPathSegment ?: "上传音频",
            uriString = uri.toString(),
            durationMs = durationMs,
            sizeBytes = size
        )
    }

    /**
     * 流式扫一遍整条音频估噪声底 → VAD 门限 = max(噪声底×3, 0.012)。
     * 绝对下限防止极安静素材把门限压到量化噪声里；系数 3 保证明显人声都能触发。
     */
    private suspend fun analyzeVoiceThreshold(ctx: Context, uri: Uri): Float {
        val rmsList = ArrayList<Float>(4096)
        val pending = ArrayList<Float>(256)
        runCatching {
            AudioDecoder.decodeStreamPcm16kMono(ctx, uri) { chunk ->
                for (s in chunk) {
                    pending.add(s)
                    if (pending.size >= 256) {
                        var ss = 0.0
                        for (v in pending) ss += (v * v).toDouble()
                        rmsList.add(sqrt(ss / 256).toFloat())
                        pending.subList(0, 256).clear()
                    }
                }
            }
        }.onFailure { Log.w(TAG, "analyze volume failed", it) }
        if (rmsList.isEmpty()) return 0.012f
        rmsList.sort()
        val noiseFloor = rmsList[(rmsList.size * 0.10f).toInt().coerceIn(0, rmsList.size - 1)]
        return (noiseFloor * 3f).coerceAtLeast(0.012f)
    }

    private fun persistCurrentAsRecord(source: SourceMode, nameOverride: String? = null) {
        val segs = _state.value.currentSegments
        if (segs.isEmpty()) return
        // 完整音频：各段 WAV 拼接（上传模式同样按 VAD 切成多段，录音模式沿用同样逻辑）；
        // 只有一段时直接复用该段文件，避免多写一份冗余 WAV
        val segPaths = segs.mapNotNull { it.audioPath }
        val fullAudioPath = when {
            segPaths.isEmpty() -> null
            segPaths.size == 1 -> segPaths[0]
            else -> {
                val p = File(audioDir, "full_${UUID.randomUUID()}.wav").absolutePath
                runCatching { WavIo.concatWavs(p, segPaths) }
                    .onFailure { Log.e(TAG, "concat wav failed", it) }
                p
            }
        }
        val record = SpeechMTRecord(
            createdAt = System.currentTimeMillis(),
            source = source,
            targetLang = if (_state.value.translateOn) _state.value.targetLang.code else null,
            segments = segs,
            audioPath = fullAudioPath,
            titleOverride = nameOverride
        )
        val newList = listOf(record) + _state.value.history
        _state.update { it.copy(history = newList) }
        viewModelScope.launch { store.save(newList) }
    }

    override fun onCleared() {
        recorder.stop()
        recordingJob?.cancel()
        inferenceJob?.cancel()
        processingJob?.cancel()
    }

    private fun hasMicPermission(context: Context): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    companion object {
        private const val TAG = "SpeechMTViewModel"
    }
}

// ──────────────────────────────────────────────────────────────────
// 历史记录持久化（JSON 文件，简单可靠）
// ──────────────────────────────────────────────────────────────────
private class HistoryStore(private val context: Context) {
    private val file: File get() = File(context.filesDir, "speech_mt_history.json")

    suspend fun load(): List<SpeechMTRecord> = withContext(Dispatchers.IO) {
        if (!file.exists()) return@withContext emptyList()
        runCatching {
            val arr = JSONArray(file.readText())
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                val segArr = o.getJSONArray("segments")
                val segs = (0 until segArr.length()).map { j ->
                    val s = segArr.getJSONObject(j)
                    Segment(
                        id = s.getString("id"),
                        transcript = s.getString("transcript"),
                        translation = s.optString("translation").takeIf { it.isNotEmpty() },
                        audioPath = s.optString("audioPath").takeIf { it.isNotEmpty() },
                        audioSamples = s.optInt("audioSamples", 0)
                    )
                }
                SpeechMTRecord(
                    id = o.getString("id"),
                    createdAt = o.getLong("createdAt"),
                    source = SourceMode.valueOf(o.getString("source")),
                    targetLang = o.optString("targetLang").takeIf { it.isNotEmpty() },
                    segments = segs,
                    audioPath = o.optString("audioPath").takeIf { it.isNotEmpty() },
                    summary = o.optString("summary").takeIf { it.isNotEmpty() },
                    excerpt = o.optString("excerpt").takeIf { it.isNotEmpty() },
                    titleOverride = o.optString("titleOverride").takeIf { it.isNotEmpty() }
                )
            }
        }.getOrElse { emptyList() }
    }

    suspend fun save(list: List<SpeechMTRecord>) = withContext(Dispatchers.IO) {
        val arr = JSONArray()
        list.forEach { r ->
            val o = JSONObject()
                .put("id", r.id)
                .put("createdAt", r.createdAt)
                .put("source", r.source.name)
                .put("targetLang", r.targetLang ?: "")
                .put("audioPath", r.audioPath ?: "")
                .put("summary", r.summary ?: "")
                .put("excerpt", r.excerpt ?: "")
                .put("titleOverride", r.titleOverride ?: "")
            val segs = JSONArray()
            r.segments.forEach { s ->
                segs.put(
                    JSONObject()
                        .put("id", s.id)
                        .put("transcript", s.transcript)
                        .put("translation", s.translation ?: "")
                        .put("audioPath", s.audioPath ?: "")
                        .put("audioSamples", s.audioSamples)
                )
            }
            o.put("segments", segs)
            arr.put(o)
        }
        file.writeText(arr.toString())
    }
}

// ──────────────────────────────────────────────────────────────────
// UI
// ──────────────────────────────────────────────────────────────────

/** 把一条历史记录导出为 txt 文本。 */
private fun buildRecordText(rec: SpeechMTRecord): String {
    val sb = StringBuilder()
    val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(rec.createdAt))
    sb.append("录音翻译 · ").append(time).append('\n')
    sb.append("来源：").append(rec.source.label).append('\n')
    if (rec.targetLang != null) sb.append("目标语言：").append(rec.targetLang).append('\n')
    sb.append('\n')
    rec.segments.forEachIndexed { idx, seg ->
        sb.append("【段${idx + 1}】\n")
        sb.append("转写：").append(seg.transcript).append('\n')
        if (seg.translation != null) sb.append("翻译：").append(seg.translation).append('\n')
        sb.append('\n')
    }
    return sb.toString().trimEnd()
}

/** 仅源语言转写文本。 */
private fun buildSourceText(rec: SpeechMTRecord): String {
    val sb = StringBuilder()
    val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(rec.createdAt))
    sb.append("录音翻译 · 源文本 · ").append(time).append('\n')
    sb.append("来源：").append(rec.source.label).append('\n')
    sb.append('\n')
    rec.segments.forEachIndexed { idx, seg ->
        sb.append("【段${idx + 1}】").append(seg.transcript).append('\n')
    }
    return sb.toString().trimEnd()
}

/** 仅目标语言译文文本（未翻译的段标注无翻译）。 */
private fun buildTargetText(rec: SpeechMTRecord): String {
    val sb = StringBuilder()
    val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(rec.createdAt))
    sb.append("录音翻译 · 译文 · ").append(time).append('\n')
    sb.append("来源：").append(rec.source.label).append('\n')
    if (rec.targetLang != null) sb.append("目标语言：").append(rec.targetLang).append('\n')
    sb.append('\n')
    rec.segments.forEachIndexed { idx, seg ->
        sb.append("【段${idx + 1}】").append(seg.translation ?: "（无翻译）").append('\n')
    }
    return sb.toString().trimEnd()
}

/** 纪要总结导出文本：头部信息 + AI 纪要正文。 */
private fun buildSummaryText(rec: SpeechMTRecord): String {
    val time = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(rec.createdAt))
    val sb = StringBuilder()
    sb.append("录音纪要 · ").append(time).append('\n')
    sb.append("来源：").append(rec.source.label).append('\n')
    if (rec.targetLang != null) sb.append("纪要语言：").append(rec.targetLang).append('\n')
    sb.append('\n')
    sb.append(rec.summary ?: "")
    return sb.toString().trimEnd()
}

/** 生成安全的下载文件名。 */
private fun buildFileName(rec: SpeechMTRecord, ext: String): String {
    // 用记录名（有纪要=会议主题）作文件名，去掉文件系统非法字符与空白
    val title = rec.title
        .replace(Regex("[\\\\/:*?\"<>|\\s]+"), "")
        .take(16)
        .ifBlank { "录音" }
    val time = SimpleDateFormat("yyyyMMdd_HHmm", Locale.getDefault()).format(Date(rec.createdAt))
    return "${title}_$time.$ext"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SpeechMTScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val vm: SpeechMTViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance(
            (context.applicationContext as Application)
        )
    )
    val state by vm.state.collectAsState()

    val micPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.startRecording()
    }

    // 用 SAF OpenDocument 而不是 GetContent：GetContent 给的 Uri 只有**临时**读权限，
    // 而上传音频要解码两遍（先扫整条估音量、再切段识别），长音频跑到第二遍时权限可能已被
    // 回收 → 解码失败 → 表现为"上传没反应 / 识别不了"。OpenDocument 可申请持久授权。
    val pickAudioLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }.onFailure { Log.w("SpeechMTScreen", "persist uri permission failed", it) }
            vm.processUploadedAudio(uri)
        }
    }

    // 待下载的历史记录 + 类型（CreateDocument 先弹系统保存对话框，回调里据此写文件）
    var pendingDownloadRec by remember { mutableStateOf<SpeechMTRecord?>(null) }
    var pendingDownloadType by remember { mutableStateOf<DownloadType?>(null) }
    var showDownloadDialog by remember { mutableStateOf(false) }

    val audioLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("audio/wav")
    ) { uri: Uri? ->
        val rec = pendingDownloadRec
        val type = pendingDownloadType
        pendingDownloadRec = null
        pendingDownloadType = null
        if (uri != null && rec != null && type == DownloadType.AUDIO && rec.audioPath != null) {
            runCatching {
                context.contentResolver.openOutputStream(uri)?.use { os ->
                    File(rec.audioPath).inputStream().use { it.copyTo(os) }
                }
            }.onFailure { Log.e("SpeechMT", "download audio failed", it) }
        }
    }

    val textLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain")
    ) { uri: Uri? ->
        val rec = pendingDownloadRec
        val type = pendingDownloadType
        pendingDownloadRec = null
        pendingDownloadType = null
        if (uri != null && rec != null && type != null) {
            val text = when (type) {
                DownloadType.SOURCE_TEXT -> buildSourceText(rec)
                DownloadType.TARGET_TEXT -> buildTargetText(rec)
                DownloadType.BILINGUAL -> buildRecordText(rec)
                DownloadType.SUMMARY -> buildSummaryText(rec)
                DownloadType.AUDIO -> ""
            }
            runCatching {
                context.contentResolver.openOutputStream(uri)?.use { os ->
                    os.write(text.toByteArray(Charsets.UTF_8))
                }
            }.onFailure { Log.e("SpeechMT", "download text failed", it) }
        }
    }

    // 抽屉展开状态（由 SpeechMTScreen 持有，便于选中/切换后自动收起）
    var historyOpen by remember { mutableStateOf(false) }
    var configOpen by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            // 自定义头部：标题左侧 + 右侧「上图标下名称」三操作块（配置 / 历史 / 开始录音），
            // 替代通用 AgentHeader——把配置与历史收纳进抽屉，主区留给实时结果
            SpeechMTHeader(
                state = state,
                configOpen = configOpen,
                historyOpen = historyOpen,
                historyCount = state.history.size,
                onBack = onBack,
                onToggleConfig = { configOpen = !configOpen },
                onToggleHistory = { historyOpen = !historyOpen },
                onStartMic = {
                    configOpen = false; historyOpen = false
                    if (ContextCompat.checkSelfPermission(
                            context, Manifest.permission.RECORD_AUDIO
                        ) == PackageManager.PERMISSION_GRANTED
                    ) vm.startRecording()
                    else micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                },
                onStopMic = vm::stopRecording,
                onPickAudio = {
                    configOpen = false; historyOpen = false
                    pickAudioLauncher.launch(arrayOf("audio/*"))
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // ── 配置抽屉 ──
            AnimatedVisibility(
                visible = configOpen,
                enter = expandVertically(
                    expandFrom = Alignment.Top,
                    animationSpec = tween(durationMillis = 220)
                ) + fadeIn(animationSpec = tween(220)),
                exit = shrinkVertically(
                    shrinkTowards = Alignment.Top,
                    animationSpec = tween(durationMillis = 180)
                ) + fadeOut(animationSpec = tween(180))
            ) {
                ConfigDrawer(
                    state = state,
                    enabled = !state.isBusy,
                    onSourceMode = vm::setSourceMode,
                    onToggleTranslate = vm::setTranslateOn,
                    onTargetLang = vm::setTargetLang,
                    onSrcLang = vm::setSrcLang
                )
            }

            // ── 历史抽屉（下拉展开） ──
            AnimatedVisibility(
                visible = historyOpen,
                enter = expandVertically(
                    expandFrom = Alignment.Top,
                    animationSpec = tween(durationMillis = 220)
                ) + fadeIn(animationSpec = tween(220)),
                exit = shrinkVertically(
                    shrinkTowards = Alignment.Top,
                    animationSpec = tween(durationMillis = 180)
                ) + fadeOut(animationSpec = tween(180))
            ) {
                HistoryDrawer(
                    history = state.history,
                    selectedId = state.viewingRecordId,
                    onSelect = { id ->
                        vm.openHistory(id)
                        historyOpen = false
                    },
                    onDownload = { rec ->
                        pendingDownloadRec = rec
                        showDownloadDialog = true
                    },
                    onDelete = vm::deleteHistory,
                    onClose = { historyOpen = false }
                )
            }

            // ── 实时 / 历史结果 ──
            ResultPanel(
                state = state,
                onCloseHistory = vm::closeHistory,
                onTabSelect = vm::setDetailTab,
                onSummarize = vm::summarize,
                onStopSummarize = vm::stopSummarize,
                onDownload = { rec ->
                    pendingDownloadRec = rec
                    showDownloadDialog = true
                },
                onDelete = { id -> vm.deleteHistory(id) },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )
        }

        // 下载类型选择对话框（音频/源文本/译文/原译对照/纪要总结）
        if (showDownloadDialog) {
            val rec = pendingDownloadRec
            if (rec != null) {
                DownloadTypeDialog(
                    hasAudio = rec.audioPath != null,
                    hasSummary = rec.summary != null,
                    onPick = { type ->
                        showDownloadDialog = false
                        pendingDownloadType = type
                        when (type) {
                            DownloadType.AUDIO -> {
                                if (rec.audioPath != null) {
                                    audioLauncher.launch(buildFileName(rec, "wav"))
                                } else {
                                    pendingDownloadType = null
                                }
                            }
                            else -> textLauncher.launch(buildFileName(rec, "txt"))
                        }
                    },
                    onDismiss = {
                        showDownloadDialog = false
                        pendingDownloadRec = null
                        pendingDownloadType = null
                    }
                )
            }
        }
    }
}

@Composable
private fun HistoryDrawer(
    history: List<SpeechMTRecord>,
    selectedId: String?,
    onSelect: (String) -> Unit,
    onDownload: (SpeechMTRecord) -> Unit,
    onDelete: (String) -> Unit,
    onClose: () -> Unit,
    modifier: Modifier = Modifier
) {
    ElevatedCard(
        modifier = modifier
            .fillMaxWidth()
            // 限制最大高度，避免遮挡结果区
            .heightIn(max = 320.dp),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            SectionTitle(
                icon = Icons.Default.History,
                text = "历史记录",
                accent = AgentOrange,
                modifier = Modifier.padding(start = 14.dp, end = 6.dp, top = 8.dp, bottom = 8.dp),
                trailing = {
                    if (history.isNotEmpty()) {
                        Text(
                            text = "共 ${history.size} 条",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                    }
                    IconButton(onClick = onClose, modifier = Modifier.size(32.dp)) {
                        Icon(
                            imageVector = Icons.Default.KeyboardArrowUp,
                            contentDescription = "收起",
                            tint = MaterialTheme.colorScheme.outline
                        )
                    }
                }
            )
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            if (history.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 28.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "暂无记录",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(history, key = { it.id }) { rec ->
                        HistoryItem(
                            rec = rec,
                            selected = rec.id == selectedId,
                            onSelect = { onSelect(rec.id) },
                            onDownload = { onDownload(rec) },
                            onDelete = { onDelete(rec.id) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun HistoryItem(
    rec: SpeechMTRecord,
    selected: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
    modifier: Modifier = Modifier
) {
    val shape = RoundedCornerShape(14.dp)
    val borderColor = if (selected) AgentOrange else MaterialTheme.colorScheme.outlineVariant
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 10.dp, vertical = 4.dp)
            .clip(shape)
            .background(
                if (selected) AgentOrange.copy(alpha = 0.12f)
                else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
            )
            .border(
                width = if (selected) 1.5.dp else 0.5.dp,
                color = borderColor,
                shape = shape
            )
            .clickable(onClick = onSelect)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // 来源角标：录音（麦克风）/ 上传（文件）
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(AgentOrange.copy(alpha = if (selected) 0.24f else 0.14f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (rec.source == SourceMode.Mic) Icons.Default.Mic
                else Icons.Default.UploadFile,
                contentDescription = null,
                tint = AgentOrange,
                modifier = Modifier.size(20.dp)
            )
        }
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = rec.title,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(modifier = Modifier.height(5.dp))
            // 元信息：时间 · 段数 · 译文语种 · 纪要标记
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                MetaChip(
                    text = rec.timeLabel,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    container = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.06f)
                )
                MetaChip(
                    text = "${rec.segments.size} 段",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    container = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.06f)
                )
                if (rec.targetLang != null) {
                    MetaChip(
                        text = "译 ${langOf(rec.targetLang)?.name ?: rec.targetLang}",
                        tint = MaterialTheme.colorScheme.primary,
                        container = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                        icon = Icons.Default.Translate
                    )
                }
                if (rec.summary != null) {
                    MetaChip(
                        text = "有纪要",
                        tint = Color(0xFF7C3AED),
                        container = Color(0xFF7C3AED).copy(alpha = 0.12f),
                        icon = Icons.Default.AutoAwesome
                    )
                }
            }
        }
        Column(horizontalAlignment = Alignment.End) {
            IconButton(onClick = onDownload, modifier = Modifier.size(36.dp)) {
                Icon(
                    imageVector = Icons.Default.Download,
                    contentDescription = "下载",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(18.dp)
                )
            }
            IconButton(onClick = onDelete, modifier = Modifier.size(36.dp)) {
                Icon(
                    imageVector = Icons.Default.DeleteOutline,
                    contentDescription = "删除",
                    tint = Color(0xFFB3261E),
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}

/** 历史条目里的小信息块（时间 / 段数 / 译文语种 / 纪要）。 */
@Composable
private fun MetaChip(
    text: String,
    tint: Color,
    container: Color,
    icon: ImageVector? = null
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(container)
            .padding(horizontal = 6.dp, vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        if (icon != null) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = tint,
                modifier = Modifier.size(11.dp)
            )
        }
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = tint,
            maxLines = 1
        )
    }
}

/**
 * 录音翻译智能体专用头部：左侧标题、右侧「上图标下名称」三操作块（配置 / 历史 / 开始录音）。
 * 主操作块随状态切换：录音中→停止、排空识别中→禁用转圈、上传模式→选文件。
 */
@Composable
private fun SpeechMTHeader(
    state: SpeechMTUiState,
    configOpen: Boolean,
    historyOpen: Boolean,
    historyCount: Int,
    onBack: () -> Unit,
    onToggleConfig: () -> Unit,
    onToggleHistory: () -> Unit,
    onStartMic: () -> Unit,
    onStopMic: () -> Unit,
    onPickAudio: () -> Unit
) {
    val gradient = Brush.linearGradient(listOf(AgentOrange, lerp(AgentOrange, Color.White, 0.22f)))
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
            // 返回按钮：半透明白底 + 白色箭头（与 AgentHeader 一致）
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
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "录音翻译智能体",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
            // ── 右侧操作块：配置 / 历史 / 主操作 ──
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                HeaderAction(icon = Icons.Default.Tune, label = "配置",
                    highlighted = configOpen, onClick = onToggleConfig)
                HeaderAction(icon = Icons.Default.History, label = "历史",
                    highlighted = historyOpen, badge = historyCount.takeIf { it > 0 }?.toString(),
                    onClick = onToggleHistory)
                when (state.sourceMode) {
                    SourceMode.Mic -> when {
                        state.isCapturing -> HeaderAction(
                            icon = Icons.Default.Stop, label = "停止", highlighted = true, onClick = onStopMic)
                        state.isBusy -> HeaderAction(  // 停止后排空识别中：禁用转圈
                            icon = null, label = "识别中", enabled = false, onClick = {})
                        else -> HeaderAction(
                            icon = Icons.Default.Mic, label = "录音", onClick = onStartMic)
                    }
                    SourceMode.Upload -> HeaderAction(
                        icon = Icons.Default.UploadFile, label = "上传",
                        enabled = !state.isBusy, onClick = onPickAudio)
                }
            }
        }
    }
}

/** 头部单个操作块：上图标（半透明白底圆角块）下名称。highlighted 时底色更亮。 */
@Composable
private fun HeaderAction(
    icon: ImageVector?,
    label: String,
    highlighted: Boolean = false,
    enabled: Boolean = true,
    badge: String? = null,
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
                Icon(
                    imageVector = icon,
                    contentDescription = label,
                    tint = Color.White,
                    modifier = Modifier.size(20.dp)
                )
            } else {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = Color.White
                )
            }
            if (badge != null) {
                Text(
                    text = badge,
                    style = MaterialTheme.typography.labelSmall,
                    color = Color.White,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(1.dp)
                        .clip(RoundedCornerShape(6.dp))
                        .background(AgentOrange)
                        .padding(horizontal = 4.dp)
                )
            }
        }
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = Color.White.copy(alpha = if (enabled) 0.95f else 0.5f)
        )
    }
}

@Composable
private fun ConfigDrawer(
    state: SpeechMTUiState,
    enabled: Boolean,
    onSourceMode: (SourceMode) -> Unit,
    onToggleTranslate: (Boolean) -> Unit,
    onTargetLang: (Language) -> Unit,
    onSrcLang: (String) -> Unit
) {
    var showLangPicker by remember { mutableStateOf(false) }
    var showSrcPicker by remember { mutableStateOf(false) }

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle(
                icon = Icons.Default.Tune,
                text = "配置",
                accent = AgentOrange
            )

            // 1) 音频输入源
            ConfigRow(label = "音频输入") {
                SegmentedSwitch(
                    options = listOf(SourceMode.Mic.label, SourceMode.Upload.label),
                    selectedIndex = if (state.sourceMode == SourceMode.Mic) 0 else 1,
                    enabled = enabled,
                    onSelected = { idx ->
                        onSourceMode(if (idx == 0) SourceMode.Mic else SourceMode.Upload)
                    }
                )
            }

            // 2) 语音语种（自动识别 / 指定语种：短语音语种识别更稳）
            ConfigRow(label = "语音语种") {
                AssistChip(
                    onClick = { if (enabled) showSrcPicker = true },
                    label = { Text(srcLangLabel(state.srcLangCode)) },
                    leadingIcon = {
                        Icon(
                            if (state.srcLangCode == LANG_AUTO) Icons.Default.AutoAwesome
                            else Icons.Default.RecordVoiceOver,
                            contentDescription = null, modifier = Modifier.size(16.dp)
                        )
                    }
                )
            }

            // 3) 是否翻译
            val blockReason = translateBlockReason(state.srcLangCode, state.targetLang.code)
            ConfigRow(label = "翻译") {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Switch(
                        checked = state.translateOn && blockReason == null,
                        onCheckedChange = { onToggleTranslate(it) },
                        enabled = enabled && blockReason == null
                    )
                    Text(
                        if (blockReason != null) "不可用"
                        else if (state.translateOn) "开启" else "关闭",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )

                    if (state.translateOn || blockReason != null) {
                        Spacer(modifier = Modifier.width(8.dp))
                        AssistChip(
                            onClick = { if (enabled) showLangPicker = true },
                            enabled = enabled,
                            label = { Text("${state.targetLang.flag} ${state.targetLang.name}") },
                            leadingIcon = {
                                Icon(Icons.Default.Translate, contentDescription = null,
                                    modifier = Modifier.size(16.dp))
                            }
                        )
                    }
                }
            }

            // 不可翻译时给出原因提示（换目标语种可解除"同语种"限制）
            if (blockReason != null) {
                Text(
                    text = blockReason,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(start = 80.dp)
                )
            }
        }
    }

    if (showLangPicker) {
        LanguagePickerDialog(
            selected = state.targetLang,
            onPick = {
                onTargetLang(it)
                showLangPicker = false
            },
            onDismiss = { showLangPicker = false }
        )
    }

    if (showSrcPicker) {
        SrcLangPickerDialog(
            accent = AgentOrange,
            selected = state.srcLangCode,
            onPick = {
                onSrcLang(it)
                showSrcPicker = false
            },
            onDismiss = { showSrcPicker = false }
        )
    }
}

@Composable
private fun ConfigRow(label: String, content: @Composable () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.width(80.dp)
        )
        Box(modifier = Modifier.weight(1f)) { content() }
    }
}

@Composable
private fun SegmentedSwitch(
    options: List<String>,
    selectedIndex: Int,
    enabled: Boolean,
    onSelected: (Int) -> Unit
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(10.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(2.dp)
    ) {
        options.forEachIndexed { idx, label ->
            val sel = idx == selectedIndex
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(
                        if (sel) MaterialTheme.colorScheme.surface else Color.Transparent
                    )
                    .clickable(enabled = enabled && !sel) { onSelected(idx) }
                    .padding(horizontal = 14.dp, vertical = 8.dp)
            ) {
                Text(
                    text = label,
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = if (sel) FontWeight.SemiBold else FontWeight.Normal,
                    color = if (sel) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun LanguagePickerDialog(
    selected: Language,
    onPick: (Language) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("目标语种") },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                languages.forEach { lang ->
                    val sel = lang.code == selected.code
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onPick(lang) }
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
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("关闭") }
        }
    )
}

@Composable
private fun ResultPanel(
    state: SpeechMTUiState,
    onCloseHistory: () -> Unit,
    onTabSelect: (HistoryDetailTab) -> Unit = {},
    onSummarize: (String) -> Unit = {},
    onStopSummarize: () -> Unit = {},
    onDownload: (SpeechMTRecord) -> Unit = {},
    onDelete: (String) -> Unit = {},
    modifier: Modifier = Modifier
) {
    val record = state.history.firstOrNull { it.id == state.viewingRecordId }
    val showingHistory = record != null
    val segments = if (record != null) record.segments else state.currentSegments
    val title = if (record != null) "查看历史 · ${record.timeLabel}" else "实时结果"
    val player = rememberSegmentPlayer()

    ElevatedCard(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            SectionTitle(
                icon = Icons.AutoMirrored.Filled.TextSnippet,
                text = title,
                accent = AgentOrange,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                trailing = {
                    if (state.isBusy && !showingHistory) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = AgentOrange
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = state.statusText.ifBlank { "进行中" },
                            style = MaterialTheme.typography.labelSmall,
                            color = AgentOrange,
                            maxLines = 1
                        )
                    } else if (state.statusText.isNotEmpty() && !showingHistory) {
                        Text(
                            text = state.statusText,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1
                        )
                    }
                    if (showingHistory) {
                        TextButton(onClick = onCloseHistory) {
                            Text("返回实时")
                        }
                    }
                }
            )
            HorizontalDivider()
            // ── 上传模式：整条音频钉在最顶侧（先看到/试听原文件），下面是切分后的逐段结果 ──
            if (!showingHistory && state.uploadPreview != null) {
                UploadPreviewCard(
                    preview = state.uploadPreview,
                    segCount = segments.size,
                    busy = state.isBusy,
                    player = player
                )
                HorizontalDivider()
            }
            // ── 历史详情：双标签页（查看历史 / 纪要总结）+ 操作行（整理纪要 / 下载 / 删除）──
            if (showingHistory && record != null) {
                TabRow(
                    selectedTabIndex = if (state.detailTab == HistoryDetailTab.SUMMARY) 1 else 0,
                    containerColor = MaterialTheme.colorScheme.surface
                ) {
                    HistoryDetailTab.entries.forEach { tab ->
                        Tab(
                            selected = state.detailTab == tab,
                            onClick = { onTabSelect(tab) },
                            text = { Text(tab.label, style = MaterialTheme.typography.labelMedium) }
                        )
                    }
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 14.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (state.summarizing) {
                        OutlinedButton(onClick = onStopSummarize, modifier = Modifier.weight(1f)) {
                            Icon(Icons.Default.Stop, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("停止生成")
                        }
                    } else {
                        Button(
                            onClick = { onSummarize(record.id) },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = AgentOrange)
                        ) {
                            Icon(Icons.Default.AutoAwesome, null, Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text(if (record.summary != null) "重新整理纪要" else "整理纪要")
                        }
                    }
                    IconButton(onClick = { onDownload(record) }, modifier = Modifier.size(40.dp)) {
                        Icon(Icons.Default.Download, contentDescription = "下载",
                            tint = MaterialTheme.colorScheme.outline, modifier = Modifier.size(20.dp))
                    }
                    IconButton(onClick = { onDelete(record.id) }, modifier = Modifier.size(40.dp)) {
                        Icon(Icons.Default.Delete, contentDescription = "删除",
                            tint = MaterialTheme.colorScheme.outline, modifier = Modifier.size(20.dp))
                    }
                }
            }
            when {
                // 纪要总结标签页
                showingHistory && state.detailTab == HistoryDetailTab.SUMMARY ->
                    SummaryPane(record, state, Modifier.fillMaxSize())
                // 模型加载中：结果区显示大字 + 缓冲动画（点击开始后模型未就绪时）
                state.isLoadingModel -> ModelLoadingBox(Modifier.fillMaxSize())
                segments.isEmpty() -> {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text(
                            text = if (state.isBusy) "等待识别中…" else "暂无内容",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.outline
                        )
                    }
                }
                else -> {
                    val listState = rememberLazyListState()
                    // 实时模式下：段数变化或最后一段文本更新时，自动滚动到底，始终展示最新结果
                    val last = segments.lastOrNull()
                    LaunchedEffect(
                        segments.size,
                        last?.id,
                        last?.transcript,
                        last?.translation,
                        showingHistory
                    ) {
                        if (!showingHistory && segments.isNotEmpty()) {
                            listState.animateScrollToItem(segments.lastIndex)
                        }
                    }
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        items(segments, key = { it.id }) { seg ->
                            SegmentCard(seg, player)
                        }
                    }
                }
            }
        }
    }
}

/** 纪要总结标签页：LLM 加载中 / 流式生成 / 已保存纪要 / 空占位 四态。 */
@Composable
private fun SummaryPane(
    record: SpeechMTRecord?,
    state: SpeechMTUiState,
    modifier: Modifier = Modifier
) {
    when {
        state.loadingLlm -> ModelLoadingBox(modifier, label = "纪要模型加载中…")
        state.summarizing -> {
            // 流式生成：文本随 token 增长，自动滚动到底
            val scroll = rememberScrollState()
            LaunchedEffect(state.summaryStream.length, state.todoStream.length, state.outlineStream.length) {
                if (scroll.maxValue > 0) scroll.scrollTo(scroll.maxValue)
            }
            Column(modifier) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(14.dp), strokeWidth = 2.dp, color = AgentOrange)
                    Text("纪要生成中…", style = MaterialTheme.typography.labelSmall, color = AgentOrange)
                }
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 14.dp, vertical = 6.dp)
                        .verticalScroll(scroll)
                ) {
                    Column {
                        // 大纲：可见的"思考过程"（先于正文出现）
                        if (state.outlineStream.isNotBlank()) {
                            Text(
                                text = "话题：" + state.outlineStream,
                                style = MaterialTheme.typography.bodySmall,
                                color = AgentOrange
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                        }
                        Text(
                            text = state.summaryStream.ifBlank { "…" },
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        // 待办单独一栏流式展示（有内容才出现）
                        if (state.todoStream.isNotBlank()) {
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(
                                text = "【待办】",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = AgentOrange
                            )
                            Text(
                                text = state.todoStream,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                        // 抽取式关键句：生成中作即时预览，完成后转为折叠附件
                        if (state.excerpt.isNotEmpty()) ExcerptCard(state.excerpt)
                    }
                }
            }
        }
        record?.summary != null -> {
            val scroll = rememberScrollState()
            Column(
                modifier = modifier
                    .padding(horizontal = 0.dp, vertical = 10.dp)
                    .verticalScroll(scroll)
            ) {
                Text(
                    text = record.summary,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.padding(horizontal = 14.dp)
                )
                // 关键句摘录：折叠附件，供核对总结的事实来源
                if (excerptLines(record.excerpt).isNotEmpty()) ExcerptCard(excerptLines(record.excerpt))
            }
        }
        else -> {
            Box(modifier = modifier, contentAlignment = Alignment.Center) {
                Text(
                    text = "点击「整理纪要」，AI 将把本条录音整理为结构化纪要",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }
    }
}

/** 模型加载占位：大字"模型加载中"+ 缓冲转圈，加载完由调用方切换状态后消失。 */
@Composable
internal fun ModelLoadingBox(modifier: Modifier = Modifier, label: String = "模型加载中…") {
    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(48.dp),
                strokeWidth = 4.dp,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = label,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "首次加载需要一点时间，请稍候",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/** 上传整条音频预览卡：文件名 / 时长 / 大小 + 整条播放（原 Uri 直播）+ 切分进度。 */
@Composable
private fun UploadPreviewCard(
    preview: UploadFilePreview,
    segCount: Int,
    busy: Boolean,
    player: SegmentPlayer,
    modifier: Modifier = Modifier
) {
    val id = "upload_whole_${preview.uriString}"
    val isCurrent = player.playingId == id
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 14.dp, vertical = 10.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(AgentOrange.copy(alpha = 0.09f))
            .border(width = 1.dp, color = AgentOrange.copy(alpha = 0.35f), shape = RoundedCornerShape(14.dp))
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(AgentOrange.copy(alpha = 0.18f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.UploadFile,
                contentDescription = null,
                tint = AgentOrange,
                modifier = Modifier.size(22.dp)
            )
        }
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = preview.fileName,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(modifier = Modifier.height(3.dp))
            Text(
                text = buildString {
                    append("整条音频 · ").append(preview.durationLabel)
                    if (preview.sizeLabel.isNotEmpty()) append(" · ").append(preview.sizeLabel)
                    append(if (busy) " · 切分识别中…" else " · 已切分 $segCount 段")
                },
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        // 整条播放：MediaPlayer 直接读原始 Uri（不用等解码/落盘）
        Box(
            modifier = Modifier
                .size(38.dp)
                .clip(RoundedCornerShape(50))
                .background(if (isCurrent) AgentOrange else MaterialTheme.colorScheme.surface)
                .border(width = 1.dp, color = AgentOrange.copy(alpha = 0.4f), shape = RoundedCornerShape(50))
                .clickable { player.toggleUri(id, Uri.parse(preview.uriString)) },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (isCurrent && player.isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                contentDescription = "播放整条音频",
                tint = if (isCurrent) Color.White else AgentOrange,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@Composable
private fun SegmentCard(seg: Segment, player: SegmentPlayer) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outlineVariant,
                shape = RoundedCornerShape(12.dp)
            )
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        // 原始音频：段建立即有（录音落盘），可播放；三者异步--音频先于转写/翻译
        if (seg.audioPath != null) {
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                val isCurrent = player.playingId == seg.id
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(RoundedCornerShape(50))
                        .background(if (isCurrent) AgentOrange else MaterialTheme.colorScheme.surface)
                        .clickable { player.toggle(seg.id, seg.audioPath) },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (isCurrent && player.isPlaying) Icons.Default.Pause
                        else Icons.Default.PlayArrow,
                        contentDescription = "播放原音",
                        tint = if (isCurrent) Color.White else AgentOrange,
                        modifier = Modifier.size(18.dp)
                    )
                }
                TagBadge(text = "原音", color = AgentOrange)
                Text(
                    text = fmtDuration(seg.audioSamples),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
        }
        // 转写（ASR partial -> 定稿，异步于音频）
        Row(verticalAlignment = Alignment.CenterVertically) {
            TagBadge(text = "转写", color = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = seg.transcript.ifBlank { "…" },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.weight(1f)
            )
        }
        // 翻译（MT 完成后出现，最后到）
        if (seg.translation != null) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                TagBadge(text = "翻译", color = Color(0xFFF97316))
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = seg.translation,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

/** 音频播放器：封装 MediaPlayer，记录当前播放项。切换/离开页面时释放。
 *  [toggle] 播已落盘的 WAV 路径；[toggleUri] 直接播原始 content Uri（整条上传音频预览用，
 *  不必等解码与落盘完成）。 */
private class SegmentPlayer(private val context: Context) {
    private var mp: MediaPlayer? = null
    var playingId by mutableStateOf<String?>(null)
        private set
    var isPlaying by mutableStateOf(false)
        private set

    fun toggle(id: String, path: String) {
        if (playingId == id) togglePause()
        else playNew(id) { m -> m.setDataSource(path) }
    }

    fun toggleUri(id: String, uri: Uri) {
        if (playingId == id) togglePause()
        else playNew(id) { m -> m.setDataSource(context, uri) }
    }

    private fun togglePause() {
        val m = mp
        if (m != null && m.isPlaying) { m.pause(); isPlaying = false }
        else { m?.start(); isPlaying = true }
    }

    private fun playNew(id: String, setSource: (MediaPlayer) -> Unit) {
        runCatching { mp?.release() }
        val m = MediaPlayer()
        try {
            setSource(m)
            m.setOnPreparedListener { it.start(); isPlaying = true }
            m.setOnCompletionListener { isPlaying = false; playingId = null }
            m.setOnErrorListener { _, _, _ -> isPlaying = false; playingId = null; true }
            m.prepareAsync()
            mp = m
            playingId = id
            isPlaying = false
        } catch (e: Exception) {
            Log.e("SegmentPlayer", "play failed", e)
            runCatching { m.release() }
            playingId = null
            isPlaying = false
        }
    }

    fun release() {
        runCatching { mp?.release() }
        mp = null
        playingId = null
        isPlaying = false
    }
}

@Composable
private fun rememberSegmentPlayer(): SegmentPlayer {
    val ctx = LocalContext.current.applicationContext
    val p = remember { SegmentPlayer(ctx) }
    DisposableEffect(Unit) { onDispose { p.release() } }
    return p
}

private fun fmtDuration(samples: Int): String {
    val totalSec = samples / 16000
    return "%d:%02d".format(totalSec / 60, totalSec % 60)
}

@Composable
private fun TagBadge(text: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(color.copy(alpha = 0.15f))
            .padding(horizontal = 6.dp, vertical = 2.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.SemiBold
        )
    }
}

/** 历史记录下载内容类型。 */
private enum class DownloadType(val label: String, val desc: String) {
    AUDIO("完整音频", "WAV 文件，各段拼接"),
    SOURCE_TEXT("源语言文本", "仅转写原文"),
    TARGET_TEXT("目标语言文本", "仅译文"),
    BILINGUAL("原译对照", "转写 + 翻译"),
    SUMMARY("纪要总结", "AI 整理的会议纪要")
}

@Composable
private fun DownloadTypeDialog(
    hasAudio: Boolean,
    hasSummary: Boolean = false,
    onPick: (DownloadType) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("选择下载内容") },
        text = {
            Column {
                DownloadType.entries.forEach { t ->
                    val disabled = (t == DownloadType.AUDIO && !hasAudio) ||
                        (t == DownloadType.SUMMARY && !hasSummary)
                    val hint = when {
                        t == DownloadType.AUDIO && !hasAudio -> "无音频"
                        t == DownloadType.SUMMARY && !hasSummary -> "未生成"
                        else -> null
                    }
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable(enabled = !disabled) { onPick(t) }
                            .padding(vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = t.label,
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.Medium,
                                color = if (!disabled) MaterialTheme.colorScheme.onSurface
                                else MaterialTheme.colorScheme.outline
                            )
                            Text(
                                text = t.desc,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        if (hint != null) {
                            Text(
                                text = hint,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.outline
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        }
    )
}

