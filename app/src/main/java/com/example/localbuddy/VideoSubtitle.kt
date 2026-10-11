package com.example.localbuddy

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import android.view.ViewGroup
import android.widget.MediaController
import android.widget.VideoView
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items as gridItems
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.sqrt

// ──────────────────────────────────────────────────────────────────
// 数据模型
// ──────────────────────────────────────────────────────────────────

/** 一条字幕：原文转写 + 译文 + 时间区间（毫秒）。 */
data class SubtitleCue(
    val startMs: Long,
    val endMs: Long,
    val transcript: String,
    val translation: String? = null
) {
    val text: String get() = translation ?: transcript
    val timeLabel: String
        get() = "${fmt(startMs)} - ${fmt(endMs)}"
    private fun fmt(ms: Long): String {
        val s = ms / 1000
        return "%02d:%02d".format(s / 60, s % 60)
    }
}

/**
 * 一个视频字幕任务：源视频本地副本（SAF Uri 权限不持久，导入时拷贝到应用目录）+
 * 字幕段 + 烧录结果视频 + AI 总结，整体持久化。相册页每张卡片 = 一个任务。
 */
data class VideoTask(
    val id: String,
    val name: String,
    val createdAt: Long,
    val srcPath: String,
    val outPath: String? = null,       // 烧录好字幕的结果视频，生成完成后非空
    val thumbPath: String? = null,
    val durationMs: Long = 0L,
    val translateOn: Boolean = true,
    val srcLangCode: String = LANG_AUTO,     // 源语种：auto=自动识别，选定后强制该语种转写
    val targetLangCode: String = "zh",
    val cues: List<SubtitleCue> = emptyList(),
    val summary: String? = null,
    /** 总结依据的关键句摘录（换行分隔，抽取式、覆盖全文头中尾），完成后作可折叠附件供溯源。 */
    val excerpt: String? = null
)

/** 页面内导航：相册（入口）/ 任务（生成与查看）。 */
enum class VsPage { Library, Task }

enum class SubtitleStage { Idle, Importing, Extracting, Recognizing, Translating, Burning, Done, Error }

data class VideoSubtitleUiState(
    val page: VsPage = VsPage.Library,
    val library: List<VideoTask> = emptyList(),
    val currentTaskId: String? = null,   // 任务页正在查看的任务
    val activeTaskId: String? = null,    // 正在生成字幕的任务（与查看页可不同）
    val stage: SubtitleStage = SubtitleStage.Idle,
    val progress: Float = 0f,            // 烧录进度 0..1
    val statusText: String = "",
    val error: String? = null,
    // ── 视频总结（LLM 流式）──
    val summaryStream: String = "",
    val todoStream: String = "",              // 流式待办增量（与要点分开展示）
    val outlineStream: String = "",           // 流式大纲（可见的"思考过程"）
    val excerpt: List<String> = emptyList(),  // 抽取式关键句（即时预览 → 完成后折叠附件）
    val summarizing: Boolean = false,
    val summarizeTaskId: String? = null,  // 正在总结的任务（流式文本只在该任务页显示）
    val loadingLlm: Boolean = false
) {
    val currentTask: VideoTask? get() = library.firstOrNull { it.id == currentTaskId }
    val isBusy: Boolean
        get() = stage in setOf(
            SubtitleStage.Importing, SubtitleStage.Extracting,
            SubtitleStage.Recognizing, SubtitleStage.Translating, SubtitleStage.Burning
        )
    /** 某个任务卡片是否生成中。 */
    fun isTaskBusy(id: String): Boolean = activeTaskId == id && isBusy
}

// ──────────────────────────────────────────────────────────────────
// 持久化（JSON 索引 + filesDir/video_subtitle 下的媒体文件）
// ──────────────────────────────────────────────────────────────────

private class VideoTaskStore(private val context: Context) {
    val dir: File get() = File(context.filesDir, "video_subtitle").apply { mkdirs() }
    private val file: File get() = File(context.filesDir, "video_subtitle_tasks.json")

    suspend fun load(): List<VideoTask> = withContext(Dispatchers.IO) {
        if (!file.exists()) return@withContext emptyList()
        runCatching {
            val arr = JSONArray(file.readText())
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                val cueArr = o.getJSONArray("cues")
                val cues = (0 until cueArr.length()).map { j ->
                    val c = cueArr.getJSONObject(j)
                    SubtitleCue(
                        startMs = c.getLong("startMs"),
                        endMs = c.getLong("endMs"),
                        transcript = c.getString("transcript"),
                        translation = c.optString("translation").takeIf { it.isNotEmpty() }
                    )
                }
                VideoTask(
                    id = o.getString("id"),
                    name = o.getString("name"),
                    createdAt = o.getLong("createdAt"),
                    srcPath = o.getString("srcPath"),
                    outPath = o.optString("outPath").takeIf { it.isNotEmpty() },
                    thumbPath = o.optString("thumbPath").takeIf { it.isNotEmpty() },
                    durationMs = o.optLong("durationMs", 0L),
                    translateOn = o.optBoolean("translateOn", true),
                    srcLangCode = o.optString("srcLangCode", LANG_AUTO),
                    targetLangCode = o.optString("targetLangCode", "zh"),
                    cues = cues,
                    summary = o.optString("summary").takeIf { it.isNotEmpty() },
                    excerpt = o.optString("excerpt").takeIf { it.isNotEmpty() }
                )
            }
        }.getOrElse { emptyList() }
    }

    suspend fun save(list: List<VideoTask>) = withContext(Dispatchers.IO) {
        val arr = JSONArray()
        list.forEach { t ->
            val cues = JSONArray()
            t.cues.forEach { c ->
                cues.put(
                    JSONObject()
                        .put("startMs", c.startMs)
                        .put("endMs", c.endMs)
                        .put("transcript", c.transcript)
                        .put("translation", c.translation ?: "")
                )
            }
            arr.put(
                JSONObject()
                    .put("id", t.id)
                    .put("name", t.name)
                    .put("createdAt", t.createdAt)
                    .put("srcPath", t.srcPath)
                    .put("outPath", t.outPath ?: "")
                    .put("thumbPath", t.thumbPath ?: "")
                    .put("durationMs", t.durationMs)
                    .put("translateOn", t.translateOn)
                    .put("srcLangCode", t.srcLangCode)
                    .put("targetLangCode", t.targetLangCode)
                    .put("cues", cues)
                    .put("summary", t.summary ?: "")
                    .put("excerpt", t.excerpt ?: "")
            )
        }
        file.writeText(arr.toString())
    }
}

// ──────────────────────────────────────────────────────────────────
// ViewModel
// ──────────────────────────────────────────────────────────────────

class VideoSubtitleViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = SpeechMTEngine.get(app)
    private val llm = LlmEngine.get(app)
    private val store = VideoTaskStore(app)
    private val _state = MutableStateFlow(VideoSubtitleUiState())
    val state: StateFlow<VideoSubtitleUiState> = _state.asStateFlow()

    private var job: Job? = null            // 字幕生成
    private var summarizeJob: Job? = null   // LLM 总结

    init {
        viewModelScope.launch { engine.warmUp() }
        viewModelScope.launch {
            val list = store.load()
            _state.update { it.copy(library = list) }
        }
    }

    /** 相册页「上传」选完视频：拷贝到应用目录（SAF 权限不持久）→ 缩略图/时长 → 建任务 → 进任务页。 */
    fun importVideo(uri: Uri, name: String?) {
        if (_state.value.isBusy) return
        _state.update {
            it.copy(stage = SubtitleStage.Importing, progress = 0f,
                statusText = "导入视频中…", error = null)
        }
        viewModelScope.launch {
            runCatching {
                val ctx = getApplication<Application>()
                val id = "v${System.currentTimeMillis()}"
                val src = File(store.dir, "src_$id.mp4")
                withContext(Dispatchers.IO) {
                    ctx.contentResolver.openInputStream(uri)?.use { input ->
                        src.outputStream().use { input.copyTo(it) }
                    } ?: throw IllegalStateException("无法读取所选视频")
                }
                // 缩略图 + 时长（失败不阻断导入，占位图兜底）
                var durationMs = 0L
                var thumbPath: String? = null
                withContext(Dispatchers.IO) {
                    val retriever = MediaMetadataRetriever()
                    try {
                        retriever.setDataSource(src.absolutePath)
                        durationMs = runCatching {
                            retriever.extractMetadata(
                                MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
                        }.getOrNull() ?: 0L
                        val frame = runCatching {
                            retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                        }.getOrNull()
                        if (frame != null) {
                            val thumb = File(store.dir, "thumb_$id.jpg")
                            runCatching {
                                thumb.outputStream().use { frame.compress(Bitmap.CompressFormat.JPEG, 85, it) }
                                thumbPath = thumb.absolutePath
                            }
                            frame.recycle()
                        }
                    } finally {
                        runCatching { retriever.release() }
                    }
                }
                val task = VideoTask(
                    id = id,
                    name = name ?: "视频",
                    createdAt = System.currentTimeMillis(),
                    srcPath = src.absolutePath,
                    thumbPath = thumbPath,
                    durationMs = durationMs
                )
                _state.update { s ->
                    s.copy(
                        library = listOf(task) + s.library,
                        currentTaskId = id, page = VsPage.Task,
                        stage = SubtitleStage.Idle, progress = 0f,
                        statusText = "", error = null
                    )
                }
                store.save(_state.value.library)
            }.onFailure {
                Log.e(TAG, "import failed", it)
                _state.update { s ->
                    s.copy(stage = SubtitleStage.Idle, statusText = "",
                        error = "导入失败：${it.message ?: "未知错误"}")
                }
            }
        }
    }

    fun openTask(id: String) {
        val busyOnIt = _state.value.isTaskBusy(id)
        _state.update {
            it.copy(
                page = VsPage.Task, currentTaskId = id,
                summaryStream = "", todoStream = "", outlineStream = "", excerpt = emptyList(),
                // 非生成中的任务：清掉上一任务遗留的瞬时状态
                stage = if (busyOnIt) it.stage else SubtitleStage.Idle,
                progress = if (busyOnIt) it.progress else 0f,
                statusText = if (busyOnIt) it.statusText else "",
                error = if (busyOnIt) it.error else null
            )
        }
    }

    fun backToLibrary() {
        _state.update { it.copy(page = VsPage.Library) }
    }

    /** 删除任务：取消进行中的生成/总结，清掉源视频/结果视频/缩略图与索引记录。 */
    fun deleteTask(id: String) {
        val task = _state.value.library.firstOrNull { it.id == id } ?: return
        if (_state.value.activeTaskId == id && _state.value.isBusy) {
            job?.cancel(); job = null
            _state.update {
                it.copy(activeTaskId = null, stage = SubtitleStage.Idle,
                    progress = 0f, statusText = "")
            }
        }
        if (_state.value.summarizing && _state.value.currentTaskId == id) {
            summarizeJob?.cancel(); summarizeJob = null
            _state.update { it.copy(summarizing = false) }
        }
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                listOfNotNull(task.srcPath, task.outPath, task.thumbPath).forEach { p ->
                    runCatching { File(p).delete() }
                }
            }
            val viewing = _state.value.currentTaskId == id
            _state.update { s ->
                s.copy(
                    library = s.library.filterNot { it.id == id },
                    currentTaskId = if (viewing) null else s.currentTaskId,
                    page = if (viewing) VsPage.Library else s.page
                )
            }
            store.save(_state.value.library)
        }
    }

    /** 切换目标语种：与源语种冲突（同语种）时自动关闭翻译。 */
    fun setTargetLang(lang: Language) = updateCurrentTask {
        val blocked = translateBlockReason(it.srcLangCode, lang.code) != null
        it.copy(targetLangCode = lang.code, translateOn = if (blocked) false else it.translateOn)
    }

    /** 开启翻译：源语种不支持翻译 / 与目标语种相同时拒绝开启（开关在 UI 上也是禁用的）。 */
    fun setTranslateOn(on: Boolean) = updateCurrentTask {
        if (on && translateBlockReason(it.srcLangCode, it.targetLangCode) != null) it
        else it.copy(translateOn = on)
    }

    /** 切换语音语种：该语种不支持翻译 / 与目标语种相同时自动关闭翻译。 */
    fun setSrcLang(code: String) = updateCurrentTask {
        val blocked = translateBlockReason(code, it.targetLangCode) != null
        it.copy(srcLangCode = code, translateOn = if (blocked) false else it.translateOn)
    }

    private fun updateCurrentTask(transform: (VideoTask) -> VideoTask) {
        val id = _state.value.currentTaskId ?: return
        _state.update { s -> s.copy(library = s.library.map { if (it.id == id) transform(it) else it }) }
        viewModelScope.launch { store.save(_state.value.library) }
    }

    private fun updateTaskInLibrary(id: String, transform: (VideoTask) -> VideoTask) {
        _state.update { s -> s.copy(library = s.library.map { if (it.id == id) transform(it) else it }) }
    }

    /**
     * 生成字幕并硬烧进视频（对当前任务）：
     * 提取音频 → VAD 句尾分段（带时间戳）→ 逐段 ASR + 翻译（实时刷到「分段字幕」标签）→ 组装字幕
     * → MediaCodec 硬烧（结果视频落在「字幕视频」标签）→ 自动触发 LLM 视频总结。
     */
    fun generate() {
        val task = _state.value.currentTask ?: return
        if (_state.value.isBusy) return
        val translateOn = task.translateOn
        val tgtLangCode = task.targetLangCode
        val srcLangCode = task.srcLangCode
        val taskId = task.id
        val srcUri = Uri.fromFile(File(task.srcPath))
        val outFile = File(store.dir, "out_$taskId.mp4")

        _state.update {
            it.copy(activeTaskId = taskId, stage = SubtitleStage.Extracting, progress = 0f,
                statusText = if (!engine.isLoaded) "模型加载中…" else "提取音频中…",
                error = null, summaryStream = "", todoStream = "", outlineStream = "", excerpt = emptyList())
        }
        // 重新生成：清空旧结果（结果视频文件在烧录成功后由新文件覆盖）
        updateTaskInLibrary(taskId) { it.copy(cues = emptyList(), outPath = null, summary = null) }
        if (outFile.exists()) runCatching { outFile.delete() }

        job = viewModelScope.launch {
            // 本地字幕快照：多协程（ASR/翻译）并发读写，最终以 publish 同步进任务记录。
            // 声明在 runCatching 外——失败时 onFailure 也能拿到已产出的段落盘。
            val cues = ArrayList<SubtitleCue>()
            fun publishCues() {
                val snapshot = synchronized(cues) { cues.map { it.copy() } }
                updateTaskInLibrary(taskId) { it.copy(cues = snapshot) }
            }
            try {
                val ctx = getApplication<Application>()
                if (!engine.isLoaded) {
                    engine.warmUp()
                    _state.update { it.copy(statusText = "提取音频中…") }
                }

                // 1-2) 两遍流式解码：第一遍扫整条帧 RMS 的 10 分位作噪声底定 VAD 门限，
                //    第二遍增量 VAD 分段、段一出来立即 ASR + 翻译，PCM 用完即释放。
                //    长视频一次性解码整条 PCM 会 OOM（30 分钟 ~115MB）；流式峰值仅当前段(≤10s)。
                //    VAD 用能量门限而非 Silero（对真实视频人声概率偏低不稳定）；音乐/噪声误触发
                //    由下游 ASR 空段丢弃兜底。
                _state.update { it.copy(stage = SubtitleStage.Recognizing, statusText = "分析音量…") }
                var maxPeak = 0f
                val rmsList = ArrayList<Float>(4096)
                val noisePending = ArrayList<Float>(256)
                withContext(Dispatchers.IO) {
                    AudioDecoder.decodeStreamPcm16kMono(ctx, srcUri) { chunk ->
                        for (s in chunk) {
                            val a = if (s < 0f) -s else s; if (a > maxPeak) maxPeak = a
                            noisePending.add(s)
                            if (noisePending.size >= 256) {
                                var ss = 0.0
                                for (v in noisePending) ss += (v * v).toDouble()
                                rmsList.add(sqrt(ss / 256).toFloat())
                                noisePending.subList(0, 256).clear()
                            }
                        }
                    }
                }
                val noiseFloor = if (rmsList.isEmpty()) 0f else {
                    rmsList.sort()
                    rmsList[(rmsList.size * 0.10f).toInt().coerceIn(0, rmsList.size - 1)]
                }
                val vadThreshold = max(noiseFloor * 3f, 0.012f)
                Log.i(TAG, "audio peak=$maxPeak noise=$noiseFloor vad threshold=$vadThreshold")

                val translationJobs = ArrayList<Job>()
                var segCount = 0
                val asrRawSamples = ArrayList<String>()   // 诊断：收集 ASR 空段的模型 raw 输出
                // 分段以语言表达完整性为主：句尾停顿 1.2s、单段上限 10s（< ASR 12s 截断上限）
                val vad = StreamingVadSegmenter(
                    EnergyVadDetector(256, vadThreshold),
                    pauseMs = 1200, maxSegMs = 10_000
                )
                _state.update { it.copy(statusText = "识别中…") }

                /** 单段处理：ASR（partial 实时上屏）→ 翻译（异步 job）。返回 false = ASR 空段被丢弃。 */
                suspend fun handleSegment(startMs: Long, endMs: Long, pcm: FloatArray,
                                          scope: kotlinx.coroutines.CoroutineScope): Boolean {
                    segCount++
                    val idx: Int
                    synchronized(cues) {
                        cues.add(SubtitleCue(startMs, endMs, ""))
                        idx = cues.lastIndex
                    }
                    publishCues()
                    // ASR：源语种按配置（auto=自动检测；选定语种=强制转写，短语音不依赖语种识别）
                    val normalized = AudioDecoder.normalizePeak(pcm)
                    val result = if (srcLangCode == LANG_AUTO) {
                        engine.asrDetect(normalized) { partial ->
                            if (partial.isNotBlank()) {
                                synchronized(cues) { cues[idx] = cues[idx].copy(transcript = partial) }
                                publishCues()
                            }
                        }
                    } else {
                        SpeechMTEngine.AsrResult(
                            engine.asr(normalized, srcLangCode) { partial ->
                                if (partial.isNotBlank()) {
                                    synchronized(cues) { cues[idx] = cues[idx].copy(transcript = partial) }
                                    publishCues()
                                }
                            }, srcLangCode
                        )
                    }
                    Log.i(TAG, "seg $segCount: start=$startMs asr=[${result.text}] lang=${result.language}")
                    if (result.text.isBlank()) {
                        synchronized(asrRawSamples) {
                            if (asrRawSamples.size < 5) asrRawSamples.add(result.raw.ifBlank { "(空)" })
                        }
                        synchronized(cues) { cues.removeAt(idx) }
                        publishCues()
                        return false
                    }
                    synchronized(cues) { cues[idx] = cues[idx].copy(transcript = result.text) }
                    publishCues()
                    _state.update { it.copy(statusText = "识别中…(段 $segCount)") }
                    // 源语种不支持翻译 / 与目标语种相同 → 跳过该段翻译
                    // （自动检测模式下按检测到的语种判定；否则 MT 会把原文原样吐出当译文）
                    if (translateOn && needsTranslation(result.language ?: "zh", tgtLangCode)) {
                        // 翻译：源语种由 MT 自检（多语种视频按段内实际语种各自翻译）
                        val srcLangCode = result.language ?: "zh"
                        _state.update {
                            it.copy(stage = SubtitleStage.Translating, statusText = "翻译中…(段 $segCount)")
                        }
                        translationJobs += scope.launch {
                            val mt = engine.translate(result.text, srcLangCode, tgtLangCode)
                            Log.i(TAG, "mt seg idx=$idx -> [$mt]")
                            if (mt.isNotBlank()) {
                                synchronized(cues) { cues[idx] = cues[idx].copy(translation = mt) }
                                publishCues()
                            }
                        }
                    }
                    return true
                }

                withContext(Dispatchers.IO) {
                    val scope = this
                    AudioDecoder.decodeStreamPcm16kMono(ctx, srcUri) { chunk ->
                        for (seg in vad.feed(chunk)) {
                            handleSegment(seg.startMs, seg.endMs, seg.pcm, scope)
                        }
                    }
                    // VAD 收尾：发出末尾未结句的语音段
                    for (seg in vad.flush()) {
                        handleSegment(seg.startMs, seg.endMs, seg.pcm, scope)
                    }
                }
                Log.i(TAG, "total segments=$segCount cues=${cues.size}")
                translationJobs.forEach { it.join() }
                publishCues()
                store.save(_state.value.library)

                if (cues.isEmpty()) {
                    val msg = if (segCount == 0) {
                        "未检测到人声：VAD 切出 0 段，整条音频被判静音。" +
                            "峰值=${"%.4f".format(maxPeak)}（正常说话通常 > 0.05），VAD 门限=${"%.4f".format(vad.currentThreshold)}。" +
                            "可能音量过低、音频轨异常或无人声。"
                    } else {
                        val raws = synchronized(asrRawSamples) {
                            asrRawSamples.joinToString(" | ") { "[${it.replace("\n", " ")}]" }
                        }
                        "识别无结果：VAD 切出 $segCount 段，但 ASR 对每段都返回空。峰值=${"%.4f".format(maxPeak)}。" +
                            "模型输出样本：$raws"
                    }
                    _state.update { it.copy(stage = SubtitleStage.Error, statusText = "", error = msg) }
                    return@launch
                }

                // 3) 硬烧字幕（译文优先，无译文则原文）
                //    burnSubtitles 内部自建 GL 线程并阻塞等待，必须切到 IO 线程，避免阻塞主线程 ANR
                val finalCues = synchronized(cues) {
                    cues.map { SubtitleCue(it.startMs, it.endMs, it.transcript, it.translation) }
                }
                _state.update {
                    it.copy(stage = SubtitleStage.Burning, progress = 0f, statusText = "烧录字幕中…")
                }
                withContext(Dispatchers.IO) {
                    // 进度节流：每 ~2% 才更新一次状态文本，避免逐帧刷状态引发大量 UI 重组拖慢
                    var lastReported = -1f
                    SubtitleEncoder.burnSubtitles(ctx, srcUri,
                        finalCues.map { SubtitleEncoder.SubtitleCue(it.startMs, it.endMs, it.text) },
                        outFile) { p ->
                        if (p - lastReported >= 0.02f || p >= 1f) {
                            lastReported = p
                            _state.update {
                                it.copy(progress = p, statusText = "烧录字幕中…${(p * 100).toInt()}%")
                            }
                        }
                    }
                }
                updateTaskInLibrary(taskId) { it.copy(cues = finalCues, outPath = outFile.absolutePath) }
                store.save(_state.value.library)
                _state.update {
                    it.copy(stage = SubtitleStage.Done, progress = 1f, statusText = "字幕生成完成")
                }
                // 4) 自动生成视频总结（失败不阻断主流程，标签页内可手动重试）
                summarize(taskId)
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                Log.e(TAG, "generate failed", t)
                // 失败也保留已识别的字幕段（下次打开任务仍能看到部分结果）
                val snapshot = synchronized(cues) { cues.map { c -> c.copy() } }
                if (snapshot.isNotEmpty()) {
                    updateTaskInLibrary(taskId) { it.copy(cues = snapshot) }
                }
                _state.update { s ->
                    s.copy(stage = SubtitleStage.Error, statusText = "",
                        error = "处理失败：${t.message ?: "未知错误"}")
                }
                store.save(_state.value.library)
            }
        }
    }

    /**
     * 视频总结：对目标语言文字（有译文用译文，否则转写）调用 LLM 流式生成，
     * 让用户不看视频即可了解内容。结果写回任务并持久化。
     */
    fun summarize(taskId: String) {
        if (_state.value.summarizing) return
        val task = _state.value.library.firstOrNull { it.id == taskId } ?: return
        val content = task.cues.mapNotNull { c ->
            (if (task.translateOn) (c.translation ?: c.transcript) else c.transcript)
                .takeUnless { it.isBlank() }
        }.joinToString("\n")
        if (content.isBlank()) {
            _state.update { it.copy(statusText = "暂无字幕文本，无法生成总结") }
            return
        }
        _state.update {
            it.copy(summarizing = true, summaryStream = "", todoStream = "", outlineStream = "",
                excerpt = emptyList(), summarizeTaskId = taskId, loadingLlm = !llm.isLoaded)
        }
        summarizeJob?.cancel()
        summarizeJob = viewModelScope.launch {
            try {
                if (!llm.isLoaded) llm.warmUp()
                _state.update { it.copy(loadingLlm = false) }
                // 抽取预热（毫秒级可见）→ 大纲 → 正文流式；超长文内部自动降级逐段模式
                val summary = llm.summarizeSmart(
                    content, system = SUMMARY_SYSTEM, contentDesc = "视频字幕", maxNew = 768,
                    onPreview = { s -> _state.update { it.copy(excerpt = s) } },
                    onOutline = { d -> _state.update { it.copy(outlineStream = it.outlineStream + d) } },
                    onPartial = { d -> _state.update { it.copy(summaryStream = it.summaryStream + d) } },
                    onTodo = { d -> _state.update { it.copy(todoStream = it.todoStream + d) } },
                    onProgress = { i, n ->
                        _state.update { it.copy(statusText = "总结中…已梳理 $i/$n 段") }
                    }
                )
                if (summary.isNotBlank()) {
                    updateTaskInLibrary(taskId) {
                        it.copy(summary = summary, excerpt = _state.value.excerpt.joinToString("\n").takeIf { s -> s.isNotBlank() })
                    }
                    store.save(_state.value.library)
                    _state.update { it.copy(statusText = "视频总结已生成") }
                } else {
                    _state.update {
                        it.copy(statusText = "总结生成失败（模型已就绪但输出为空，可能字幕过短或模型未正常生成，详见 logcat Tag=LlmEngine）")
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "summarize failed", e)
                _state.update { it.copy(statusText = "总结生成失败：${e.message ?: "未知"}") }
            } finally {
                _state.update { it.copy(summarizing = false, summarizeTaskId = null, loadingLlm = false) }
            }
        }
    }

    /** 停止总结（保留已流式生成的部分：写回任务持久化）。 */
    fun stopSummarize() {
        val j = summarizeJob ?: return
        summarizeJob = null
        val partial = combinePointsTodo(_state.value.summaryStream, _state.value.todoStream)
        val taskId = _state.value.currentTaskId
        j.cancel()
        viewModelScope.launch {
            if (partial.isNotBlank() && taskId != null) {
                updateTaskInLibrary(taskId) {
                    it.copy(summary = partial,
                        excerpt = _state.value.excerpt.joinToString("\n").takeIf { s -> s.isNotBlank() })
                }
                store.save(_state.value.library)
            }
        }
    }

    override fun onCleared() {
        job?.cancel()
        summarizeJob?.cancel()
    }

    companion object {
        private const val TAG = "VideoSubtitleVM"
        /** 视频总结系统提示：强总结性、要点列表化、与字幕（目标语言）同语言输出。 */
        private const val SUMMARY_SYSTEM =
            "你是一个专业的视频内容总结助手。请对用户提供的视频字幕文本做高度提炼的总结，要求：\n" +
                "1. 开头用一两句话概括这段视频的主题和整体内容，让读者不看视频也能了解大意；\n" +
                "2. 多个要点用列表形式（- 开头）逐条列出，每条是一个完整、信息密度高的句子，可合并同话题的零散内容；\n" +
                "3. 有结论、决定或关键信息时单独列出（如「结论」「要点」小节）；\n" +
                "4. 忽略口语中的寒暄、重复和无意义内容，不要逐句复述字幕；\n" +
                "5. 使用与字幕文本相同的语言输出，直接给出总结正文，不要任何额外解释或前缀；\n" +
                // 抽取路径靠【大纲】/【总结】/【待办】小节切分流式输出，系统提示必须让位给用户消息里的格式
                SUMMARY_FORMAT_RULE
    }
}

// ──────────────────────────────────────────────────────────────────
// UI — 页面骨架（相册 / 任务 两页，VM 状态切换）
// ──────────────────────────────────────────────────────────────────

@Composable
fun VideoSubtitleScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val vm: VideoSubtitleViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance(
            (context.applicationContext as Application)
        )
    )
    val state by vm.state.collectAsState()

    // 任务页内系统返回 → 回相册页（再按一次才退出智能体）
    BackHandler(enabled = state.page == VsPage.Task) { vm.backToLibrary() }

    when (state.page) {
        VsPage.Library -> LibraryPage(
            state = state,
            onBack = onBack,
            onImport = vm::importVideo,
            onOpen = vm::openTask,
            onDelete = vm::deleteTask
        )
        VsPage.Task -> TaskPage(state = state, vm = vm, onBack = onBack)
    }
}

// ──────────────────────────────────────────────────────────────────
// UI — 相册页（入口）
// ──────────────────────────────────────────────────────────────────

@Composable
private fun LibraryPage(
    state: VideoSubtitleUiState,
    onBack: () -> Unit,
    onImport: (Uri, String?) -> Unit,
    onOpen: (String) -> Unit,
    onDelete: (String) -> Unit
) {
    val context = LocalContext.current
    // 上传：系统文件选择器（ACTION_OPEN_DOCUMENT）选视频 → 导入（拷贝+缩略图）→ 自动进任务页。
    // 不用 GetContent（相册）：部分厂商相册对超长/超大视频过滤或索引不全（>1h 视频不显示），
    // SAF 文档 UI 直列存储上的全部文件，无此问题。
    val pickVideoLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            val name = runCatching {
                context.contentResolver.query(uri, null, null, null, null)?.use { c ->
                    val idx = c.getColumnIndex("_display_name")
                    if (idx >= 0 && c.moveToFirst()) c.getString(idx) else null
                }
            }.getOrNull()
            onImport(uri, name)
        }
    }

    var pendingDelete by remember { mutableStateOf<VideoTask?>(null) }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            LibraryHeader(
                busy = state.isBusy,   // 导入中或后台有任务在生成时都禁用上传
                onBack = onBack,
                onUpload = { pickVideoLauncher.launch(arrayOf("video/*", "application/mp4")) }
            )
        }
    ) { innerPadding ->
        Column(modifier = Modifier.fillMaxSize().padding(innerPadding)) {
            if (state.error != null) {
                Text(
                    text = state.error!!,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                )
            }
            if (state.library.isEmpty()) {
                // 空态引导
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.VideoLibrary,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.outline,
                            modifier = Modifier.size(56.dp)
                        )
                        Text(
                            text = "还没有视频\n点击右上角「上传」开始生成字幕",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.outline,
                            lineHeight = 22.sp,
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                    }
                }
            } else {
                // 相册式两列网格：缩略图 + 名称 + 状态，点击进任务页，右上角删除
                LazyVerticalGrid(
                    columns = GridCells.Fixed(2),
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(
                        start = 16.dp, end = 16.dp, top = 12.dp, bottom = 24.dp
                    ),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    gridItems(state.library, key = { it.id }) { task ->
                        VideoCard(
                            task = task,
                            busy = state.isTaskBusy(task.id),
                            onClick = { onOpen(task.id) },
                            onDelete = { pendingDelete = task }
                        )
                    }
                }
            }
        }
    }

    // 删除确认
    pendingDelete?.let { task ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("删除视频") },
            text = { Text("将删除「${task.name}」及其字幕、结果视频和总结，无法恢复。") },
            confirmButton = {
                TextButton(onClick = { onDelete(task.id); pendingDelete = null }) {
                    Text("删除", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("取消") }
            }
        )
    }
}

// ── 相册页头部：渐变 + 返回 + 标题 + 「上传」操作块 ──
@Composable
private fun LibraryHeader(busy: Boolean, onBack: () -> Unit, onUpload: () -> Unit) {
    val gradient = Brush.linearGradient(listOf(AgentCyan, lerp(AgentCyan, Color.White, 0.22f)))
    Column(modifier = Modifier.background(gradient)) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
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
                        text = "本地视频字幕智能体",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = "上传视频 · 生成字幕 · AI 总结",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.White.copy(alpha = 0.88f)
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                // 「上传」操作块：上图标下名称（导入中禁用防重复）
                Column(
                    modifier = Modifier
                        .clip(RoundedCornerShape(12.dp))
                        .clickable(enabled = !busy, onClick = onUpload)
                        .padding(horizontal = 4.dp, vertical = 2.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(Color.White.copy(alpha = if (busy) 0.12f else 0.22f)),
                        contentAlignment = Alignment.Center
                    ) {
                        if (busy) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                strokeWidth = 2.dp, color = Color.White
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.UploadFile,
                                contentDescription = "上传",
                                tint = Color.White,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                    Text(
                        text = "上传",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White.copy(alpha = if (busy) 0.5f else 0.95f)
                    )
                }
            }
        }
        if (busy) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().height(3.dp),
                color = Color.White,
                trackColor = Color.White.copy(alpha = 0.25f)
            )
        }
    }
}

// ── 相册卡片：缩略图（时长角标 + 状态角标 + 删除）+ 名称 + 日期/语种 ──
@Composable
private fun VideoCard(
    task: VideoTask,
    busy: Boolean,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    val doneColor = Color(0xFF22C55E)
    ElevatedCard(
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(104.dp)
                    .background(Color(0xFF111827))
            ) {
                // 缩略图（导入失败时占位图标）
                val bmp = remember(task.thumbPath) {
                    task.thumbPath?.let { runCatching { BitmapFactory.decodeFile(it) }.getOrNull() }
                }
                if (bmp != null) {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                } else {
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Icon(
                            imageVector = Icons.Default.Videocam,
                            contentDescription = null,
                            tint = Color.White.copy(alpha = 0.4f),
                            modifier = Modifier.size(32.dp)
                        )
                    }
                }
                // 时长角标（右下）
                if (task.durationMs > 0) {
                    Text(
                        text = fmtVideoDuration(task.durationMs),
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White,
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .padding(6.dp)
                            .clip(RoundedCornerShape(4.dp))
                            .background(Color.Black.copy(alpha = 0.6f))
                            .padding(horizontal = 5.dp, vertical = 1.dp)
                    )
                }
                // 状态角标（左上）：生成中 / 已完成 / 待生成
                val (statusLabel, statusColor) = when {
                    busy -> "生成中" to AgentCyan
                    task.outPath != null -> "已完成" to doneColor
                    else -> "待生成" to Color(0xFF9CA3AF)
                }
                Row(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(6.dp)
                        .clip(RoundedCornerShape(6.dp))
                        .background(Color.Black.copy(alpha = 0.6f))
                        .padding(horizontal = 6.dp, vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    if (busy) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(9.dp), strokeWidth = 1.5.dp, color = statusColor
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .size(6.dp)
                                .clip(RoundedCornerShape(3.dp))
                                .background(statusColor)
                        )
                    }
                    Text(
                        text = statusLabel,
                        style = MaterialTheme.typography.labelSmall,
                        color = statusColor,
                        fontWeight = FontWeight.SemiBold
                    )
                }
                // 删除按钮（右上）
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(4.dp)
                        .size(26.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color.Black.copy(alpha = 0.55f))
                        .clickable(onClick = onDelete),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "删除",
                        tint = Color.White,
                        modifier = Modifier.size(15.dp)
                    )
                }
            }
            // 名称 + 日期
            Column(
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                Text(
                    text = task.name,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = fmtDate(task.createdAt),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// UI — 任务页（新任务流程：原视频 → 选语种 → 生成 → 三标签页）
// ──────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TaskPage(
    state: VideoSubtitleUiState,
    vm: VideoSubtitleViewModel,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val task = state.currentTask ?: return

    // 任务页内的瞬时状态（stage 等属于全局 state，按 activeTaskId 区分是否作用于本任务）
    val busy = state.isTaskBusy(task.id)
    val stage = when {
        busy -> state.stage
        state.activeTaskId == task.id && state.stage == SubtitleStage.Error -> SubtitleStage.Error
        task.outPath != null -> SubtitleStage.Done
        else -> SubtitleStage.Idle
    }

    // 下载结果视频：先弹系统保存对话框，回调里把输出文件复制到用户选定 URI
    var pendingDownload by remember { mutableStateOf<File?>(null) }
    val downloadLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("video/mp4")
    ) { uri: Uri? ->
        val file = pendingDownload
        pendingDownload = null
        if (uri != null && file != null) {
            runCatching {
                context.contentResolver.openOutputStream(uri)?.use { os ->
                    file.inputStream().use { it.copyTo(os) }
                }
            }.onFailure { Log.e("VideoSubtitle", "download failed", it) }
        }
    }

    val tabTitles = listOf("分段字幕", "字幕视频", "视频总结")

    // 整页纵向滚动：生成后原视频/配置可上划出屏，三标签页区域获得接近全屏的高度；
    // 标签页本体用 HorizontalPager 支持左右滑动切换。
    val scrollState = rememberScrollState()
    val pagerState = rememberPagerState(initialPage = 0, pageCount = { tabTitles.size })
    val scope = rememberCoroutineScope()
    val screenH = LocalConfiguration.current.screenHeightDp.dp
    /** 生成字幕：切到「分段字幕」标签并把标签页区域滚到屏顶。 */
    fun focusTabs() {
        scope.launch {
            pagerState.animateScrollToPage(0)
            scrollState.animateScrollTo(scrollState.maxValue)
        }
    }

    Scaffold(
        topBar = {
            AgentHeader(
                accent = AgentCyan,
                icon = Icons.Default.Subtitles,
                title = "本地视频字幕智能体",
                subtitle = task.name,
                onBack = vm::backToLibrary
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(scrollState)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // ── 1) 原视频（可播放）──
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
            ) {
                Column {
                    SectionTitle(
                        icon = Icons.Default.PlayCircle,
                        text = "原视频",
                        accent = AgentCyan,
                        modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)
                    )
                    key(task.srcPath) {
                        VideoPlayer(
                            path = task.srcPath,
                            autoPlay = false,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 12.dp)
                                .padding(bottom = 12.dp)
                                .height(200.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.Black)
                        )
                    }
                }
            }

            // ── 2) 目标语种 + 生成 ──
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    GenerateConfig(
                        task = task,
                        enabled = !busy,
                        onTargetLang = vm::setTargetLang,
                        onToggleTranslate = vm::setTranslateOn,
                        onSrcLang = vm::setSrcLang
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        when {
                            busy -> {
                                Button(onClick = {}, enabled = false, colors = agentButtonColors(AgentCyan)) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(state.statusText.ifBlank { "处理中…" })
                                }
                            }
                            stage == SubtitleStage.Done -> {
                                Button(onClick = { focusTabs(); vm.generate() },
                                    colors = agentButtonColors(AgentCyan)) {
                                    Icon(Icons.Default.Refresh, contentDescription = null,
                                        modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("重新生成")
                                }
                            }
                            else -> {
                                Button(onClick = { focusTabs(); vm.generate() },
                                    colors = agentButtonColors(AgentCyan)) {
                                    Icon(Icons.Default.AutoAwesome, contentDescription = null,
                                        modifier = Modifier.size(18.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("生成字幕")
                                }
                            }
                        }
                        if (stage == SubtitleStage.Burning) {
                            LinearProgressIndicator(
                                progress = { state.progress },
                                modifier = Modifier.weight(1f).height(8.dp)
                                    .clip(RoundedCornerShape(4.dp))
                            )
                        } else if (state.statusText.isNotEmpty() && !busy) {
                            Text(
                                text = state.statusText,
                                style = MaterialTheme.typography.bodySmall,
                                color = if (stage == SubtitleStage.Error) MaterialTheme.colorScheme.error
                                else AgentCyan,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                    if (stage == SubtitleStage.Error && state.error != null) {
                        Text(
                            text = state.error!!,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            }

            // ── 3) 三标签页：分段字幕 / 字幕视频 / 视频总结 ──
            // 可滚动 Column 中 weight 无意义，显式给到接近全屏的高度：滚到底后标签页
            // 区域铺满可视区（内部列表/滚动自行嵌套滚动）。
            ElevatedCard(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(maxOf(screenH - 230.dp, 400.dp)),
                shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.fillMaxSize()) {
                    TabRow(selectedTabIndex = pagerState.currentPage) {
                        tabTitles.forEachIndexed { i, title ->
                            Tab(
                                selected = pagerState.currentPage == i,
                                onClick = { scope.launch { pagerState.animateScrollToPage(i) } },
                                text = {
                                    Text(
                                        text = title,
                                        style = MaterialTheme.typography.labelLarge,
                                        fontWeight = if (pagerState.currentPage == i) FontWeight.SemiBold
                                        else FontWeight.Normal
                                    )
                                }
                            )
                        }
                    }
                    HorizontalPager(
                        state = pagerState,
                        modifier = Modifier.fillMaxSize()
                    ) { page ->
                        when (page) {
                            0 -> SegmentsTab(
                                cues = task.cues,
                                busy = busy,
                                modifier = Modifier.fillMaxSize()
                                    .padding(horizontal = 14.dp, vertical = 10.dp)
                            )
                            1 -> ResultVideoTab(
                                task = task,
                                burning = busy && state.stage == SubtitleStage.Burning,
                                progress = state.progress,
                                onDownload = { file ->
                                    pendingDownload = file
                                    downloadLauncher.launch(file.name)
                                },
                                modifier = Modifier.fillMaxSize()
                            )
                            else -> SummaryTab(
                                task = task,
                                stream = if (state.summarizeTaskId == task.id) state.summaryStream else "",
                                todoStream = if (state.summarizeTaskId == task.id) state.todoStream else "",
                                outlineStream = if (state.summarizeTaskId == task.id) state.outlineStream else "",
                                excerpt = if (state.summarizeTaskId == task.id) state.excerpt
                                    else excerptLines(task.excerpt),
                                summarizing = state.summarizeTaskId == task.id,
                                loadingLlm = state.loadingLlm,
                                onGenerate = { vm.summarize(task.id) },
                                onStop = vm::stopSummarize,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }
                }
            }
        }
    }
}

// ── 目标语种配置（源语种自动检测，无需选择）──
@Composable
private fun GenerateConfig(
    task: VideoTask,
    enabled: Boolean,
    onTargetLang: (Language) -> Unit,
    onToggleTranslate: (Boolean) -> Unit,
    onSrcLang: (String) -> Unit
) {
    var tgtPicker by remember { mutableStateOf(false) }
    var srcPicker by remember { mutableStateOf(false) }
    val targetLang = languages.firstOrNull { it.code == task.targetLangCode } ?: languages[0]

    // 语音语种：自动识别 / 指定（短语音语种识别更稳）
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            text = "语音语种",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        AssistChip(
            onClick = { if (enabled) srcPicker = true },
            enabled = enabled,
            label = { Text(srcLangLabel(task.srcLangCode)) },
            leadingIcon = {
                Icon(
                    if (task.srcLangCode == LANG_AUTO) Icons.Default.AutoAwesome
                    else Icons.Default.RecordVoiceOver,
                    contentDescription = null, modifier = Modifier.size(16.dp)
                )
            }
        )
    }

    // 源语种不支持翻译 / 与目标语种相同时禁用翻译开关（配置层面就拦住，运行时另有兜底）
    val blockReason = translateBlockReason(task.srcLangCode, task.targetLangCode)
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            text = "翻译",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface
        )
        Switch(
            checked = task.translateOn && blockReason == null,
            onCheckedChange = { if (enabled && blockReason == null) onToggleTranslate(it) },
            enabled = enabled && blockReason == null
        )
        if (task.translateOn || blockReason != null) {
            AssistChip(
                onClick = { if (enabled) tgtPicker = true },
                enabled = enabled,
                label = { Text("${targetLang.flag} ${targetLang.name}") },
                leadingIcon = {
                    Icon(Icons.Default.Translate, contentDescription = null,
                        modifier = Modifier.size(16.dp))
                }
            )
        }
    }
    if (blockReason != null) {
        Text(
            text = blockReason,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.error
        )
    }

    if (tgtPicker) {
        LanguagePickerDialog(
            selected = targetLang,
            onPick = { onTargetLang(it); tgtPicker = false },
            onDismiss = { tgtPicker = false }
        )
    }

    if (srcPicker) {
        SrcLangPickerDialog(
            accent = AgentCyan,
            selected = task.srcLangCode,
            onPick = { onSrcLang(it); srcPicker = false },
            onDismiss = { srcPicker = false }
        )
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
        title = { Text("选择目标语种") },
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
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } }
    )
}

// ── 标签 1：实时分段字幕（样式同录音翻译：时间 + 转写 + 翻译，自动滚到底）──
@Composable
private fun SegmentsTab(
    cues: List<SubtitleCue>,
    busy: Boolean,
    modifier: Modifier = Modifier
) {
    if (cues.isEmpty()) {
        Box(modifier = modifier, contentAlignment = Alignment.Center) {
            Text(
                text = if (busy) "等待识别…" else "暂无字幕，点击「生成字幕」开始",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.outline
            )
        }
        return
    }
    val listState = rememberLazyListState()
    val last = cues.lastOrNull()
    LaunchedEffect(cues.size, last?.transcript, last?.translation) {
        if (cues.isNotEmpty()) listState.animateScrollToItem(cues.lastIndex)
    }
    LazyColumn(
        state = listState,
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        items(cues) { cue -> SubtitleCard(cue) }
    }
}

@Composable
private fun SubtitleCard(cue: SubtitleCue) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(12.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TagBadge(text = cue.timeLabel, color = AgentCyan)
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            TagBadge(text = "转写", color = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = cue.transcript.ifBlank { "…" },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.weight(1f)
            )
        }
        if (cue.translation != null) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                TagBadge(text = "译文", color = Color(0xFFF97316))
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = cue.translation,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

// ── 标签 2：烧录好字幕的结果视频 ──
@Composable
private fun ResultVideoTab(
    task: VideoTask,
    burning: Boolean,
    progress: Float,
    onDownload: (File) -> Unit,
    modifier: Modifier = Modifier
) {
    val outFile = task.outPath?.let { File(it) }
    if (outFile != null && outFile.exists()) {
        Column(
            modifier = modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            key(outFile.absolutePath) {
                VideoPlayer(
                    path = outFile.absolutePath,
                    autoPlay = true,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(220.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color.Black)
                )
            }
            Button(
                onClick = { onDownload(outFile) },
                modifier = Modifier.fillMaxWidth(),
                colors = agentButtonColors(AgentCyan)
            ) {
                Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("下载视频")
            }
        }
    } else {
        Box(modifier = modifier, contentAlignment = Alignment.Center) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (burning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(32.dp), strokeWidth = 3.dp, color = AgentCyan)
                    Text(
                        text = "烧录字幕中…${(progress * 100).toInt()}%",
                        style = MaterialTheme.typography.bodySmall,
                        color = AgentCyan
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.Movie,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.outline,
                        modifier = Modifier.size(40.dp)
                    )
                    Text(
                        text = "字幕生成完成后，\n烧录好字幕的视频将在这里展示",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.outline,
                        lineHeight = 20.sp,
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center
                    )
                }
            }
        }
    }
}

// ── 标签 3：视频总结（LLM 对目标语言文字流式生成）──
@Composable
private fun SummaryTab(
    task: VideoTask,
    stream: String,
    todoStream: String,
    outlineStream: String,
    excerpt: List<String>,
    summarizing: Boolean,
    loadingLlm: Boolean,
    onGenerate: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // 操作行：生成中可停止；已有总结可重新生成
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (summarizing) {
                Button(onClick = onStop, colors = agentButtonColors(AgentCyan)) {
                    Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(if (loadingLlm) "模型加载中…" else "停止生成")
                }
                LinearProgressIndicator(
                    modifier = Modifier.weight(1f).height(8.dp).clip(RoundedCornerShape(4.dp)),
                    color = AgentCyan
                )
            } else {
                Button(
                    onClick = onGenerate,
                    enabled = task.cues.isNotEmpty(),
                    colors = agentButtonColors(AgentCyan)
                ) {
                    Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(if (task.summary != null) "重新总结" else "生成总结")
                }
                Spacer(modifier = Modifier.weight(1f))
                if (task.cues.isEmpty()) {
                    Text(
                        text = "生成字幕后自动总结",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }
        }
        HorizontalDivider()

        val content = when {
            summarizing -> stream
            else -> task.summary
        }
        // 生成中：待办单独展示（持久化后的 summary 是完整文本，无需拆分）
        val todoShown = if (summarizing) todoStream.trim() else ""
        if (content.isNullOrBlank() && todoShown.isBlank()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    text = "AI 视频总结：不看视频，一眼了解视频内容",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        } else {
            // 总结按 Markdown 渲染（LLM 要点列表输出）
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                // 大纲：可见的"思考过程"（生成中实时增长）
                if (outlineStream.isNotBlank()) {
                    Text(
                        text = "话题：" + outlineStream,
                        style = MaterialTheme.typography.bodySmall,
                        color = AgentCyan
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
                if (!content.isNullOrBlank()) MarkdownText(markdown = content)
                if (todoShown.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "【待办】",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = AgentCyan
                    )
                    MarkdownText(markdown = todoShown)
                }
                if (excerpt.isNotEmpty()) ExcerptCard(excerpt)
                if (summarizing) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "▍", style = MaterialTheme.typography.bodyMedium,
                        color = AgentCyan
                    )
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// 共用小组件
// ──────────────────────────────────────────────────────────────────

/** 关键句摘录（换行分隔的持久化串）→ 列表。 */
internal fun excerptLines(raw: String?): List<String> =
    raw?.split('\n')?.filter { it.isNotBlank() } ?: emptyList()

/**
 * 关键句摘录卡：**默认折叠**的溯源附件。
 *
 * 生成中它同时是"即时预览"（抽取零延迟，先于 LLM 输出出现），生成完成后保留下来，
 * 便于核对总结里的每个数字/结论在原文里到底怎么说的。
 */
@Composable
internal fun ExcerptCard(sentences: List<String>, modifier: Modifier = Modifier) {
    var expanded by remember { mutableStateOf(false) }
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 14.dp, vertical = 8.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f))
            .padding(10.dp)
    ) {
        Row(
            modifier = Modifier.clickable { expanded = !expanded },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                imageVector = if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.outline,
                modifier = Modifier.size(18.dp)
            )
            Text(
                text = "关键句摘录（${sentences.size}）",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (expanded) {
            Spacer(modifier = Modifier.height(6.dp))
            sentences.forEachIndexed { i, s ->
                Text(
                    text = "${i + 1}. $s",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(vertical = 2.dp)
                )
            }
        }
    }
}

/** 视频播放器：VideoView + 系统 MediaController（点画面出进度条）。autoPlay 时循环播放。 */
@Composable
private fun VideoPlayer(path: String, autoPlay: Boolean, modifier: Modifier = Modifier) {
    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            VideoView(ctx).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT)
                val controller = MediaController(ctx)
                setMediaController(controller)
                controller.setAnchorView(this)
                setVideoPath(path)
                setOnPreparedListener { mp ->
                    if (autoPlay) {
                        mp.isLooping = true
                        start()
                    } else {
                        seekTo(1)   // 显示首帧，避免黑屏
                    }
                }
            }
        },
        onRelease = { it.stopPlayback() }
    )
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

private fun fmtVideoDuration(ms: Long): String {
    val s = ms / 1000
    return "%d:%02d".format(s / 60, s % 60)
}

private fun fmtDate(ts: Long): String =
    SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(ts))
