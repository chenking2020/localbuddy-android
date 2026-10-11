package com.example.localbuddy

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
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
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

// ──────────────────────────────────────────────────────────────────
// 共享状态：Service 写，UI 读
// ──────────────────────────────────────────────────────────────────

/** 实时听译的共享状态。Service 推理过程中更新；UI 观察展示结果列表。 */
object ListenSubtitleController {
    data class State(
        val isCapturing: Boolean = false,
        val isLoadingModel: Boolean = false,   // 模型加载中（点击开始后模型未就绪时，结果区显示加载态）
        val statusText: String = "",
        val segments: List<Segment> = emptyList()
    )

    private val _state = MutableStateFlow(State())
    val state: StateFlow<State> = _state.asStateFlow()

    fun startSession() { _state.value = State() }
    fun setCapturing(c: Boolean, status: String) =
        _state.update { it.copy(isCapturing = c, statusText = status) }
    fun setLoadingModel(b: Boolean) =
        _state.update { it.copy(isLoadingModel = b) }
    fun addSegmentPlaceholder(id: String) =
        _state.update { it.copy(segments = it.segments + Segment(id = id, transcript = "")) }
    fun updateTranscript(id: String, transcript: String) =
        _state.update { s -> s.copy(segments = s.segments.map { if (it.id == id) it.copy(transcript = transcript) else it }) }
    fun updateTranslation(id: String, translation: String) =
        _state.update { s -> s.copy(segments = s.segments.map { if (it.id == id) it.copy(translation = translation) else it }) }
    fun removeSegment(id: String) =
        _state.update { s -> s.copy(segments = s.segments.filterNot { it.id == id }) }
}

// ──────────────────────────────────────────────────────────────────
// 前台 Service：持有 MediaProjection 捕获回放音频 + 推理 + 悬浮字幕
// ──────────────────────────────────────────────────────────────────

/**
 * 实时视频听音智能体的前台 Service。捕获本机回放音频 -> 能量 VAD 分段 -> ASR(asr_text 多语种) ->
 * 翻译 -> 悬浮窗显示译文(+原文)。用户切到视频 App 后继续运行（前台 Service + 系统悬浮窗）。
 *
 * 生命周期：UI 拿到 MediaProjection 同意结果后 [startForegroundService] 带 ACTION_START +
 * resultCode/data 启动本服务；停止发 ACTION_STOP（或点通知"停止"）。服务与 UI 生命周期解耦：
 * UI 关闭不停止服务，仅用户主动停止 / 通知停止。
 *
 * 仅 Android 10+（API 29）支持回放捕获；UI 已按 SDK 拦截，本服务假定运行在 29+。
 */
@RequiresApi(29)
class ListenSubtitleService : Service() {

    private val engine by lazy { SpeechMTEngine.get(this) }
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val mainHandler = Handler(Looper.getMainLooper())
    private var windowManager: WindowManager? = null
    private var overlayContainer: LinearLayout? = null
    private var overlayTranslation: TextView? = null
    private var overlayOriginal: TextView? = null
    private var overlayParams: WindowManager.LayoutParams? = null
    // 拖动状态：记录按下时的悬浮窗位置 + 触摸坐标，MOVE 时按增量更新位置
    private var dragStartX = 0
    private var dragStartY = 0
    private var dragStartTouchX = 0f
    private var dragStartTouchY = 0f
    private var recorder: PlaybackCaptureRecorder? = null
    private var pipelineJob: Job? = null
    @Volatile private var latestSegId: String? = null
    // 悬浮窗当前已显示的段索引。译文异步完成，到达时若该段不比已显示的旧才更新悬浮窗，
    // 避免旧译文（晚到）覆盖新译文。用索引而非 latestSegId：后者在新段开始即推进，
    // 会导致本段译文完成时被判为"非最新"而永远不显示（悬浮窗卡在"正在听音…"）。
    @Volatile private var overlaySegIndex: Int = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
                val data = IntentCompat.getParcelableExtra(intent, EXTRA_DATA, Intent::class.java)
                val translateOn = intent.getBooleanExtra(EXTRA_TRANSLATE, true)
                val targetLangCode = intent.getStringExtra(EXTRA_TARGET_LANG) ?: "en"
                val srcLangCode = intent.getStringExtra(EXTRA_SRC_LANG) ?: LANG_AUTO
                startForegroundWithNotification()
                if (data != null) startPipeline(resultCode, data, translateOn, targetLangCode, srcLangCode)
                else stopSelfSafely("未获得屏幕捕获授权")
            }
            ACTION_STOP -> stopSelfSafely(null)
        }
        return START_NOT_STICKY
    }

    /** 启动前台通知（Android 14+ 强制 mediaProjection 类型需在 getMediaProjection 前调用）。 */
    private fun startForegroundWithNotification() {
        ensureChannel()
        val stopIntent = Intent(this, ListenSubtitleService::class.java).setAction(ACTION_STOP)
        val stopPi = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val notif = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("实时听音翻译中")
            .setContentText("正在捕获本机音频并翻译，切到视频 App 即可看到悬浮字幕")
            .setSmallIcon(R.drawable.ic_listen)
            .setOngoing(true)
            .addAction(0, "停止", stopPi)
            .build()
        ServiceCompat.startForeground(
            this, NOTIF_ID, notif,
            if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION else 0
        )
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val mgr = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                mgr.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "实时听音翻译", NotificationManager.IMPORTANCE_LOW)
                        .apply { setShowBadge(false) }
                )
            }
        }
    }

    /** 启动捕获 + 推理管线。采集与推理解耦（同录音翻译）：录音协程塞 Channel，推理协程消费。 */
    private fun startPipeline(resultCode: Int, data: Intent, translateOn: Boolean,
                              targetLangCode: String, srcLangCode: String = LANG_AUTO) {
        val projection = (getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager)
            .getMediaProjection(resultCode, data)
        if (projection == null) { stopSelfSafely("MediaProjection 获取失败"); return }
        val rec = PlaybackCaptureRecorder(projection, pauseMs = 400, maxSegMs = 2000)
        recorder = rec
        ListenSubtitleController.startSession()
        overlaySegIndex = 0
        showOverlay()
        // 立即在悬浮窗显示状态（首段 ASR 要几秒，避免用户以为没反应；ASR partial 一到即替换）
        setOverlayText(if (translateOn) "正在听音…" else "正在听音·仅转写", null)

        val channel = Channel<FloatArray>(capacity = 8, onBufferOverflow = BufferOverflow.DROP_OLDEST)
        pipelineJob = scope.launch(Dispatchers.IO) {
            if (!engine.isLoaded) {
                ListenSubtitleController.setCapturing(true, "模型加载中…")
                ListenSubtitleController.setLoadingModel(true)
                engine.warmUp()
                ensureActive()
                ListenSubtitleController.setLoadingModel(false)
                ListenSubtitleController.setCapturing(true, "正在听音…")
            } else {
                ListenSubtitleController.setCapturing(true, "正在听音…")
            }
            // 采集协程：只读回放、塞 Channel
            launch {
                runCatching { rec.stream().collect { pcm -> channel.trySend(pcm) } }
                    .onFailure {
                        Log.e(TAG, "capture error", it)
                        ListenSubtitleController.setCapturing(true, "捕获失败：${it.message ?: "未知"}")
                    }
                channel.close()
            }
            // 推理协程：按序消费 PCM -> ASR；翻译异步并发
            var segCount = 0
            val translationJobs = mutableListOf<Job>()
            for (pcm in channel) {
                ensureActive()
                segCount++
                val segIdx = segCount
                val segId = UUID.randomUUID().toString()
                latestSegId = segId
                ListenSubtitleController.addSegmentPlaceholder(segId)
                // token 级流式：partial 实时刷新列表；悬浮窗仅显示译文--
                // 开翻译时不刷新原文（译文未到前保留上一条译文/状态），仅转写模式才实时显示原文
                // 源语种按配置：auto=自动检测；选定语种=强制转写（短语音不依赖语种识别）
                val normalizedPcm = AudioDecoder.normalizePeak(pcm)
                val asrResult = if (srcLangCode == LANG_AUTO) {
                    engine.asrDetect(normalizedPcm) { partial ->
                        if (partial.isNotBlank()) {
                            ListenSubtitleController.updateTranscript(segId, partial)
                            if (!translateOn && segId == latestSegId) setOverlayText(null, partial)
                        }
                    }
                } else {
                    SpeechMTEngine.AsrResult(
                        engine.asr(normalizedPcm, srcLangCode) { partial ->
                            if (partial.isNotBlank()) {
                                ListenSubtitleController.updateTranscript(segId, partial)
                                if (!translateOn && segId == latestSegId) setOverlayText(null, partial)
                            }
                        }, srcLangCode
                    )
                }
                val asr = asrResult.text
                Log.i(TAG, "asr seg=$segIdx text=[$asr] lang=${asrResult.language}")
                if (asr.isBlank()) {
                    ListenSubtitleController.removeSegment(segId)
                    if (segId == latestSegId) setOverlayText(null, null)
                    continue
                }
                ListenSubtitleController.updateTranscript(segId, asr)
                if (translateOn) {
                    translationJobs += launch {
                        // token 级流式：译文边生成边刷新列表与悬浮窗（悬浮字幕逐字浮现，实时性同转写）
                        val mt = engine.translate(asr, asrResult.language ?: "zh", targetLangCode) { partial ->
                            if (partial.isNotBlank()) {
                                ListenSubtitleController.updateTranslation(segId, partial)
                                // 仅当本段不比已显示的旧时才更新悬浮窗（避免旧译文覆盖新译文）
                                if (segIdx >= overlaySegIndex) {
                                    overlaySegIndex = segIdx
                                    setOverlayText(partial, null)   // 仅译文，不显示原语言
                                }
                            }
                        }
                        if (mt.isNotBlank()) {
                            ListenSubtitleController.updateTranslation(segId, mt)
                            // 译文异步到达：仅当本段不比已显示的旧时才更新悬浮窗（避免旧译文覆盖新译文）
                            if (segIdx >= overlaySegIndex) {
                                overlaySegIndex = segIdx
                                setOverlayText(mt, null)   // 仅译文，不显示原语言
                            }
                        }
                    }
                } else if (segId == latestSegId) {
                    setOverlayText(null, asr)   // 仅转写模式：悬浮窗显示原文转写
                }
                ListenSubtitleController.setCapturing(true, "听音中…(段 $segCount)")
            }
            translationJobs.forEach { it.join() }
            ListenSubtitleController.setCapturing(false, "")
        }
    }

    // ── 悬浮窗 ──
    private fun showOverlay() {
        mainHandler.post {
            if (overlayContainer != null) return@post
            val wm = getSystemService(WINDOW_SERVICE) as WindowManager
            windowManager = wm
            val density = resources.displayMetrics.density
            val container = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = GradientDrawable().apply {
                    cornerRadius = 20f * density
                    setColor(0xCC000000.toInt())
                }
                setPadding((14 * density).toInt(), (10 * density).toInt(),
                    (14 * density).toInt(), (10 * density).toInt())
            }
            val translation = TextView(this).apply {
                textSize = 16f; setTextColor(Color.WHITE)
                setShadowLayer(2f, 1f, 1f, 0xAA000000.toInt()); maxLines = 3
            }
            val original = TextView(this).apply {
                textSize = 12.5f; setTextColor(0xCCFFFFFF.toInt()); maxLines = 2
            }
            container.addView(translation); container.addView(original)
            overlayContainer = container; overlayTranslation = translation; overlayOriginal = original
            container.visibility = View.GONE   // 有文本时再显示
            // 可拖动：去掉 FLAG_NOT_TOUCHABLE（接收触摸以拖动），保留 FLAG_NOT_FOCUSABLE（不抢焦点）
            //   与 FLAG_LAYOUT_NO_LIMITS（可拖出屏幕边界）。代价：字幕区域内的点击不再穿透到下层
            //   视频，需要点字幕下方控件时先把它拖开。文字视图默认非 clickable，触摸会冒泡到容器。
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                y = (80 * density).toInt()
            }
            overlayParams = params
            container.setOnTouchListener { _, ev ->
                when (ev.action) {
                    MotionEvent.ACTION_DOWN -> {
                        dragStartX = params.x
                        dragStartY = params.y
                        dragStartTouchX = ev.rawX
                        dragStartTouchY = ev.rawY
                    }
                    MotionEvent.ACTION_MOVE -> {
                        // BOTTOM|CENTER_HORIZONTAL gravity：x 为相对中心的水平偏移(右为正)，
                        // y 为相对底边的垂直偏移(上为正)。故上拖(rawY 减小)->y 增。
                        params.x = dragStartX + (ev.rawX - dragStartTouchX).toInt()
                        params.y = dragStartY + (dragStartTouchY - ev.rawY).toInt()
                        runCatching { wm.updateViewLayout(container, params) }
                    }
                }
                true   // 消费触摸（用于拖动）
            }
            runCatching { wm.addView(container, params) }
                .onFailure { Log.e(TAG, "addView overlay failed", it) }
        }
    }

    /** 更新悬浮窗文本（译文大字 + 原文小字）；两者皆空则隐藏。 */
    private fun setOverlayText(translation: String?, original: String?) {
        mainHandler.post {
            val c = overlayContainer ?: return@post
            val tr = overlayTranslation; val og = overlayOriginal
            if (translation.isNullOrBlank()) { tr?.text = ""; tr?.visibility = View.GONE }
            else { tr?.text = translation; tr?.visibility = View.VISIBLE }
            if (original.isNullOrBlank()) { og?.text = ""; og?.visibility = View.GONE }
            else { og?.text = original; og?.visibility = View.VISIBLE }
            c.visibility = if (translation.isNullOrBlank() && original.isNullOrBlank()) View.GONE else View.VISIBLE
        }
    }

    private fun removeOverlay() {
        mainHandler.post {
            overlayContainer?.let { runCatching { windowManager?.removeView(it) } }
            overlayContainer = null; overlayTranslation = null; overlayOriginal = null
            overlayParams = null
        }
    }

    private fun stopSelfSafely(reason: String?) {
        recorder?.stop()
        pipelineJob?.cancel()
        removeOverlay()
        ListenSubtitleController.setLoadingModel(false)
        ListenSubtitleController.setCapturing(false, reason ?: "")
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        recorder?.stop()
        scope.cancel()
        removeOverlay()
        ListenSubtitleController.setLoadingModel(false)
        ListenSubtitleController.setCapturing(false, "")
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "com.example.localbuddy.LISTEN_START"
        const val ACTION_STOP = "com.example.localbuddy.LISTEN_STOP"
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_DATA = "data"
        const val EXTRA_TRANSLATE = "translate"
        const val EXTRA_TARGET_LANG = "targetLang"
        const val EXTRA_SRC_LANG = "srcLang"
        private const val NOTIF_ID = 4201
        private const val CHANNEL_ID = "listen_subtitle"
        private const val TAG = "ListenSubtitleService"

        fun start(context: Context, resultCode: Int, data: Intent, translate: Boolean,
                  targetLang: String, srcLang: String = LANG_AUTO) {
            val intent = Intent(context, ListenSubtitleService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_RESULT_CODE, resultCode)
                putExtra(EXTRA_DATA, data)
                putExtra(EXTRA_TRANSLATE, translate)
                putExtra(EXTRA_TARGET_LANG, targetLang)
                putExtra(EXTRA_SRC_LANG, srcLang)
            }
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, ListenSubtitleService::class.java).apply { action = ACTION_STOP }
            )
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// ViewModel（仅持有配置：翻译开关/目标语种 + 预热引擎）
// ──────────────────────────────────────────────────────────────────
class ListenSubtitleViewModel(app: android.app.Application) : AndroidViewModel(app) {
    private val engine = SpeechMTEngine.get(app)
    // 用 StateFlow 持有配置：UI collectAsState 后改动才会重组（plain var 不触发重组 -> 选项不更新）
    val translateOn = MutableStateFlow(true)
    val targetLang = MutableStateFlow(languages[0])   // 默认中文
    val srcLangCode = MutableStateFlow(LANG_AUTO)     // 源语种：auto=自动识别

    init { viewModelScope.launch { engine.warmUp() } }
}

// ──────────────────────────────────────────────────────────────────
// UI
// ──────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ListenSubtitleScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val vm: ListenSubtitleViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance(
            (context.applicationContext as android.app.Application)
        )
    )
    val state by ListenSubtitleController.state.collectAsState()
    val translateOn by vm.translateOn.collectAsState()
    val targetLang by vm.targetLang.collectAsState()
    val srcLangCode by vm.srcLangCode.collectAsState()

    // 投影授权结果 -> 启动服务（读 .value 拿当前配置，避免 launcher 闭包捕获陈旧值）
    val projectionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            ListenSubtitleService.start(
                context, result.resultCode, result.data!!,
                vm.translateOn.value, vm.targetLang.value.code, vm.srcLangCode.value
            )
        } else {
            ListenSubtitleController.setCapturing(false, "未授予屏幕捕获授权")
        }
    }
    // 悬浮窗权限返回：已授权则继续请求投影
    val overlaySettingsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (Settings.canDrawOverlays(context)) requestProjection(context, projectionLauncher)
        else ListenSubtitleController.setCapturing(false, "需要“显示在其他应用上层”权限")
    }
    // 通知权限返回：已授权则继续请求投影
    val notifPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) requestProjection(context, projectionLauncher)
        else ListenSubtitleController.setCapturing(false, "需要通知权限")
    }

    fun startFlow() {
        if (Build.VERSION.SDK_INT < 29) {
            ListenSubtitleController.setCapturing(false, "需要 Android 10 以上")
            return
        }
        if (!Settings.canDrawOverlays(context)) {
            ListenSubtitleController.setCapturing(true, "请授予悬浮窗权限后返回")
            overlaySettingsLauncher.launch(
                Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${context.packageName}"))
            )
            return
        }
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, android.Manifest.permission.POST_NOTIFICATIONS)
            != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            notifPermLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        requestProjection(context, projectionLauncher)
    }

    Scaffold(
        topBar = {
            AgentHeader(
                accent = AgentGreen,
                icon = Icons.Default.Subtitles,
                title = "实时视频听音智能体",
                subtitle = "悬浮 · 实时 · 全程离线",
                onBack = onBack
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
            ListenConfigPanel(
                translateOn = translateOn,
                targetLang = targetLang,
                srcLangCode = srcLangCode,
                isCapturing = state.isCapturing,
                statusText = state.statusText,
                onToggleTranslate = { vm.translateOn.value = it },
                onTargetLang = { vm.targetLang.value = it },
                onSrcLang = { vm.srcLangCode.value = it },
                onStart = { startFlow() },
                onStop = { ListenSubtitleService.stop(context) }
            )
            ListenResultPanel(
                segments = state.segments,
                isCapturing = state.isCapturing,
                isLoadingModel = state.isLoadingModel,
                modifier = Modifier.fillMaxWidth().weight(1f)
            )
        }
    }
}

private fun requestProjection(context: Context, launcher: ActivityResultLauncher<Intent>) {
    val mpm = context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    launcher.launch(mpm.createScreenCaptureIntent())
}

@Composable
private fun ListenConfigPanel(
    translateOn: Boolean,
    targetLang: Language,
    srcLangCode: String,
    isCapturing: Boolean,
    statusText: String,
    onToggleTranslate: (Boolean) -> Unit,
    onTargetLang: (Language) -> Unit,
    onSrcLang: (String) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit
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
                accent = AgentGreen
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("音频源", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.width(72.dp))
                Text("本机播放声音", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            // 语音语种：自动识别 / 指定（短语音语种识别更稳）
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("语音语种", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.width(72.dp))
                AssistChip(
                    onClick = { showSrcPicker = true },
                    label = { Text(srcLangLabel(srcLangCode)) },
                    leadingIcon = {
                        Icon(
                            if (srcLangCode == LANG_AUTO) Icons.Default.AutoAwesome
                            else Icons.Default.RecordVoiceOver,
                            contentDescription = null, modifier = Modifier.size(16.dp)
                        )
                    }
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Switch(checked = translateOn, onCheckedChange = onToggleTranslate)
                Text(if (translateOn) "翻译" else "仅转写", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                if (translateOn) {
                    Spacer(Modifier.width(8.dp))
                    AssistChip(
                        onClick = { showLangPicker = true },
                        label = { Text("${targetLang.flag} ${targetLang.name}") },
                        leadingIcon = { Icon(Icons.Default.Translate, contentDescription = null, modifier = Modifier.size(16.dp)) }
                    )
                }
            }
            Row(modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                if (isCapturing) {
                    Button(onClick = onStop, colors = agentButtonColors(AgentGreen)) {
                        Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp)); Text("停止听音")
                    }
                } else {
                    Button(onClick = onStart, colors = agentButtonColors(AgentGreen)) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp)); Text("开始听音")
                    }
                }
                if (statusText.isNotEmpty()) {
                    Text(statusText, style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary, modifier = Modifier.weight(1f))
                }
            }
            Text("提示：开始后切到视频 App，字幕悬浮在屏幕下方。仅 Android 10+；DRM/受保护内容无法捕获。",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
    if (showLangPicker) {
        ListenLangPickerDialog(
            selected = targetLang,
            onPick = { onTargetLang(it); showLangPicker = false },
            onDismiss = { showLangPicker = false }
        )
    }
    if (showSrcPicker) {
        SrcLangPickerDialog(
            accent = AgentGreen,
            selected = srcLangCode,
            onPick = { onSrcLang(it); showSrcPicker = false },
            onDismiss = { showSrcPicker = false }
        )
    }
}

@Composable
private fun ListenLangPickerDialog(selected: Language, onPick: (Language) -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("目标语种") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                languages.forEach { lang ->
                    val sel = lang.code == selected.code
                    Row(Modifier.fillMaxWidth().clickable { onPick(lang) }
                        .padding(horizontal = 8.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(lang.flag, fontSize = 18.sp); Spacer(Modifier.width(10.dp))
                        Text(lang.name, Modifier.weight(1f), fontWeight = if (sel) FontWeight.SemiBold else FontWeight.Normal)
                        if (sel) Icon(Icons.Default.Check, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } }
    )
}

@Composable
private fun ListenResultPanel(
    segments: List<Segment>,
    isCapturing: Boolean,
    isLoadingModel: Boolean,
    modifier: Modifier = Modifier
) {
    ElevatedCard(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
    ) {
        Column(Modifier.fillMaxSize()) {
            SectionTitle(
                icon = Icons.Default.Hearing,
                text = "实时结果",
                accent = AgentGreen,
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                trailing = {
                    if (isCapturing) {
                        CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp,
                            color = AgentGreen)
                        Spacer(Modifier.width(8.dp))
                        Text("进行中", style = MaterialTheme.typography.labelSmall, color = AgentGreen)
                    }
                }
            )
            HorizontalDivider()
            when {
                isLoadingModel -> ModelLoadingBox(Modifier.fillMaxSize())
                segments.isEmpty() -> {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text(if (isCapturing) "等待识别中…" else "点击“开始听音”后切到视频 App",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.outline)
                    }
                }
                else -> {
                    val listState = rememberLazyListState()
                    val last = segments.lastOrNull()
                    LaunchedEffect(segments.size, last?.id, last?.transcript, last?.translation) {
                        if (segments.isNotEmpty()) listState.animateScrollToItem(segments.lastIndex)
                    }
                    LazyColumn(state = listState, modifier = Modifier.fillMaxSize().padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        items(segments, key = { it.id }) { seg -> ListenSegmentCard(seg) }
                    }
                }
            }
        }
    }
}

@Composable
private fun ListenSegmentCard(seg: Segment) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
        .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
        .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(12.dp))
        .padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            ListenTagBadge("转写", MaterialTheme.colorScheme.primary); Spacer(Modifier.width(8.dp))
            Text(seg.transcript.ifBlank { "…" }, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.weight(1f))
        }
        if (seg.translation != null) {
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                ListenTagBadge("译文", ComposeColor(0xFFF97316)); Spacer(Modifier.width(8.dp))
                Text(seg.translation, style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun ListenTagBadge(text: String, color: ComposeColor) {
    Box(Modifier.clip(RoundedCornerShape(6.dp)).background(color.copy(alpha = 0.15f))
        .padding(horizontal = 6.dp, vertical = 2.dp)) {
        Text(text, style = MaterialTheme.typography.labelSmall, color = color, fontWeight = FontWeight.SemiBold)
    }
}
