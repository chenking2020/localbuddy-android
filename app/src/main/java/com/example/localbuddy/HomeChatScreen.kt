package com.example.localbuddy

import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 首页两种交互模式：文本对话 / 语音通话。
 * 二者共处同一个对话框里，来回切换不需要离开首页。
 */
enum class HomeChatMode(val label: String) {
    TEXT("文本对话"),
    VOICE("语音通话")
}

/** 首页智能体宫格一行几张小卡片。 */
private const val AGENT_GRID_COLUMNS = 3

/** 语音输入的阶段：空闲 / 加载模型 / 录音转写中 / 停止后收尾识别中。 */
private enum class VoiceInputPhase { IDLE, LOADING, RECORDING, RECOGNIZING }

private const val TAG = "HomeChatScreen"

/**
 * 融合首页：顶部应用栏 + 六个智能体小卡片 + 模式切换 + 对话/通话区 + 底部输入/通话控件。
 *
 * 两个专属页面（TextChatScreen / VoiceCallScreen）的状态与业务逻辑原样复用，
 * 这里只负责把它们装进同一个对话框；两个 ViewModel 共享 LlmEngine 单例，
 * 所以同屏只会有一份模型常驻内存。
 */
@Composable
fun HomeChatScreen(
    agents: List<FunctionItem>,
    onAgentClick: (String) -> Unit,
    onSettingsClick: () -> Unit,
    textVm: TextChatViewModel = viewModel(),
    callVm: VoiceCallViewModel = viewModel()
) {
    val ts by textVm.state.collectAsState()
    val cs by callVm.state.collectAsState()
    val sessions by textVm.sessions.collectAsState()
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current

    var mode by remember { mutableStateOf(HomeChatMode.TEXT) }
    /** 左侧抽屉：历史记录与「新建任务」入口都在这里。 */
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)

    val chatList = rememberLazyListState()
    val callList = rememberLazyListState()
    val scope = rememberCoroutineScope()

    // ── 文本对话：上传文档（针对文档问答） ──
    val pickMimes = arrayOf(
        "text/plain",
        "application/pdf",
        "image/*",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    )
    val pickLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            val mime = context.contentResolver.getType(uri)
            val name = runCatching {
                context.contentResolver.query(uri, null, null, null, null)?.use { c ->
                    val i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    if (i >= 0 && c.moveToFirst()) c.getString(i) else null
                }
            }.getOrNull() ?: uri.lastPathSegment ?: "文档"
            textVm.attachDoc(uri, name, mime)
        }
    }

    // ── 麦克风权限：语音通话与语音输入共用同一个申请器 ──
    var pendingMicAction by remember { mutableStateOf<(() -> Unit)?>(null) }
    var pendingMicDenied by remember { mutableStateOf<String?>(null) }
    var voicePhase by remember { mutableStateOf(VoiceInputPhase.IDLE) }
    var voiceError by remember { mutableStateOf<String?>(null) }
    var recorder by remember { mutableStateOf<MicRecorder?>(null) }
    /** 已请求停止：模型还在加载时点了停止/切走，采集协程据此不再开麦。 */
    val voiceStop = remember { java.util.concurrent.atomic.AtomicBoolean(false) }
    /** 剩余语音整段丢弃（发送/切模式时不再往输入框里补字）。 */
    val voiceDiscard = remember { java.util.concurrent.atomic.AtomicBoolean(false) }
    /** 本轮转写「已确认」的文字前缀 = 输入框里除当前段流式文本之外的内容。 */
    var voiceBase by remember { mutableStateOf("") }
    /** 采集 → 推理的音频段队列：有界、推理跟不上时丢最旧段，不让推理拖住麦克风。 */
    var voiceChannel by remember { mutableStateOf<Channel<FloatArray>?>(null) }
    /** 当前正在转写的段序号：只让最新一段的流式文本进输入框，避免迟到 token 写花内容。 */
    val voiceSegSeq = remember { java.util.concurrent.atomic.AtomicInteger(0) }

    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        val action = pendingMicAction
        val denied = pendingMicDenied
        pendingMicAction = null
        pendingMicDenied = null
        if (granted) action?.invoke() else voiceError = denied ?: "需要麦克风权限"
    }

    /** 麦克风权限统一入口：已授权直接执行，否则弹系统授权框。 */
    fun requireMic(action: () -> Unit, deniedMsg: String) {
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) {
            action()
        } else {
            pendingMicAction = action
            pendingMicDenied = deniedMsg
            micLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    fun requestCall() {
        requireMic({ callVm.startCall() }, "需要麦克风权限才能进行语音通话")
    }

    /**
     * 开启语音输入：采集与推理解耦（同 SpeechMT 录音模式）——
     * 采集协程只读麦克风、把 VAD 段塞进有界 Channel；推理协程按序消费做 ASR，
     * 并把结果实时追加到输入框现有文字之后（[appendVoiceText]）。
     *
     * 以开启瞬间输入框里的内容为基准 [voiceBase]，之后每段转写都追加在它后面，
     * 所以「打字写一半 → 开麦接着说」或「说完停止 → 再开一次」都能续在同一段文字里。
     */
    fun startVoiceInput() {
        if (voicePhase != VoiceInputPhase.IDLE) return
        voiceError = null
        voiceStop.set(false)
        voiceDiscard.set(false)
        requireMic({
            voiceBase = ts.input
            voicePhase = VoiceInputPhase.LOADING
            val rec = MicRecorder(pauseMs = 900, maxSegMs = 8_000)
            recorder = rec
            val ch = Channel<FloatArray>(capacity = 8, onBufferOverflow = BufferOverflow.DROP_OLDEST)
            voiceChannel = ch

            // ① 采集协程：只管麦克风，推理快慢不影响录音；队列满丢最旧段
            scope.launch(Dispatchers.IO) {
                val engine = SpeechMTEngine.get(context.applicationContext)
                try {
                    engine.warmUp()                  // 幂等；首次会解包并加载 ASR 模型
                    // 加载期间用户可能已点停止
                    if (voiceStop.get()) return@launch
                    withContext(Dispatchers.Main) { voicePhase = VoiceInputPhase.RECORDING }
                    // trySend：停止后若队列已关闭也不抛异常（send 会抛 ClosedSendChannelException）
                    rec.stream().collect { ch.trySend(it) }
                } catch (e: Throwable) {
                    if (e is CancellationException) throw e
                    Log.e(TAG, "voice capture failed", e)
                    withContext(Dispatchers.Main) {
                        voiceError = "录音失败：${e.message ?: "未知错误"}"
                    }
                } finally {
                    rec.stop()
                    // 采集结束 → 推理排空后退出；可能已被 stopVoiceInput 关过，忽略重复关闭
                    runCatching { ch.close() }
                }
            }

            // ② 推理协程：按序转写，token 级流式先顶到输入框，段结束再固化到 [voiceBase]
            scope.launch(Dispatchers.Default) {
                val engine = SpeechMTEngine.get(context.applicationContext)
                for (pcm in ch) {
                    if (voiceDiscard.get()) break
                    val seg = voiceSegSeq.incrementAndGet()
                    val text = runCatching {
                        engine.asrDetect(AudioDecoder.normalizePeak(pcm)) { partial ->
                            // 当前段的流式结果：输入框 = 已确认前缀 + 正在生成的这段
                            val p = partial.trimStart()
                            if (p.isNotEmpty() && voiceSegSeq.get() == seg) {
                                scope.launch(Dispatchers.Main.immediate) {
                                    if (voiceSegSeq.get() == seg) {
                                        textVm.setInput(appendVoiceText(voiceBase, p))
                                    }
                                }
                            }
                        }.text.trim()
                    }.onFailure { Log.e(TAG, "asr failed", it) }.getOrNull()
                    withContext(Dispatchers.Main) {
                        if (voiceDiscard.get()) return@withContext
                        // 段完成：以已确认前缀为基准追加；若期间用户手动改过或已发送清空了
                        // 输入框，则以输入框当前内容为新基准，避免把旧文字又写回去
                        val base = if (ts.input.startsWith(voiceBase)) voiceBase else ts.input
                        voiceBase = if (text.isNullOrBlank()) base else appendVoiceText(base, text)
                        textVm.setInput(voiceBase)
                    }
                }
                withContext(Dispatchers.Main) {
                    recorder = null
                    voiceChannel = null
                    voicePhase = VoiceInputPhase.IDLE
                }
            }
        }, "需要麦克风权限才能使用语音输入")
    }

    /**
     * 停止语音输入。已采集但未转写的段会继续识别完并追加进输入框；
     * @param discard 为 true 时剩余段直接丢弃（发送/切换模式场景：不该再往输入框里补字）。
     */
    fun stopVoiceInput(discard: Boolean = false) {
        if (voicePhase == VoiceInputPhase.IDLE) return
        voiceStop.set(true)
        if (discard) voiceDiscard.set(true)
        recorder?.stop()                     // 采集流退出 → channel.close() → 推理排空后回 IDLE
        // 兜底：模型还在加载、采集流根本没起来时，直接关队列让推理协程退出
        runCatching { voiceChannel?.close() }
        if (voicePhase != VoiceInputPhase.RECOGNIZING) voicePhase = VoiceInputPhase.RECOGNIZING
    }

    /**
     * 语音输入开关：点一下开启（录音 + 实时转写，文字逐段追加进输入框），
     * 再点一下停止——剩下的尾巴识别完也追加进去。之后就完全等同手打的文字，由用户点发送。
     */
    fun toggleVoiceInput() {
        if (voicePhase == VoiceInputPhase.IDLE) startVoiceInput() else stopVoiceInput()
    }

    /**
     * 切到某个模式。两种模式都要用麦克风，LLM 推理同时只能服务一路，所以切换前先把另一路停下来。
     */
    fun switchTo(m: HomeChatMode) {
        if (m == mode) return
        if (m == HomeChatMode.VOICE) {
            focusManager.clearFocus()
            if (ts.generating) textVm.stop()
            stopVoiceInput(discard = true)   // 语音输入还在录的话先收掉，别和通话抢麦克风
        } else {
            callVm.hangUp()
        }
        mode = m
    }

    /** 开一个新任务：断开当前会话（旧的已进历史），按选中的类型切换模式。 */
    fun startNewTask(m: HomeChatMode) {
        focusManager.clearFocus()
        textVm.clearChat()
        switchTo(m)
    }

    // 离开首页彻底释放麦克风：本页不像 VoiceCallScreen 那样有单独的返回动作，
    // 只能在页面销毁时兜底挂断。
    DisposableEffect(Unit) {
        onDispose {
            callVm.hangUp()
            stopVoiceInput(discard = true)
        }
    }

    // ── 文本流式输出的自动跟随 ──
    // 与 TextChatScreen 同一套逻辑（常驻协程 + 滚动锚点），不能写成
    // LaunchedEffect(末条消息内容)：那会每个 token 重启一次协程，滚动被反复取消。
    LaunchedEffect(Unit) {
        var stick = true
        var anchor: Pair<Int, Int>? = null
        var lastSize = 0
        var lastViewportH = 0

        snapshotFlow {
            val m = ts.messages
            val last = m.lastOrNull()
            val info = chatList.layoutInfo
            val first = info.visibleItemsInfo.firstOrNull()
            BottomSnap(
                size = m.size,
                textLen = last?.text?.length ?: -1,
                thinkLen = last?.thinking?.length ?: -1,
                anchorIndex = first?.index ?: -1,
                anchorOffset = first?.offset ?: 0,
                total = info.totalItemsCount,
                viewportH = info.viewportSize.height
            )
        }
            .distinctUntilChanged()
            .collect { snap ->
                if (snap.total == 0) return@collect
                val a = snap.anchorIndex to snap.anchorOffset
                val scrolled = a != anchor
                anchor = a
                // 视口变矮/变高（输入框展开、输入法弹出收起）：此时列表位置没动但可视范围变了，
                // 不能拿"距底部距离"重新判定——那样会误判成不在底部、从此不再跟随，
                // 只剩"贴着底部的补一次滚"。
                val viewportChanged = lastViewportH != 0 && snap.viewportH != lastViewportH
                lastViewportH = snap.viewportH
                if (snap.size != lastSize) {
                    stick = true
                    lastSize = snap.size
                } else if (scrolled && !viewportChanged) {
                    stick = chatList.isNearBottom()
                }
                if (stick) chatList.scrollToBottom()
            }
    }

    // 消息条数一变就无条件贴底一次：兜住上面 stick 万一被判成 false 之后"再也不跟随"的情况
    LaunchedEffect(ts.messages.size) {
        if (ts.messages.isNotEmpty()) chatList.scrollToBottom()
    }

    // 通话转写追加时滚到最新一条
    LaunchedEffect(cs.turns.size) {
        if (cs.turns.isNotEmpty()) runCatching { callList.animateScrollToItem(cs.turns.lastIndex) }
    }

    // 智能体小卡片只在「首页最初状态」出现：一旦进入对话/通话就整个收起，
    // 从抽屉里开新任务回到初始态后才重新显示。
    val idle = if (mode == HomeChatMode.TEXT) ts.messages.isEmpty()
    else !cs.inCall && cs.turns.isEmpty()

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            AppDrawer(
                sessions = sessions,
                currentId = textVm.currentSessionId,
                onNewTask = {
                    scope.launch { drawerState.close() }
                    startNewTask(HomeChatMode.TEXT)
                },
                onOpenSession = { id ->
                    textVm.openSession(id)
                    switchTo(HomeChatMode.TEXT)
                    scope.launch { drawerState.close() }
                },
                onDeleteSession = textVm::deleteSession,
                onSettings = {
                    scope.launch { drawerState.close() }
                    onSettingsClick()
                }
            )
        }
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
        ) {
            HomeTopBar(
                onMenuClick = { scope.launch { drawerState.open() } },
                // 电话图标：激活 / 退出语音通话模式（豆包式入口），设置齿轮在它右边
                onCallClick = {
                    focusManager.clearFocus()
                    if (mode == HomeChatMode.VOICE) {
                        callVm.hangUp()
                        mode = HomeChatMode.TEXT
                    } else {
                        switchTo(HomeChatMode.VOICE)
                    }
                },
                callActive = mode == HomeChatMode.VOICE,
                onSettingsClick = onSettingsClick
            )

            if (idle) {
                AgentGrid(
                    agents = agents,
                    onAgentClick = onAgentClick,
                    modifier = Modifier.padding(top = 12.dp)
                )
            }

            if (mode == HomeChatMode.TEXT) {
                ts.error?.let { err -> ErrorBanner(text = err, onDismiss = textVm::dismissError) }
                if (ts.preparing) {
                    PreparingRow(ts.preparingText, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
                }
            } else {
                cs.error?.let { err -> CallErrorBanner(text = err, onDismiss = callVm::dismissError) }
            }

            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                if (mode == HomeChatMode.TEXT) {
                    if (ts.messages.isEmpty()) {
                        // 首屏已有智能体卡片，空态只留一句引导语，不占空间
                        ChatEmptyState(onPick = { textVm.setInput(it) }, compact = true)
                    } else {
                        LazyColumn(
                            state = chatList,
                            modifier = Modifier.fillMaxSize(),
                            // 底部多留一点：最后一条气泡不贴着输入栏，视觉上不会被压住
                            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 16.dp),
                            verticalArrangement = Arrangement.spacedBy(14.dp)
                        ) {
                            items(ts.messages, key = { it.id }) { m ->
                                ChatBubble(
                                    msg = m,
                                    accent = AgentBlue,
                                    // 精致模式的 thinking 里装的是执行方案，不是模型的 <think> 推理链
                                    planDoing = "生成执行方案中…",
                                    planDone = "已生成执行方案"
                                )
                            }
                        }
                    }
                } else {
                    Column(modifier = Modifier.fillMaxSize()) {
                        CallStage(
                            phase = cs.phase,
                            inCall = cs.inCall,
                            startedAtMs = cs.startedAtMs,
                            preparing = cs.preparing,
                            preparingText = cs.preparingText,
                            modifier = Modifier.fillMaxWidth()
                        )
                        if (cs.turns.isEmpty()) {
                            CallHint(ttsReady = cs.ttsReady, modifier = Modifier.weight(1f))
                        } else {
                            LazyColumn(
                                state = callList,
                                modifier = Modifier.weight(1f).fillMaxWidth(),
                                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                items(cs.turns, key = { it.id }) { t -> CallTurnRow(t) }
                                item { Spacer(Modifier.height(4.dp)) }
                            }
                        }
                    }
                }
            }

            // 语音输入自身的错误提示（权限被拒 / 没听清 / 识别失败），5 秒后自动消失
            voiceError?.let { err ->
                ErrorBanner(text = err, onDismiss = { voiceError = null })
            }

            // 语音输入过程中的状态条（录音中 / 识别中）
            if (voicePhase != VoiceInputPhase.IDLE) {
                VoiceInputBar(
                    phase = voicePhase,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
                )
            }

            // 底部两个控件各自处理导航栏/输入法内边距，外层不再重复加
            if (mode == HomeChatMode.TEXT) {
                ChatInputBar(
                    value = ts.input,
                    onValue = textVm::setInput,
                    generating = ts.generating,
                    docName = ts.docName,
                    docChars = ts.docChars,
                    onRemoveDoc = textVm::removeDoc,
                    onPickDoc = { runCatching { pickLauncher.launch(pickMimes) } },
                    onSend = {
                        focusManager.clearFocus()
                        stopVoiceInput(discard = true)   // 发送即结束本轮转写，剩余不再补字
                        textVm.send()
                    },
                    onStop = textVm::stop,
                    thinkMode = ts.thinkMode,
                    onThinkMode = textVm::setThinkMode,
                    // 输入框展开（点输入框/弹输入法）会把列表视口挤矮，补一次滚到底，
                    // 免得最后一条消息被挤到看不见的地方
                    onFocusChange = { if (it) scope.launch { chatList.scrollToBottom() } },
                    showMic = true,
                    onMicClick = ::toggleVoiceInput,
                    micActive = voicePhase != VoiceInputPhase.IDLE
                )
            } else {
                CallControls(
                    inCall = cs.inCall,
                    onStart = { requestCall() },
                    onHangUp = callVm::hangUp,
                    voices = cs.voices,
                    voiceIndex = cs.voiceIndex,
                    onPickVoice = callVm::setVoice
                )
            }
        }
    }
}

/**
 * 把新一段转写追加到已有文字后面：中文直接连接，只有两侧都是英文/数字时才补空格，
 * 免得 "hello"+"world" 粘成 "helloworld"（中文补空格会很难看，故只在 ASCII 词间接）。
 */
private fun appendVoiceText(base: String, text: String): String {
    if (base.isBlank()) return text.trimStart()
    if (text.isBlank()) return base.trimEnd()
    val b = base.trimEnd()
    val t = text.trimStart()
    val space = if (isAsciiWordChar(b.last()) && isAsciiWordChar(t.first())) " " else ""
    return b + space + t
}

private fun isAsciiWordChar(c: Char): Boolean =
    c in 'a'..'z' || c in 'A'..'Z' || c in '0'..'9'

// ── 语音输入状态条 ──
@Composable
private fun VoiceInputBar(phase: VoiceInputPhase, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(AgentRose.copy(alpha = 0.10f))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        when (phase) {
            VoiceInputPhase.RECORDING -> {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(Color(0xFFDC2626))
                )
                Text(
                    text = "正在听…转写文字实时写入输入框，点击麦克风按钮结束",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
            }
            VoiceInputPhase.RECOGNIZING -> {
                CircularProgressIndicator(
                    modifier = Modifier.size(13.dp),
                    strokeWidth = 2.dp,
                    color = AgentRose
                )
                Text(
                    text = "正在识别最后一段…",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
            }
            else -> {
                CircularProgressIndicator(
                    modifier = Modifier.size(13.dp),
                    strokeWidth = 2.dp,
                    color = AgentRose
                )
                Text(
                    text = "正在加载语音识别模型…首次使用需要一点时间",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}
// ── 左侧抽屉：新建任务 + 历史记录 ──
@Composable
private fun AppDrawer(
    sessions: List<ChatSession>,
    currentId: String?,
    onNewTask: () -> Unit,
    onOpenSession: (String) -> Unit,
    onDeleteSession: (String) -> Unit,
    onSettings: () -> Unit
) {
    // 抽屉宽度 = 屏宽的 80%，高度充满（ModalDrawerSheet 默认只给 360dp）
    val drawerWidth = LocalConfiguration.current.screenWidthDp.dp * 0.8f
    ModalDrawerSheet(
        modifier = Modifier.fillMaxHeight().width(drawerWidth),
        drawerShape = RoundedCornerShape(topEnd = 20.dp, bottomEnd = 20.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxHeight()
                .padding(vertical = 8.dp)
        ) {
            // ── 头部 ──
            Row(
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(
                            Brush.linearGradient(
                                listOf(AgentPurple, lerp(AgentPurple, Color.White, 0.25f))
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.AutoAwesome,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }
                Spacer(modifier = Modifier.width(11.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "端侧离线工作搭子",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = "全程离线 · 数据不出设备",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // ── 新建任务 ──
            Button(
                onClick = onNewTask,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                shape = RoundedCornerShape(14.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Add,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("新建任务", style = MaterialTheme.typography.labelLarge)
            }

            Spacer(modifier = Modifier.height(14.dp))

            // ── 历史记录 ──
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "历史记录",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(1f)
                )
                if (sessions.isNotEmpty()) {
                    Text(
                        text = "${sessions.size} 条",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            if (sessions.isEmpty()) {
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp),
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = "还没有历史会话",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "点上面的「新建任务」开始第一段对话，结束后会自动存到这里。",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f),
                        lineHeight = 17.sp
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    items(sessions, key = { it.id }) { s ->
                        SessionRow(
                            session = s,
                            selected = s.id == currentId,
                            onClick = { onOpenSession(s.id) },
                            onDelete = { onDeleteSession(s.id) }
                        )
                    }
                    item { Spacer(modifier = Modifier.height(8.dp)) }
                }
            }

            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.5f)
            )
            NavigationDrawerItem(
                label = { Text("设置", style = MaterialTheme.typography.labelLarge) },
                icon = {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                },
                selected = false,
                onClick = onSettings,
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier.padding(horizontal = 12.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

/** 抽屉里的一条历史会话：标题 + 时间条数，右侧删除。 */
@Composable
private fun SessionRow(
    session: ChatSession,
    selected: Boolean,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(
                if (selected) MaterialTheme.colorScheme.secondaryContainer
                else Color.Transparent
            )
            .clickable(onClick = onClick)
            .padding(start = 12.dp, end = 4.dp, top = 9.dp, bottom = 9.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Icon(
            imageVector = Icons.AutoMirrored.Filled.Chat,
            contentDescription = null,
            tint = if (selected) MaterialTheme.colorScheme.onSecondaryContainer
            else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(17.dp)
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = session.title.ifBlank { "未命名会话" },
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Medium,
                color = if (selected) MaterialTheme.colorScheme.onSecondaryContainer
                else MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "${formatSessionTime(session.updatedAt)} · ${session.messages.size} 条",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        IconButton(onClick = onDelete, modifier = Modifier.size(34.dp)) {
            Icon(
                imageVector = Icons.Default.DeleteOutline,
                contentDescription = "删除会话",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(17.dp)
            )
        }
    }
}

private fun formatSessionTime(ms: Long): String =
    SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(ms))
// ── 顶部应用栏 ──
/**
 * 顶部应用栏：左侧汉堡键拉开抽屉（历史 / 新建任务），右侧是语音通话开关与设置。
 * 电话图标亮起表示处于语音通话模式，再点一次挂断并回到文本对话（豆包式入口）。
 */
@Composable
private fun HomeTopBar(
    onMenuClick: () -> Unit,
    onCallClick: () -> Unit,
    callActive: Boolean,
    onSettingsClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.statusBars)
            .padding(start = 8.dp, end = 12.dp, top = 8.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        IconButton(onClick = onMenuClick) {
            Icon(
                imageVector = Icons.Default.Menu,
                contentDescription = "打开侧边栏",
                tint = MaterialTheme.colorScheme.onBackground
            )
        }
        Spacer(modifier = Modifier.width(4.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "端侧离线工作搭子",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground
            )
            Text(
                text = "离线智能服务",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        IconButton(onClick = onCallClick) {
            Icon(
                imageVector = Icons.Default.Call,
                contentDescription = if (callActive) "退出语音通话" else "语音通话",
                tint = if (callActive) AgentRose else MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        IconButton(onClick = onSettingsClick) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = "设置",
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}


// ── 智能体宫格（未开始对话时展示，位于对话框上方） ──
// 整块保持紧凑：分区标题压成一行小字，卡片只留图标 + 单行名字
@Composable
private fun AgentGrid(
    agents: List<FunctionItem>,
    onAgentClick: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "更多智能体",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "${agents.size} 个",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            agents.chunked(AGENT_GRID_COLUMNS).forEach { row ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(IntrinsicSize.Max),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    row.forEach { item ->
                        AgentMiniCard(
                            item = item,
                            onClick = { onAgentClick(item.route) },
                            modifier = Modifier.weight(1f).fillMaxHeight()
                        )
                    }
                    // 末行不满时补占位，保证每行卡片等宽
                    repeat(AGENT_GRID_COLUMNS - row.size) { Spacer(modifier = Modifier.weight(1f)) }
                }
            }
        }
    }
}

/**
 * 小卡片：只显示图标和智能体名字（tag 与描述都不展示），六宫格排两行。
 * 名字去掉「智能体」后缀以便在卡片宽度内单行放下，整卡点击进入对应智能体。
 */
@Composable
private fun AgentMiniCard(
    item: FunctionItem,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val enabled = !item.isComingSoon
    val alpha = if (enabled) 1f else 0.45f
    val shape = RoundedCornerShape(16.dp)
    Column(
        modifier = modifier
            .clip(shape)
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.35f), shape)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 4.dp, vertical = 9.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(5.dp, Alignment.CenterVertically)
    ) {
        Box(
            modifier = Modifier
                .size(30.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(item.accent.copy(alpha = 0.14f * alpha)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = item.icon,
                contentDescription = null,
                tint = item.accent.copy(alpha = alpha),
                modifier = Modifier.size(17.dp)
            )
        }
        Text(
            text = item.title.removeSuffix("智能体"),
            style = MaterialTheme.typography.labelMedium.copy(fontSize = 11.sp),
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = alpha),
            textAlign = TextAlign.Center,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            lineHeight = 14.sp
        )
    }
}


