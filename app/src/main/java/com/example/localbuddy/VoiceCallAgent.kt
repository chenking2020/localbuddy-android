package com.example.localbuddy

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.CallEnd
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

// ──────────────────────────────────────────────────────────────────
// 语音通话智能体
// ──────────────────────────────────────────────────────────────────

/** 通话状态机：半双工（听完再说），避免把自己的外放声音又听进去。 */
enum class CallPhase(val label: String) {
    IDLE("未通话"),
    LISTENING("正在聆听…"),
    RECOGNIZING("识别中…"),
    THINKING("思考中…"),
    SPEAKING("正在说话…")
}

data class CallTurn(val id: Long, val role: String, val text: String)

data class VoiceCallUiState(
    val inCall: Boolean = false,
    val phase: CallPhase = CallPhase.IDLE,
    val startedAtMs: Long = 0L,
    val turns: List<CallTurn> = emptyList(),
    val preparing: Boolean = false,
    val preparingText: String = "",
    val voices: List<TtsEngine.BuiltinVoice> = emptyList(),
    val voiceIndex: Int = 0,
    /** 语音合成是否就绪；未就绪时回复只以文字呈现（不阻塞对话）。 */
    val ttsReady: Boolean = false,
    val error: String? = null
)

private const val TAG = "VoiceCallVM"

/** 通话里保留的最近轮次数（0.6B 上下文有限，通话也不依赖很长的记忆）。 */
private const val CALL_HISTORY_TURNS = 4

/** 通话回复的生成上限：三句话以内，太长用户在通话里听不完（也是 TTS 时长的上限）。 */
private const val CALL_MAX_NEW = 160

/** 单句语音上限（12.5 帧/秒）：250 帧 ≈ 20s。 */
private const val CALL_MAX_FRAMES = 250

/** 低于该长度的 PCM 直接丢弃（<0.4s 基本是噪声碎片，送 ASR 只会出乱码）。 */
private const val MIN_UTTERANCE_SAMPLES = 16_000 * 2 / 5

class VoiceCallViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(VoiceCallUiState())
    val state: StateFlow<VoiceCallUiState> = _state.asStateFlow()

    private val llm = LlmEngine.get(app)
    private val tts = TtsEngine.get(app)
    private val recorder = MicRecorder(pauseMs = 600, maxSegMs = 8_000)
    private val player = PcmPlayer()

    private var callJob: Job? = null
    private var ttsJob: Job? = null
    private var idSeq = 0L
    private val history = ArrayList<ChatTurn>(CALL_HISTORY_TURNS + 2)

    fun setVoice(i: Int) = _state.update { it.copy(voiceIndex = i) }
    fun dismissError() = _state.update { it.copy(error = null) }
    fun fail(msg: String) = _state.update { it.copy(error = msg) }

    /** 挂断：停录音、停播放、取消所有后台任务。 */
    fun hangUp() {
        recorder.stop()
        player.stop()
        callJob?.cancel(); callJob = null
        ttsJob?.cancel(); ttsJob = null
        history.clear()
        _state.update {
            it.copy(inCall = false, phase = CallPhase.IDLE, preparing = false, preparingText = "")
        }
    }

    /** 开始通话：加载模型 → 进入聆听 → 循环「听一句 → 识别 → 生成 → 说一句」。 */
    fun startCall() {
        if (_state.value.inCall) return
        _state.update {
            it.copy(inCall = true, phase = CallPhase.IDLE, turns = emptyList(),
                startedAtMs = System.currentTimeMillis(), preparing = true,
                preparingText = "加载模型中…", error = null)
        }
        val ctx = getApplication<Application>()
        callJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                // ASR + LLM 是通话的必经链路，必须等；TTS 体积大（~780MB）放后台并行加载，
                // 没加载完之前回复只显示文字，不阻塞对话。
                _state.update { it.copy(preparingText = "加载语音识别模型中…") }
                SpeechMTEngine.get(ctx).warmUp()
                if (!isActive) return@launch
                _state.update { it.copy(preparingText = "加载对话模型中…") }
                llm.warmUp()
                if (!isActive) return@launch
                ttsJob = viewModelScope.launch(Dispatchers.IO) {
                    runCatching {
                        _state.update { it.copy(preparingText = "加载语音合成模型中…") }
                        tts.warmUp()
                        _state.update {
                            it.copy(voices = tts.builtinVoices(), ttsReady = tts.canSpeak)
                        }
                    }.onFailure { Log.w(TAG, "TTS load failed, call falls back to text-only", it) }
                    _state.update { it.copy(preparing = false, preparingText = "") }
                }

                _state.update { it.copy(preparing = false, preparingText = "", phase = CallPhase.LISTENING) }
                // 一条流跑到挂断为止：emit 在处理期间挂起 → 采集循环随之暂停，
                // 处理完恢复时 AudioRecord 缓冲已被覆盖，拿到的是最新音频，不会有陈旧回声。
                recorder.stream().collect { pcm ->
                    if (!_state.value.inCall) return@collect
                    // 半双工：识别/生成/播放期间丢弃采集段（否则会把自己外放的声音听进去）
                    if (_state.value.phase != CallPhase.LISTENING) return@collect
                    runCatching { processUtterance(pcm) }
                        .onFailure { e ->
                            if (e is kotlinx.coroutines.CancellationException) throw e
                            Log.e(TAG, "turn failed", e)
                        }
                    if (_state.value.inCall) {
                        _state.update { it.copy(phase = CallPhase.LISTENING) }
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // 挂断
            } catch (e: Throwable) {
                Log.e(TAG, "call failed", e)
                _state.update {
                    it.copy(inCall = false, phase = CallPhase.IDLE, preparing = false,
                        preparingText = "", error = "通话中断：${e.message ?: "未知"}")
                }
            }
        }
    }

    /** 一句话的完整处理：ASR → LLM → TTS → 播放。 */
    private suspend fun processUtterance(pcm: FloatArray) {
        if (pcm.size < MIN_UTTERANCE_SAMPLES) return
        _state.update { it.copy(phase = CallPhase.RECOGNIZING) }

        val heard = SpeechMTEngine.get(getApplication<Application>())
            .asrDetect(AudioDecoder.normalizePeak(pcm)).text.trim()
        if (heard.isBlank()) return
        if (!_state.value.inCall) return

        val userId = addTurn(ROLE_USER, heard)
        _state.update { it.copy(phase = CallPhase.THINKING) }

        val turns = buildList {
            add(ChatTurn(ROLE_SYSTEM, CALL_SYSTEM))
            history.forEach { add(it) }
            add(ChatTurn(ROLE_USER, heard))
        }
        val ansId = addTurn(ROLE_ASSISTANT, "")
        val reply = llm.chat(
            messages = turns,
            maxNew = CALL_MAX_NEW,
            enableThinking = false,     // 通话要的是快，思考链会先吞掉几百个 token
            onPartial = { p -> patch(ansId, p) }
        )
        val final = reply.trim().ifBlank { "抱歉，我没听清，能再说一遍吗？" }
        patch(ansId, final)

        history.add(ChatTurn(ROLE_USER, heard))
        history.add(ChatTurn(ROLE_ASSISTANT, final))
        while (history.size > CALL_HISTORY_TURNS) history.removeAt(0)
        if (!_state.value.inCall) return

        // 说话：未就绪时跳过（回复已以文字呈现）
        if (tts.canSpeak) {
            _state.update { it.copy(phase = CallPhase.SPEAKING) }
            val speech = cleanForSpeech(final)
            val audio = runCatching { tts.synthesize(speech, _state.value.voiceIndex, CALL_MAX_FRAMES) }
                .onFailure { Log.w(TAG, "tts failed", it) }.getOrNull()
            if (audio != null && _state.value.inCall) {
                runCatching { player.play(audio.pcmInterleaved, audio.sampleRate, 2) }
                    .onFailure { Log.w(TAG, "playback failed", it) }
            }
        }
    }

    private fun addTurn(role: String, text: String): Long {
        val id = idSeq++
        _state.update { it.copy(turns = it.turns + CallTurn(id, role, text)) }
        return id
    }

    private fun patch(id: Long, text: String) {
        _state.update { s ->
            s.copy(turns = s.turns.map { if (it.id == id) it.copy(text = text) else it })
        }
    }

    override fun onCleared() {
        hangUp()
        player.release()
        super.onCleared()
    }
}

/**
 * 通话场景的 system：核心是「口语、短」。
 *
 * 与文本对话的差异：TTS 会把文本逐字念出来——Markdown、列表符号、标题都会被读成杂音，
 * 所以这里明确禁用一切书面格式；长度也压到三句话内（端上生成慢 + 用户听不完双重约束）。
 */
private const val CALL_SYSTEM =
    "你正在和用户进行实时语音通话。要求：\n" +
        "1. 用口语化的短句回答，像真人说话一样自然：不要用列表、不要用 Markdown、不要写标题和编号；\n" +
        "2. 每轮回答控制在三句话以内，只说最关键的内容；\n" +
        "3. 使用与用户相同的语言；\n" +
        "4. 直接回答，不要复述用户的话，不要加“好的”“明白了”之类的客套前缀；\n" +
        "5. 不要输出表情符号、括号补充说明或任何不适合朗读的符号。"

/** 去掉不适合朗读的符号（TTS 会把 `*`、`` ` ``、`#` 之类念出来或跳过造成断句怪异）。 */
private fun cleanForSpeech(text: String): String =
    text.replace(Regex("[*`#~_>|\\[\\]()（）【】《》]"), " ")
        .replace(Regex("\\s+"), " ")
        .trim()
        .ifBlank { text }

// ──────────────────────────────────────────────────────────────────
// UI
// ──────────────────────────────────────────────────────────────────

@Composable
fun VoiceCallScreen(onBack: () -> Unit, vm: VoiceCallViewModel = viewModel()) {
    val s by vm.state.collectAsState()
    val context = androidx.compose.ui.platform.LocalContext.current
    val listState = rememberLazyListState()

    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.startCall() else vm.fail("需要麦克风权限才能进行语音通话")
    }

    fun requestStart() {
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) vm.startCall() else micLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    DisposableEffect(Unit) { onDispose { vm.hangUp() } }

    LaunchedEffect(s.turns.size) {
        if (s.turns.isNotEmpty()) runCatching { listState.animateScrollToItem(s.turns.lastIndex) }
    }

    // 顶部状态栏由 AgentHeader 自己处理，底部导航条由 CallControls 处理，不再叠加 Scaffold 的 innerPadding。
    Scaffold(containerColor = MaterialTheme.colorScheme.background) { _ ->
        Column(modifier = Modifier.fillMaxSize()) {
            AgentHeader(
                accent = AgentRose,
                icon = Icons.Default.Call,
                title = "语音通话智能体",
                subtitle = "语音 · 实时 · 全程离线",
                onBack = onBack
            )

            s.error?.let { err -> CallErrorBanner(err) { vm.dismissError() } }

            CallStage(
                phase = s.phase,
                inCall = s.inCall,
                startedAtMs = s.startedAtMs,
                preparing = s.preparing,
                preparingText = s.preparingText,
                modifier = Modifier.fillMaxWidth()
            )

            if (s.turns.isEmpty()) {
                CallHint(ttsReady = s.ttsReady, modifier = Modifier.weight(1f))
            } else {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(s.turns, key = { it.id }) { t -> CallTurnRow(t) }
                    item { Spacer(Modifier.height(4.dp)) }
                }
            }

            CallControls(
                inCall = s.inCall,
                onStart = { requestStart() },
                onHangUp = vm::hangUp,
                voices = s.voices,
                voiceIndex = s.voiceIndex,
                onPickVoice = vm::setVoice
            )
        }
    }
}

/** 通话主视觉：状态 + 计时 + 头像 + 声波。 */
@Composable
internal fun CallStage(
    phase: CallPhase,
    inCall: Boolean,
    startedAtMs: Long,
    preparing: Boolean,
    preparingText: String,
    modifier: Modifier = Modifier
) {
    var elapsed by remember { mutableLongStateOf(0L) }
    LaunchedEffect(inCall, startedAtMs) {
        while (inCall) {
            elapsed = System.currentTimeMillis() - startedAtMs
            delay(500)
        }
        elapsed = 0L
    }

    val bg = Brush.verticalGradient(
        listOf(
            AgentRose.copy(alpha = 0.10f),
            MaterialTheme.colorScheme.background
        )
    )
    Column(
        modifier = modifier
            .background(bg)
            .padding(vertical = 18.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // 状态 + 计时
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (inCall) {
                Box(
                    Modifier.size(7.dp).clip(CircleShape)
                        .background(if (phase == CallPhase.LISTENING) Color(0xFF22C55E) else AgentRose)
                )
            }
            Text(
                text = when {
                    preparing -> preparingText.ifBlank { "准备中…" }
                    !inCall -> "点击下方按钮开始通话"
                    else -> phase.label
                },
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        if (inCall) {
            Spacer(Modifier.height(2.dp))
            Text(
                formatDuration(elapsed),
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface
            )
        }

        Spacer(Modifier.height(16.dp))

        // 头像：聆听/说话时外圈呼吸
        val active = inCall && (phase == CallPhase.LISTENING || phase == CallPhase.SPEAKING)
        CallAvatar(active = active, speaking = phase == CallPhase.SPEAKING, inCall = inCall)

        Spacer(Modifier.height(14.dp))

        if (inCall) {
            WaveBars(
                active = phase == CallPhase.SPEAKING,
                color = AgentRose,
                modifier = Modifier.height(28.dp)
            )
        } else {
            Text(
                "全程离线：语音识别、对话生成、语音合成都在本机完成",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 28.dp)
            )
        }
    }
}

@Composable
internal fun CallAvatar(active: Boolean, speaking: Boolean, inCall: Boolean) {
    val t = rememberInfiniteTransition(label = "avatar")
    val pulse by t.animateFloat(
        initialValue = 1f, targetValue = 1.14f,
        animationSpec = infiniteRepeatable(tween(900, easing = LinearEasing), RepeatMode.Reverse),
        label = "pulse"
    )
    Box(contentAlignment = Alignment.Center, modifier = Modifier.size(132.dp)) {
        if (active) {
            Box(
                Modifier
                    .size(112.dp)
                    .scale(pulse)
                    .alpha(0.16f)
                    .clip(CircleShape)
                    .background(AgentRose)
            )
        }
        Box(
            modifier = Modifier
                .size(108.dp)
                .clip(CircleShape)
                .background(
                    Brush.linearGradient(
                        listOf(AgentRose, lerp(AgentRose, Color.White, 0.28f))
                    )
                )
                .border(1.dp, Color.White.copy(alpha = 0.22f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = when {
                    !inCall -> Icons.Default.Call
                    speaking -> Icons.AutoMirrored.Filled.VolumeUp
                    else -> Icons.Default.Mic
                },
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(44.dp)
            )
        }
    }
}

/** 说话时的声波条：激活时高低起伏，静默时压平。 */
@Composable
internal fun WaveBars(active: Boolean, color: Color, modifier: Modifier = Modifier) {
    val t = rememberInfiniteTransition(label = "wave")
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(11) { i ->
            val h by t.animateFloat(
                initialValue = 0.25f,
                targetValue = if (active) 1f else 0.3f,
                animationSpec = infiniteRepeatable(
                    tween(durationMillis = 520, delayMillis = i * 70, easing = LinearEasing),
                    RepeatMode.Reverse
                ),
                label = "bar$i"
            )
            Box(
                Modifier
                    .width(3.dp)
                    .fillMaxHeight((0.22f + h * 0.78f).coerceIn(0.12f, 1f))
                    .clip(RoundedCornerShape(2.dp))
                    .background(color.copy(alpha = 0.75f))
            )
        }
    }
}

/** 一条通话转写：用户靠右玫红，助手靠左灰底。 */
@Composable
internal fun CallTurnRow(turn: CallTurn) {
    val isUser = turn.role == ROLE_USER
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
    ) {
        Row(
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (!isUser) {
                Box(
                    Modifier.size(28.dp).clip(RoundedCornerShape(9.dp))
                        .background(AgentRose.copy(alpha = 0.13f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.RecordVoiceOver, null, tint = AgentRose, modifier = Modifier.size(16.dp))
                }
            }
            Box(
                modifier = Modifier
                    .weight(1f, fill = false)
                    .clip(RoundedCornerShape(14.dp))
                    .background(
                        if (isUser) AgentRose.copy(alpha = 0.12f)
                        else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f)
                    )
                    .padding(horizontal = 12.dp, vertical = 9.dp)
            ) {
                Text(
                    turn.text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
            if (isUser) {
                Box(
                    Modifier.size(28.dp).clip(RoundedCornerShape(9.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Mic, null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

/** 未开始通话时的说明区。 */
@Composable
internal fun CallHint(ttsReady: Boolean, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Default.GraphicEq, null,
            tint = AgentRose.copy(alpha = 0.35f), modifier = Modifier.size(40.dp))
        Spacer(Modifier.height(10.dp))
        Text(
            "像打电话一样直接说话",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold
        )
        Spacer(Modifier.height(6.dp))
        Text(
            "说完停顿一下，智能体就会识别并回答；它说话时请先听完再开口。\n" +
                if (ttsReady) "语音合成已就绪，回复会以语音播出。"
                else "首次通话需要加载语音合成模型，加载完成前的回复先以文字呈现。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            lineHeight = 19.sp
        )
    }
}

/** 底部：音色选择 + 大圆形通话/挂断按钮。 */
@Composable
internal fun CallControls(
    inCall: Boolean,
    onStart: () -> Unit,
    onHangUp: () -> Unit,
    voices: List<TtsEngine.BuiltinVoice>,
    voiceIndex: Int,
    onPickVoice: (Int) -> Unit
) {
    var showVoices by remember { mutableStateOf(false) }
    val navBar = WindowInsets.navigationBars.asPaddingValues()

    Surface(tonalElevation = 2.dp, shadowElevation = 6.dp, color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 12.dp)
                .padding(bottom = navBar.calculateBottomPadding() + 14.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (voices.isNotEmpty()) {
                Row(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f))
                        .clickable { showVoices = true }
                        .padding(horizontal = 12.dp, vertical = 7.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(Icons.AutoMirrored.Filled.VolumeUp, null,
                        tint = AgentRose, modifier = Modifier.size(15.dp))
                    Text(
                        voices.getOrNull(voiceIndex)?.displayName ?: "默认音色",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text("切换", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Spacer(Modifier.height(14.dp))
            }

            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .background(if (inCall) Color(0xFFDC2626) else Color(0xFF16A34A))
                    .clickable { if (inCall) onHangUp() else onStart() },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    if (inCall) Icons.Default.CallEnd else Icons.Default.Call,
                    contentDescription = if (inCall) "挂断" else "开始通话",
                    tint = Color.White,
                    modifier = Modifier.size(28.dp)
                )
            }
            Spacer(Modifier.height(8.dp))
            Text(
                if (inCall) "挂断" else "开始通话",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }

    if (showVoices) {
        VoicePickerDialog(
            voices = voices,
            selected = voiceIndex,
            onPick = { onPickVoice(it); showVoices = false },
            onDismiss = { showVoices = false }
        )
    }
}

@Composable
internal fun VoicePickerDialog(
    voices: List<TtsEngine.BuiltinVoice>,
    selected: Int,
    onPick: (Int) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("选择通话音色") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                voices.forEachIndexed { i, v ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (i == selected) AgentRose.copy(alpha = 0.12f) else Color.Transparent)
                            .clickable { onPick(i) }
                            .padding(horizontal = 8.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(v.displayName, fontWeight = FontWeight.SemiBold,
                                style = MaterialTheme.typography.bodyMedium)
                            if (v.group.isNotBlank()) {
                                Text(v.group, style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                        }
                        if (i == selected) {
                            Icon(Icons.Default.GraphicEq, null, tint = AgentRose, modifier = Modifier.size(16.dp))
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } }
    )
}

@Composable
internal fun CallErrorBanner(text: String, onDismiss: () -> Unit) {
    LaunchedEffect(text) { delay(4000); onDismiss() }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.errorContainer)
            .padding(horizontal = 12.dp, vertical = 9.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onErrorContainer, modifier = Modifier.weight(1f))
        Icon(Icons.Default.Close, "关闭", tint = MaterialTheme.colorScheme.onErrorContainer,
            modifier = Modifier.size(16.dp).clickable(onClick = onDismiss))
    }
}

internal fun formatDuration(ms: Long): String {
    val total = (ms / 1000).coerceAtLeast(0)
    val m = total / 60
    val sec = total % 60
    return "%02d:%02d".format(m, sec)
}
