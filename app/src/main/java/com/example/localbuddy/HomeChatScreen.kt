package com.example.localbuddy

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.flow.distinctUntilChanged

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
    var showHistory by remember { mutableStateOf(false) }

    val chatList = rememberLazyListState()
    val callList = rememberLazyListState()

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

    // ── 语音通话：麦克风权限 ──
    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) callVm.startCall() else callVm.fail("需要麦克风权限才能进行语音通话")
    }
    fun requestCall() {
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) callVm.startCall() else micLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    // 离开首页彻底释放麦克风：本页不像 VoiceCallScreen 那样有单独的返回动作，
    // 只能在页面销毁时兜底挂断。
    DisposableEffect(Unit) { onDispose { callVm.hangUp() } }

    // ── 文本流式输出的自动跟随 ──
    // 与 TextChatScreen 同一套逻辑（常驻协程 + 滚动锚点），不能写成
    // LaunchedEffect(末条消息内容)：那会每个 token 重启一次协程，滚动被反复取消。
    LaunchedEffect(Unit) {
        var stick = true
        var anchor: Pair<Int, Int>? = null
        var lastSize = 0

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
                total = info.totalItemsCount
            )
        }
            .distinctUntilChanged()
            .collect { snap ->
                if (snap.total == 0) return@collect
                val a = snap.anchorIndex to snap.anchorOffset
                val scrolled = a != anchor
                anchor = a
                if (snap.size != lastSize) {
                    stick = true
                    lastSize = snap.size
                } else if (scrolled) {
                    stick = chatList.isNearBottom()
                }
                if (stick) chatList.scrollToBottom()
            }
    }

    // 通话转写追加时滚到最新一条
    LaunchedEffect(cs.turns.size) {
        if (cs.turns.isNotEmpty()) runCatching { callList.animateScrollToItem(cs.turns.lastIndex) }
    }

    // 已经有会话内容时，智能体区折叠成一条横向滑动条，把高度让给对话区
    val hasConversation = ts.messages.isNotEmpty() || cs.inCall || cs.turns.isNotEmpty()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        HomeTopBar(onSettingsClick = onSettingsClick)

        if (hasConversation) {
            AgentStrip(
                agents = agents,
                onAgentClick = onAgentClick,
                modifier = Modifier.padding(top = 6.dp)
            )
        } else {
            AgentGrid(
                agents = agents,
                onAgentClick = onAgentClick,
                modifier = Modifier.padding(top = 12.dp)
            )
        }

        ModeTabs(
            mode = mode,
            onModeChange = { m ->
                if (m != mode) {
                    // 两种模式都要用麦克风 / LLM 推理同时只能服务一路，切换前先把另一路停下来
                    if (m == HomeChatMode.VOICE) {
                        focusManager.clearFocus()
                        if (ts.generating) textVm.stop()
                    } else {
                        callVm.hangUp()
                    }
                    mode = m
                }
            },
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp)
        )

        if (mode == HomeChatMode.TEXT) {
            ts.error?.let { err -> ErrorBanner(text = err, onDismiss = textVm::dismissError) }
            if (ts.preparing) {
                PreparingRow(ts.preparingText, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
            }
            if (ts.messages.isNotEmpty()) {
                ChatToolbar(
                    mode = ts.thinkMode,
                    onMode = textVm::setThinkMode,
                    canClear = !ts.generating,
                    onClear = textVm::clearChat,
                    onHistory = { showHistory = true },
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
                )
            }
        } else {
            cs.error?.let { err -> CallErrorBanner(text = err, onDismiss = callVm::dismissError) }
        }

        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            if (mode == HomeChatMode.TEXT) {
                if (ts.messages.isEmpty()) {
                    ChatEmptyState(onPick = { textVm.setInput(it) })
                } else {
                    LazyColumn(
                        state = chatList,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        items(ts.messages, key = { it.id }) { m ->
                            ChatBubble(msg = m, accent = AgentBlue)
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
                onSend = { focusManager.clearFocus(); textVm.send() },
                onStop = textVm::stop
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

    if (showHistory) {
        HistorySheet(
            sessions = sessions,
            currentId = textVm.currentSessionId,
            onOpen = textVm::openSession,
            onDelete = textVm::deleteSession,
            onNew = textVm::clearChat,
            onDismiss = { showHistory = false }
        )
    }
}

// ── 顶部应用栏 ──
@Composable
private fun HomeTopBar(onSettingsClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.statusBars)
            .padding(start = 20.dp, end = 12.dp, top = 8.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(34.dp)
                .clip(RoundedCornerShape(11.dp))
                .background(Brush.linearGradient(listOf(AgentPurple, lerp(AgentPurple, Color.White, 0.25f)))),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.AutoAwesome,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(19.dp)
            )
        }
        Spacer(modifier = Modifier.width(10.dp))
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
        IconButton(onClick = onSettingsClick) {
            Icon(
                imageVector = Icons.Default.Settings,
                contentDescription = "设置",
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

// ── 模式切换（分段控件） ──
@Composable
private fun ModeTabs(
    mode: HomeChatMode,
    onModeChange: (HomeChatMode) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(40.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f))
            .padding(4.dp)
    ) {
        HomeChatMode.values().forEach { m ->
            val selected = m == mode
            val accent = if (m == HomeChatMode.TEXT) AgentBlue else AgentRose
            val contentColor = if (selected) Color.White else MaterialTheme.colorScheme.onSurfaceVariant
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clip(RoundedCornerShape(16.dp))
                    .background(
                        if (selected) Brush.linearGradient(listOf(accent, lerp(accent, Color.White, 0.22f)))
                        else Brush.linearGradient(listOf(Color.Transparent, Color.Transparent))
                    )
                    .clickable { onModeChange(m) },
                contentAlignment = Alignment.Center
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = if (m == HomeChatMode.TEXT) Icons.AutoMirrored.Filled.Chat
                        else Icons.Default.Call,
                        contentDescription = null,
                        tint = contentColor,
                        modifier = Modifier.size(15.dp)
                    )
                    Text(
                        text = m.label,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium,
                        color = contentColor
                    )
                }
            }
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

// ── 对话进行中：智能体折叠成横向滑动条 ──
@Composable
private fun AgentStrip(
    agents: List<FunctionItem>,
    onAgentClick: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.padding(horizontal = 20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "更多智能体",
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onBackground,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "横向滑动查看全部",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Spacer(modifier = Modifier.height(5.dp))
        LazyRow(
            contentPadding = PaddingValues(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            items(agents, key = { it.id }) { item ->
                AgentChip(item = item, onClick = { onAgentClick(item.route) })
            }
        }
    }
}

@Composable
private fun AgentChip(item: FunctionItem, onClick: () -> Unit) {
    val enabled = !item.isComingSoon
    val alpha = if (enabled) 1f else 0.45f
    val shape = RoundedCornerShape(12.dp)
    Row(
        modifier = Modifier
            .clip(shape)
            .background(MaterialTheme.colorScheme.surface)
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.35f), shape)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 9.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        Icon(
            imageVector = item.icon,
            contentDescription = null,
            tint = item.accent.copy(alpha = alpha),
            modifier = Modifier.size(14.dp)
        )
        Text(
            text = item.title.removeSuffix("智能体"),
            style = MaterialTheme.typography.labelMedium.copy(fontSize = 11.sp),
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = alpha),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}
