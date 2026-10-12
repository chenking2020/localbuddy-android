package com.example.localbuddy

import android.app.Application
import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.scrollBy
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max

// ──────────────────────────────────────────────────────────────────
// 文本对话智能体
// ──────────────────────────────────────────────────────────────────

/**
 * 推理模式。两种模式走同一套权重，差别在「是否先规划」。
 *
 * - 快速模式：单次推理直接出正文，响应快，适合闲聊 / 简单问答。
 * - 精致模式：先生成一份极简执行方案，再按方案逐条展开作答（见 [TextChatViewModel.send]）。
 *   这是 WorkBuddy「先规划再执行」在端侧 0.6B 上的简要实现：模型太小，撑不起完整的
 *   多步工具编排，只取其骨架——先想清楚要点，再照着要点作答。
 */
enum class ThinkMode(val label: String, val hint: String) {
    FAST("快速模式", "直接作答，响应更快"),
    DEEP("精致模式", "先出执行方案，再逐条作答")
}

/** 一条对话气泡。 */
data class ChatMessage(
    val id: Long,
    val role: String,
    val text: String,
    /** 深度思考模式下的推理过程（模型 <think> 块内容，仅 assistant 有）。 */
    val thinking: String = "",
    val streaming: Boolean = false
)

data class TextChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val thinkMode: ThinkMode = ThinkMode.FAST,
    val input: String = "",
    val generating: Boolean = false,
    /** 模型首次加载（约 900MB）中：首次发送会等这一步。 */
    val preparing: Boolean = false,
    val preparingText: String = "",
    val docName: String? = null,
    val docChars: Int = 0,
    val error: String? = null
)

/** 一个历史会话：从「新建」到下一次新建之间的整段对话。 */
data class ChatSession(
    val id: String,
    /** 列表标题，取首条用户提问的前若干字。 */
    val title: String,
    val updatedAt: Long,
    val messages: List<ChatMessage>
)

/** 最多保留的历史会话数，避免 JSON 无限膨胀。 */
private const val MAX_SESSIONS = 50

/**
 * 历史会话持久化：filesDir 下的单个 JSON 文件。
 *
 * 与语音翻译 / 跨语言沟通智能体同一套路——读写都放 IO 线程，用 runCatching 兜底：
 * 记录损坏时退化成「没有历史」，绝不能把对话页拖崩。
 */
private class TextChatStore(private val context: Context) {
    private val file: File get() = File(context.filesDir, "text_chat_sessions.json")

    suspend fun load(): List<ChatSession> = withContext(Dispatchers.IO) {
        if (!file.exists()) return@withContext emptyList<ChatSession>()
        runCatching {
            val arr = JSONArray(file.readText())
            buildList {
                for (i in 0 until arr.length()) {
                    val o = arr.optJSONObject(i) ?: continue
                    val ja = o.optJSONArray("messages")
                    val msgs = buildList {
                        if (ja != null) for (j in 0 until ja.length()) {
                            val m = ja.optJSONObject(j) ?: continue
                            add(
                                ChatMessage(
                                    id = m.optLong("id"),
                                    role = m.optString("role"),
                                    text = m.optString("text"),
                                    thinking = m.optString("thinking")
                                )
                            )
                        }
                    }
                    add(ChatSession(o.optString("id"), o.optString("title"), o.optLong("updatedAt"), msgs))
                }
            }
        }.getOrElse { emptyList() }
    }

    suspend fun save(list: List<ChatSession>) = withContext(Dispatchers.IO) {
        runCatching {
            val arr = JSONArray()
            list.forEach { s ->
                val msgs = JSONArray()
                s.messages.forEach { m ->
                    msgs.put(JSONObject().apply {
                        put("id", m.id)
                        put("role", m.role)
                        put("text", m.text)
                        put("thinking", m.thinking)
                    })
                }
                arr.put(JSONObject().apply {
                    put("id", s.id)
                    put("title", s.title)
                    put("updatedAt", s.updatedAt)
                    put("messages", msgs)
                })
            }
            file.writeText(arr.toString())
        }
    }
}

private const val TAG = "TextChatVM"

/**
 * 端上 0.6B 的可用上下文只有 ~1k token，历史留太多会把当前问题挤出去。
 * 保留最近 3 轮问答（6 条），配合 [LlmEngine.chat] 的预算裁剪双重保险。
 */
private const val CHAT_HISTORY_TURNS = 6

/**
 * 文档摘录的 token 预算。总预算 1024，扣掉 system（~150）、历史（~300）、
 * 问题与生成，留给文档约 420（按「1 汉字 ≈ 1 token」的保守口径折算成字数）。
 */
private const val DOC_EXCERPT_TOKENS = 420

private const val MAX_NEW_FAST = 320

/** 精致模式 · 方案阶段：只产出要点，够用即可，别把 token 预算吃光。 */
private const val MAX_NEW_PLAN = 96

/** 精致模式 · 作答阶段：比快速模式更长，要按方案逐条展开。 */
private const val MAX_NEW_REFINE = 512

/**
 * 精致模式的方案阶段提示词：只要要点，不要解释。
 *
 * 端侧 0.6B 的长文本生成容易跑偏，所以这里把输出收紧到「2-4 条、每条 15 字内」的要点列表，
 * 既降低生成负担，也让第二步有明确的条目可以逐条覆盖。
 */
private const val PLAN_SYSTEM =
    "你是端侧离线助手的「方案规划」模块。针对用户的问题，先给出一份极简的执行方案。要求：\n" +
        "1. 只列 2-4 条要点，每条不超过 15 个字；\n" +
        "2. 每行一条，行首用「- 」，不要写编号以外的任何说明；\n" +
        "3. 不要复述用户的问题，不要客套，不要解释为什么要这么做；\n" +
        "4. 使用与用户提问相同的语言。"

class TextChatViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(TextChatUiState())
    val state: StateFlow<TextChatUiState> = _state.asStateFlow()

    /** 历史会话列表（倒序），供「历史」面板展示。 */
    private val _sessions = MutableStateFlow<List<ChatSession>>(emptyList())
    val sessions: StateFlow<List<ChatSession>> = _sessions.asStateFlow()

    private val store = TextChatStore(app)

    private var genJob: Job? = null
    private var docJob: Job? = null
    private var idSeq = 0L
    /** 已上传文档的全文（不进 UI state，只有字数对外）。 */
    private var docText: String = ""
    /** 送进模型的多轮历史（只含 user/assistant，system 每轮现拼）。 */
    private val history = ArrayList<ChatTurn>(CHAT_HISTORY_TURNS + 2)
    /** 当前会话在存储里的 id；null 表示还没产生过内容，首次发言时才分配。 */
    private var currentId: String? = null
    private var currentTitle = ""

    /** 当前会话 id，供历史面板高亮「正在看的这条」。 */
    val currentSessionId: String? get() = currentId

    init {
        // 恢复最近一次会话：否则每次进页面都像全新开始，之前的对话全丢了。
        viewModelScope.launch {
            val loaded = store.load()
            _sessions.value = loaded
            loaded.firstOrNull()?.takeIf { it.messages.isNotEmpty() }?.let { applySession(it) }
        }
        // 预加载 LLM：否则首次发送时 tokenizer/session 未就绪，generate 直接返回空串。幂等。
        viewModelScope.launch {
            _state.update { it.copy(preparing = true, preparingText = "加载对话模型中…") }
            LlmEngine.get(getApplication<Application>()).warmUp()
            _state.update { it.copy(preparing = false, preparingText = "") }
        }
    }

    /** 把一个已存会话载入为当前会话：恢复气泡、续上模型上下文、接过 id 继续追加。 */
    private fun applySession(s: ChatSession) {
        currentId = s.id
        currentTitle = s.title
        idSeq = (s.messages.maxOfOrNull { it.id } ?: 0L) + 1
        history.clear()
        // 思考过程不进模型上文（否则后续轮次会被推理链带偏），只取正文
        s.messages.filter { it.text.isNotBlank() }.takeLast(CHAT_HISTORY_TURNS)
            .forEach { history.add(ChatTurn(it.role, it.text)) }
        _state.update { it.copy(messages = s.messages) }
    }

    /**
     * 把当前气泡落盘（异步）。空内容不写，避免给「新建」留下空会话条目。
     *
     * 每轮结束（含被「停止」截断）都调一次，返回上一页或杀进程后回来还在。
     */
    private fun persistCurrent() {
        val msgs = _state.value.messages
            .filter { it.text.isNotBlank() || it.thinking.isNotBlank() }
            .map { if (it.streaming) it.copy(streaming = false) else it }
        if (msgs.isEmpty()) return

        val id = currentId ?: ("c" + System.currentTimeMillis()).also { currentId = it }
        if (currentTitle.isBlank()) {
            currentTitle = msgs.firstOrNull { it.role == ROLE_USER }?.text?.trim().orEmpty()
                .replace('\n', ' ').take(30).ifBlank { "新会话" }
        }
        val session = ChatSession(id, currentTitle, System.currentTimeMillis(), msgs)
        viewModelScope.launch(Dispatchers.IO) {
            val list = _sessions.value.toMutableList()
            val idx = list.indexOfFirst { it.id == id }
            if (idx >= 0) list[idx] = session else list.add(0, session)
            val trimmed = list.sortedByDescending { it.updatedAt }.take(MAX_SESSIONS)
            _sessions.value = trimmed
            store.save(trimmed)
        }
    }

    /** 打开某个历史会话，继续在同一个会话里聊。 */
    fun openSession(id: String) {
        val s = _sessions.value.firstOrNull { it.id == id } ?: return
        genJob?.cancel(); genJob = null
        applySession(s)
        _state.update { it.copy(generating = false, error = null) }
    }

    /** 删除某个历史会话；删的正是当前这条时顺手回到空界面。 */
    fun deleteSession(id: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val list = _sessions.value.filterNot { it.id == id }
            _sessions.value = list
            store.save(list)
            if (currentId == id) {
                currentId = null
                currentTitle = ""
                history.clear()
                genJob?.cancel(); genJob = null
                _state.update { it.copy(messages = emptyList(), generating = false) }
            }
        }
    }

    fun setInput(v: String) = _state.update { it.copy(input = v) }
    fun setThinkMode(m: ThinkMode) { if (!_state.value.generating) _state.update { it.copy(thinkMode = m) } }
    fun dismissError() = _state.update { it.copy(error = null) }

    /**
     * 新建会话：清空气泡与模型上文（保留已上传文档，方便连续追问）。
     * 当前会话已写进历史，这里只是断开、让下次发言另开一条。
     */
    fun clearChat() {
        genJob?.cancel(); genJob = null
        history.clear()
        currentId = null
        currentTitle = ""
        _state.update { it.copy(messages = emptyList(), generating = false) }
    }

    fun stop() {
        if (!_state.value.generating) return
        genJob?.cancel(); genJob = null
        _state.update { s ->
            s.copy(generating = false,
                messages = s.messages.map { if (it.streaming) it.copy(streaming = false) else it })
        }
    }

    // ── 文档 ──

    fun attachDoc(uri: Uri, name: String, mime: String?) {
        docJob?.cancel()
        _state.update { it.copy(preparing = true, preparingText = "读取文档中…", error = null) }
        docJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val ctx = getApplication<Application>()
                // 与文档识别翻译智能体同一口径：OCR 与 LLM「加载 + 推理」并发在部分设备原生崩溃，
                // 先等 LLM 加载完再动 OCR。
                LlmEngine.get(ctx).warmUp()
                val text = DocTextReader.read(ctx, uri, mime, name)
                if (text.isBlank()) throw IllegalStateException("文档未提取到文字，请换一份试试")
                docText = text
                _state.update {
                    it.copy(docName = name, docChars = text.length,
                        preparing = false, preparingText = "")
                }
            } catch (e: Throwable) {
                Log.e(TAG, "attach doc failed", e)
                _state.update {
                    it.copy(preparing = false, preparingText = "", error = "文档读取失败：${e.message ?: "未知"}")
                }
            }
        }
    }

    fun removeDoc() {
        docJob?.cancel()
        docText = ""
        _state.update { it.copy(docName = null, docChars = 0) }
    }

    // ── 发送 ──

    fun send() {
        val q = _state.value.input.trim()
        if (q.isBlank() || _state.value.generating) return
        val refined = _state.value.thinkMode == ThinkMode.DEEP
        val llm = LlmEngine.get(getApplication<Application>())

        val userMsg = ChatMessage(idSeq++, ROLE_USER, q)
        val ansId = idSeq++
        _state.update {
            it.copy(
                input = "",
                generating = true,
                error = null,
                messages = it.messages + userMsg +
                    ChatMessage(ansId, ROLE_ASSISTANT, "", streaming = true)
            )
        }

        val doc = docText
        val docName = _state.value.docName
        genJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                if (!llm.isLoaded) {
                    _state.update { it.copy(preparing = true, preparingText = "加载对话模型中…") }
                    llm.warmUp()
                    _state.update { it.copy(preparing = false, preparingText = "") }
                }
                // 文档问答：按问题从全文里检索最相关的句子，只把摘录塞进 prompt
                val userContent = buildUserContent(q, doc, docName)

                // ── 精致模式：第一段先出执行方案，第二段按方案作答 ──
                // 方案写进 thinking 字段，复用气泡里那块可折叠区域展示（生成中展开、正文一出自动收起）。
                val plan = if (refined) {
                    val raw = llm.generate(
                        user = userContent,
                        system = PLAN_SYSTEM,
                        maxNew = MAX_NEW_PLAN,
                        enableThinking = false,
                        onPartial = { t -> patch(ansId) { it.copy(thinking = t) } }
                    )
                    val p = raw.trim()
                    patch(ansId) { it.copy(thinking = p) }
                    p
                } else ""

                val turns = buildList {
                    add(ChatTurn(ROLE_SYSTEM, buildSystem(docName, doc.length, plan)))
                    history.forEach { add(it) }
                    add(ChatTurn(ROLE_USER, userContent))
                }
                val answer = llm.chat(
                    messages = turns,
                    maxNew = if (refined) MAX_NEW_REFINE else MAX_NEW_FAST,
                    enableThinking = false,
                    onPartial = { p -> patch(ansId) { it.copy(text = p) } }
                )
                // 中途被「停止」：保留已生成的部分内容，不要拿空串覆盖掉它
                if (!currentCoroutineContext().isActive) {
                    patch(ansId) { it.copy(streaming = false) }
                    rememberPartial(ansId, q)
                    persistCurrent()
                    return@launch
                }
                val final = answer.trim().ifBlank { "（模型未生成内容，请换个说法重试）" }
                patch(ansId) { it.copy(text = final, streaming = false) }
                // 历史只存正文（思考过程不进上文，否则后续轮次会被推理链带偏）
                history.add(ChatTurn(ROLE_USER, q))
                history.add(ChatTurn(ROLE_ASSISTANT, final))
                while (history.size > CHAT_HISTORY_TURNS) history.removeAt(0)
                persistCurrent()
            } catch (e: kotlinx.coroutines.CancellationException) {
                patch(ansId) { it.copy(streaming = false) }
                persistCurrent()
            } catch (e: Throwable) {
                Log.e(TAG, "chat failed", e)
                patch(ansId) { it.copy(text = "", streaming = false) }
                _state.update { it.copy(error = "生成失败：${e.message ?: "未知"}") }
            } finally {
                _state.update { it.copy(generating = false, preparing = false, preparingText = "") }
            }
        }
    }

    /** 停止后把已生成的部分内容补进历史，保证下一轮能接得上上下文。 */
    private fun rememberPartial(ansId: Long, question: String) {
        val partial = _state.value.messages.firstOrNull { it.id == ansId }?.text.orEmpty().trim()
        if (partial.isBlank()) return
        history.add(ChatTurn(ROLE_USER, question))
        history.add(ChatTurn(ROLE_ASSISTANT, partial))
        while (history.size > CHAT_HISTORY_TURNS) history.removeAt(0)
    }

    private fun patch(id: Long, f: (ChatMessage) -> ChatMessage) {
        _state.update { s ->
            s.copy(messages = s.messages.map { if (it.id == id) f(it) else it })
        }
    }

    /**
     * @param plan 精致模式生成的执行方案；非空时要求模型按方案逐条展开。
     *             方案为空（生成失败 / 模型不配合）时自动退化成普通作答，不至于卡住。
     */
    private fun buildSystem(docName: String?, docLen: Int, plan: String? = null): String {
        val core = "你是「端侧离线工作搭子」内置的文本对话助手，运行在用户手机本地、全程离线。要求：\n" +
            "1. 直接回答用户的问题，不要复述问题，不要加“好的”“以下是”之类客套前缀；\n" +
            "2. 回答简洁清晰，要点多时用列表分条，不要写长篇套话；\n" +
            "3. 使用与用户提问相同的语言回答；\n" +
            "4. 不确定的信息要说明不确定，不要编造事实；\n"
        val docPart = if (docName != null)
            "5. 用户已上传文档《$docName》（共 $docLen 字）。" +
                "回答与该文档相关的问题时，只依据用户消息里给出的文档摘录作答；" +
                "摘录中没有的信息，明确说明文档未提及，不要臆测。\n"
        else ""
        val planPart = if (!plan.isNullOrBlank())
            "本次回答请严格按下面这份执行方案逐条展开，不要遗漏任何一条：\n$plan\n"
        else ""
        return core + docPart + planPart
    }

    private suspend fun buildUserContent(q: String, doc: String, docName: String?): String =
        withContext(Dispatchers.Default) {
            if (doc.isBlank() || docName == null) return@withContext q
            val excerpt = TextExtractor.retrieve(
                doc, q, DOC_EXCERPT_TOKENS,
                tokenCounter = { it.length },
                maxSentences = 10
            )
            if (excerpt.isEmpty()) return@withContext q
            val body = excerpt.mapIndexed { i, s -> "(${i + 1}) $s" }.joinToString("\n")
            "以下是文档《$docName》中与本次问题最相关的摘录：\n$body\n\n" +
                "请依据上述摘录回答下面的问题；摘录之外的信息若文档未提及，请直接说明。\n" +
                "问题：$q"
        }
}

// ──────────────────────────────────────────────────────────────────
// UI
// ──────────────────────────────────────────────────────────────────

@Composable
fun TextChatScreen(onBack: () -> Unit, vm: TextChatViewModel = viewModel()) {
    val s by vm.state.collectAsState()
    val sessions by vm.sessions.collectAsState()
    val context = LocalContext.current
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    var showHistory by remember { mutableStateOf(false) }

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
            vm.attachDoc(uri, name, mime)
        }
    }
    val pickMimes = arrayOf(
        "text/plain",
        "application/pdf",
        "image/*",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    )

    // ── 自动滚动到底部 ──
    // 不能用 LaunchedEffect(最后一条消息的内容) 这种写法：它每来一个 token 就重启一次协程，
    // 而 LazyListState 的滚动要过 MutatorMutex —— 协程在等锁 / 等首帧布局时被取消就是一次
    // 空滚。token 间隔只有几十毫秒，绝大多数滚动请求都被取消在半途，表现就是"完全不滚动"。
    // 改成常驻协程 + snapshotFlow 串行处理，滚动不再被中途取消。
    //
    // 另外不能每次都拿"距底部距离"判断是否跟随：内容在底部增长时，末尾气泡会被挤出可见区，
    // 距离判据直接判成"不在底部" -> 不滚 -> 永远不在底部 -> 死锁。
    // 改用**滚动锚点**判断：锚点只在真正滚动时才变，内容增长不会动它，
    // 于是能准确区分"用户翻历史"和"内容变长"。
    LaunchedEffect(Unit) {
        var stick = true                  // 是否跟随到底部
        var anchor: Pair<Int, Int>? = null
        var lastSize = 0

        snapshotFlow {
            val m = s.messages
            val last = m.lastOrNull()
            val info = listState.layoutInfo
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
                if (snap.total == 0) return@collect         // 还没完成首次布局
                val a = snap.anchorIndex to snap.anchorOffset
                val scrolled = a != anchor
                anchor = a
                if (snap.size != lastSize) {
                    // 自己刚发的消息：无条件重新跟随。
                    // 必须优先于下面的锚点判断——新消息插进来时锚点也会变，
                    // 若先按锚点判定，此时还没滚过去，会被误判成"不在底部"而从此不滚。
                    stick = true
                    lastSize = snap.size
                } else if (scrolled) {
                    // 真的滚动过（用户翻历史 / 上一次跟滚）→ 按新位置重新判定
                    stick = listState.isNearBottom()
                }
                if (stick) listState.scrollToBottom()
            }
    }

    // 不套 Scaffold 的 innerPadding：顶部由 AgentHeader 的 windowInsetsPadding(statusBars) 处理，
    // 底部由输入栏自己算 max(导航栏, 输入法) —— 两层都加会重复留白，而少了任何一层又会被遮挡。
    Scaffold(
        containerColor = MaterialTheme.colorScheme.background
    ) { _ ->
        Column(modifier = Modifier.fillMaxSize()) {
            AgentHeader(
                accent = AgentBlue,
                icon = Icons.AutoMirrored.Filled.Chat,
                title = "文本对话智能体",
                subtitle = "文本 · 问答 · 深度思考 · 全程离线",
                onBack = onBack
            )

            ChatToolbar(
                canClear = s.messages.isNotEmpty() && !s.generating,
                onNew = vm::clearChat,
                onClear = vm::clearChat,
                onHistory = { showHistory = true },
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)
            )

            s.error?.let { err -> ErrorBanner(text = err, onDismiss = vm::dismissError) }

            // 模型加载 / 文档解析的进度放在会话区之上，首屏空态时也能看到
            if (s.preparing) {
                PreparingRow(s.preparingText, modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
            }

            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                if (s.messages.isEmpty()) {
                    ChatEmptyState(onPick = { vm.setInput(it) })
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 10.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        items(s.messages, key = { it.id }) { m ->
                            ChatBubble(msg = m, accent = AgentBlue)
                        }
                        // 不要在末尾再放 item { Spacer }：那样最后一项就不是消息气泡了，
                        // "距底部还有多远"就无从测量，自动滚动会失准。
                    }
                }
            }

            ChatInputBar(
                value = s.input,
                onValue = vm::setInput,
                generating = s.generating,
                docName = s.docName,
                docChars = s.docChars,
                onRemoveDoc = vm::removeDoc,
                onPickDoc = { runCatching { pickLauncher.launch(pickMimes) } },
                onSend = { focusManager.clearFocus(); vm.send() },
                onStop = vm::stop,
                thinkMode = s.thinkMode,
                onThinkMode = vm::setThinkMode
            )
        }
    }

    if (showHistory) {
        HistorySheet(
            sessions = sessions,
            currentId = vm.currentSessionId,
            onOpen = vm::openSession,
            onDelete = vm::deleteSession,
            onNew = vm::clearChat,
            onDismiss = { showHistory = false }
        )
    }
}

/**
 * 滚到底部。
 *
 * 主手段是 `scrollBy(极大值)`：超出可滚距离的部分会被 LazyListState 直接丢弃，
 * 天然钳到 maxScroll，一步到位。
 *
 * 不能用 `scrollToItem(lastIndex)` —— 它对齐的是该项**顶部**，而最后一条消息常常比视口还高
 * （深度思考 + 长正文），那样会先往上跳到消息开头、再由下面的循环补回底部，
 * 流式输出每 token 来一次就是疯狂闪烁。
 *
 * 后面的循环是兜底：内容每帧都在变长，万一一次没到底就再按 layoutInfo 补几下。
 */
internal suspend fun LazyListState.scrollToBottom() {
    if (layoutInfo.totalItemsCount == 0) {
        // 首次布局完成前 LazyList 认为不可滚动，此时滚动是空操作——先等首帧
        snapshotFlow { layoutInfo.totalItemsCount }.first { it > 0 }
    }
    scrollBy(1_000_000f)
    repeat(6) {
        if (!canScrollForward) return
        val info = layoutInfo
        val last = info.visibleItemsInfo.lastOrNull() ?: return
        val gap = last.offset + last.size - info.viewportEndOffset
        if (gap <= 0) return
        scrollBy(gap.toFloat())
    }
}

/**
 * 当前是否贴在底部。
 *
 * 只在**真正发生滚动**之后才用它重新判定（见上方的锚点判断），
 * 所以不用担心"内容边增长边测量"导致的误判。
 */
internal fun LazyListState.isNearBottom(thresholdPx: Int = 120): Boolean {
    if (!canScrollForward) return true
    val info = layoutInfo
    val lastVisible = info.visibleItemsInfo.lastOrNull() ?: return true
    if (lastVisible.index < info.totalItemsCount - 1) return false
    return lastVisible.offset + lastVisible.size - info.viewportEndOffset <= thresholdPx
}

/** 一次"要不要跟滚"的判断快照：消息内容 + 列表滚动锚点。 */
internal data class BottomSnap(
    val size: Int,
    val textLen: Int,
    val thinkLen: Int,
    val anchorIndex: Int,
    val anchorOffset: Int,
    val total: Int
)

/** 推理模式切换 + 清空会话。做成一条与背景同色的圆角工具栏，弱化到不抢消息区的注意力。 */
@Composable
/**
 * 会话操作栏。模式切换已挪到输入区上方（点击输入框后展开），这里只留会话本身的操作。
 * @param canClear 当前会话是否有内容；空会话时「新建 / 清空」置灰，但「历史」始终可用。
 */
internal fun ChatToolbar(
    canClear: Boolean,
    onNew: () -> Unit,
    onClear: () -> Unit,
    onHistory: () -> Unit,
    modifier: Modifier = Modifier
) {
    Row(modifier = modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Spacer(Modifier.weight(1f))
        ToolbarTextButton(
            icon = Icons.Default.Add,
            label = "新建对话",
            enabled = canClear,
            onClick = onNew
        )
        ToolbarTextButton(
            icon = Icons.Default.History,
            label = "历史",
            onClick = onHistory
        )
        ToolbarTextButton(
            icon = Icons.Default.Close,
            label = "清空会话",
            enabled = canClear,
            onClick = onClear
        )
    }
}

/** 工具栏里「图标 + 文字」的小按钮，禁用时整体变淡且不响应点击。 */
@Composable
private fun ToolbarTextButton(
    icon: ImageVector,
    label: String,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    val tint = if (enabled) MaterialTheme.colorScheme.onSurfaceVariant
    else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f)
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Icon(icon, contentDescription = null, tint = tint, modifier = Modifier.size(15.dp))
        Text(label, style = MaterialTheme.typography.labelSmall, color = tint)
    }
}

/**
 * 历史会话面板。
 *
 * 之前文本对话的内容只活在 ViewModel 里，返回上一页 / 杀进程就没了，也没有任何入口能回看。
 * 这里给出列表：点一条即载入并继续在同一个会话里聊，右侧可删除，顶部可新建。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun HistorySheet(
    sessions: List<ChatSession>,
    currentId: String?,
    onOpen: (String) -> Unit,
    onDelete: (String) -> Unit,
    onNew: () -> Unit,
    onDismiss: () -> Unit
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 2.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("历史会话", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.weight(1f))
            TextButton(onClick = { onNew(); onDismiss() }) { Text("新建") }
        }
        if (sessions.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxWidth().padding(vertical = 44.dp),
                contentAlignment = Alignment.Center
            ) {
                Text("还没有历史会话", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        } else {
            LazyColumn(modifier = Modifier.fillMaxWidth()) {
                items(sessions, key = { it.id }) { s ->
                    HistoryRow(
                        session = s,
                        active = s.id == currentId,
                        onOpen = { onOpen(s.id); onDismiss() },
                        onDelete = { onDelete(s.id) }
                    )
                }
            }
        }
        Spacer(Modifier.height(20.dp))
    }
}

@Composable
private fun HistoryRow(
    session: ChatSession,
    active: Boolean,
    onOpen: () -> Unit,
    onDelete: () -> Unit
) {
    val bg = if (active) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
    else Color.Transparent
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 3.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(bg)
            .clickable(onClick = onOpen)
            .padding(start = 12.dp, end = 4.dp, top = 8.dp, bottom = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                session.title.ifBlank { "新会话" },
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Spacer(Modifier.height(2.dp))
            Text(
                "${fmtTime(session.updatedAt)} · ${session.messages.size} 条",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        IconButton(onClick = onDelete) {
            Icon(Icons.Default.Delete, "删除",
                tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(18.dp))
        }
    }
}

private fun fmtTime(ms: Long): String =
    SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(ms))

/** 单条气泡。用户：主题色实心 + 白字；助手：浅底 + 可折叠的思考过程。 */
@Composable
internal fun ChatBubble(
    msg: ChatMessage,
    accent: Color,
    /** 思考/方案区进行中与完成时的标题文案。 */
    planDoing: String = "深度思考中…",
    planDone: String = "已深度思考"
) {
    val isUser = msg.role == ROLE_USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Top,
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            AssistantAvatar(accent)
            Spacer(Modifier.width(8.dp))
        }
        // 气泡最宽占 84%：留出对手侧的一列空白，长对话里左右分区才清楚
        Column(modifier = Modifier.fillMaxWidth(0.84f), horizontalAlignment = if (isUser) Alignment.End else Alignment.Start) {
            if (msg.thinking.isNotBlank()) {
                ThinkBlock(msg = msg, accent = accent, doingLabel = planDoing, doneLabel = planDone)
                Spacer(Modifier.height(6.dp))
            }
            if (msg.text.isNotBlank()) {
                UserOrAnswerBubble(msg = msg, accent = accent, isUser = isUser)
            } else if (msg.streaming && msg.thinking.isBlank()) {
                // 还没有任何内容（不思考模式的首 token 前）：打字点，别留一个空气泡
                BubbleSurface(isUser = false, accent = accent) { TypingDots(accent) }
            }
        }
    }
}

@Composable
internal fun AssistantAvatar(accent: Color) {
    Box(
        modifier = Modifier
            .size(32.dp)
            .clip(RoundedCornerShape(11.dp))
            .background(accent.copy(alpha = 0.13f)),
        contentAlignment = Alignment.Center
    ) {
        Icon(Icons.Default.SmartToy, null, tint = accent, modifier = Modifier.size(18.dp))
    }
}

@Composable
internal fun UserOrAnswerBubble(msg: ChatMessage, accent: Color, isUser: Boolean) {
    if (isUser) {
        // 用户气泡：主题色实心 + **白字**（此前沿用 Markdown/正文的默认深色，蓝底上看不清）。
        // 用户输入是纯文本，直接 Text 渲染即可，不需要 Markdown。
        Box(
            modifier = Modifier
                .clip(ChatBubbleShape(isUser = true))
                .background(
                    Brush.linearGradient(
                        listOf(accent, lerp(accent, Color.White, 0.18f))
                    )
                )
                .padding(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Text(
                msg.text,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White,
                lineHeight = 21.sp
            )
        }
        return
    }
    BubbleSurface(isUser = false, accent = accent) {
        // 助手回复可选中复制；只包正文，不包思考区（思考区本身要响应点击折叠）
        SelectionContainer {
            // 流式过程中也按 Markdown 渲染：端侧模型输出慢，生成动辄十几秒，
            // 若只在结束后才渲染，整个过程就只能看着带 # 和 ** 的源码。
            // 解析器已容忍流式下的残缺语法：未闭合的 ``` 整段按代码块渲染，
            // 未配对的 ** / ` 原样显示（与常见客户端一致，闭合后即变成对应样式）。
            MarkdownText(msg.text, modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
internal fun BubbleSurface(
    isUser: Boolean,
    accent: Color,
    content: @Composable () -> Unit
) {
    Box(
        modifier = Modifier
            .clip(ChatBubbleShape(isUser = isUser))
            .background(MaterialTheme.colorScheme.surface)
            // 浅色主题下 surface 与 background 反差很小，加一道描边把气泡边界勾出来
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f),
                ChatBubbleShape(isUser = isUser))
            .padding(horizontal = 14.dp, vertical = 10.dp)
    ) {
        content()
    }
}

/**
 * 气泡圆角：靠对话一侧的那个角收窄成"尾巴"，一眼能分出谁说的。
 * 四角 18dp，尾巴角 5dp。
 */
internal fun ChatBubbleShape(isUser: Boolean) = RoundedCornerShape(
    topStart = 18.dp,
    topEnd = 18.dp,
    bottomStart = if (isUser) 18.dp else 5.dp,
    bottomEnd = if (isUser) 5.dp else 18.dp
)

/**
 * 深度思考的推理过程：**可折叠**。
 *
 * 展开策略：思考中保持展开（能看到模型在想什么），一旦正文开始输出或生成结束就自动收起，
 * 让答案回到视觉中心；之后用户可点击标题手动展开/收起。
 */
@Composable
internal fun ThinkBlock(
    msg: ChatMessage,
    accent: Color,
    doingLabel: String = "深度思考中…",
    doneLabel: String = "已深度思考"
) {
    var expanded by remember(msg.id) { mutableStateOf(true) }
    LaunchedEffect(msg.streaming, msg.text.isNotBlank()) {
        if (!msg.streaming || msg.text.isNotBlank()) expanded = false
    }
    val thinking = msg.streaming && msg.text.isBlank()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(accent.copy(alpha = 0.07f))
            .animateContentSize()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { expanded = !expanded }
                .padding(horizontal = 12.dp, vertical = 9.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                Icons.Default.Psychology, null,
                tint = accent, modifier = Modifier.size(15.dp)
            )
            Text(
                if (thinking) doingLabel else doneLabel,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.SemiBold,
                color = accent,
                modifier = Modifier.weight(1f, fill = false)
            )
            if (thinking) {
                TypingDots(accent, dotSize = 4.dp)
            } else {
                Text(
                    if (expanded) "收起" else "展开",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Icon(
                Icons.Default.ExpandMore, null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .size(16.dp)
                    .rotate(if (expanded) 180f else 0f)
            )
        }
        if (expanded) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(IntrinsicSize.Min)
            ) {
                // 左侧色条：把思考区和正文区在视觉上彻底分开
                Box(
                    Modifier
                        .width(3.dp)
                        .fillMaxHeight()
                        .padding(vertical = 2.dp)
                        .clip(CircleShape)
                        .background(accent.copy(alpha = 0.5f))
                        .align(Alignment.CenterStart)
                )
                Text(
                    msg.thinking,
                    modifier = Modifier.padding(start = 15.dp, end = 12.dp, top = 2.dp, bottom = 11.dp),
                    style = MaterialTheme.typography.bodySmall,
                    fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = 19.sp
                )
            }
        }
    }
}

@Composable
internal fun TypingDots(accent: Color, dotSize: androidx.compose.ui.unit.Dp = 6.dp) {
    val t = rememberInfiniteTransition(label = "typing")
    Row(
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(3) { i ->
            val a by t.animateFloat(
                initialValue = 0.25f, targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    tween(600, delayMillis = i * 150, easing = LinearEasing),
                    RepeatMode.Reverse
                ),
                label = "dot$i"
            )
            Box(
                Modifier
                    .size(dotSize)
                    .alpha(a)
                    .clip(CircleShape)
                    .background(accent)
            )
        }
    }
}

@Composable
internal fun PreparingRow(text: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        CircularProgressIndicator(modifier = Modifier.size(13.dp), strokeWidth = 2.dp, color = AgentBlue)
        Text(text.ifBlank { "处理中…" }, style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

/** 空态：品牌图标 + 引导语 + 2×2 的示例问题（点了填进输入框，不直接发送）。 */
@Composable
internal fun ChatEmptyState(onPick: (String) -> Unit) {
    val samples = listOf(
        "用三句话解释什么是大语言模型" to "概念解释",
        "帮我写一封请假邮件，语气正式一点" to "文案写作",
        "如何高效地做一份会议纪要？" to "方法建议",
        "把「罗列要点」扩写成一段通顺的文字" to "文本润色"
    )
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(RoundedCornerShape(24.dp))
                .background(
                    Brush.linearGradient(
                        listOf(AgentBlue.copy(alpha = 0.16f), AgentBlue.copy(alpha = 0.06f))
                    )
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.AutoMirrored.Filled.Chat, null, tint = AgentBlue, modifier = Modifier.size(34.dp))
        }
        Spacer(Modifier.height(16.dp))
        Text("开始一段离线对话", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        Text(
            "所有推理都在本机完成，不联网、不上传。切到「深度思考」会让模型先展开推理再作答；" +
                "点输入框左侧的回形针可以上传文档，之后提问会先在你上传的内容里检索。",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            lineHeight = 20.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
        Spacer(Modifier.height(24.dp))
        samples.chunked(2).forEach { row ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                row.forEach { (q, tag) ->
                    SuggestionCard(
                        text = q, tag = tag,
                        modifier = Modifier.weight(1f),
                        onClick = { onPick(q) }
                    )
                }
            }
            Spacer(Modifier.height(10.dp))
        }
    }
}

@Composable
internal fun SuggestionCard(
    text: String,
    tag: String,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(MaterialTheme.colorScheme.surface)
            .clickable(onClick = onClick)
            .padding(horizontal = 13.dp, vertical = 12.dp)
    ) {
        Text(tag, style = MaterialTheme.typography.labelSmall, color = AgentBlue,
            fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(4.dp))
        Text(text, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface, lineHeight = 19.sp,
            maxLines = 3, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
    }
}

/**
 * 底部输入区：文档芯片（有文档时）+ 胶囊输入框 + 发送/停止。
 *
 * 底部内边距取 **max(导航栏, 输入法)**：两个都加会被顶高一大截，只加一个又会在另一场景下
 * 被遮住；同时 manifest 里配了 `adjustResize`，输入法高度才能正确进入 WindowInsets.ime。
 */
@Composable
/**
 * 豆包式输入卡片：
 * - 未聚焦时是紧凑单行（输入框 + 语音 + 发送）；
 * - 点击聚焦后展开成大卡片：上面是多行输入区（高度变大），下面一行操作按钮——
 *   语音（按住说话）、⚡快速/精致选择、+（上传文档）、圆形发送键。
 */
internal fun ChatInputBar(
    value: String,
    onValue: (String) -> Unit,
    generating: Boolean,
    docName: String?,
    docChars: Int,
    onRemoveDoc: () -> Unit,
    onPickDoc: () -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    /** 快速 / 精致模式：非空时在展开态的操作行里显示选择器。 */
    thinkMode: ThinkMode? = null,
    onThinkMode: ((ThinkMode) -> Unit)? = null,
    /** 输入框焦点变化回调（展开/收起也由这里驱动）。 */
    onFocusChange: (Boolean) -> Unit = {},
    /** 是否显示语音输入按钮（按住说话、松开发送）。 */
    showMic: Boolean = false,
    /** 语音按钮的手势 modifier：首页用长按手势，不能是普通 clickable。 */
    micModifier: Modifier = Modifier,
    /** 正在录音：按钮转为按下态。 */
    micActive: Boolean = false
) {
    val navBar = WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding()
    val ime = WindowInsets.ime.asPaddingValues().calculateBottomPadding()
    val bottomPad = max(navBar.value, ime.value).dp

    var focused by remember { mutableStateOf(false) }
    var modeMenuOpen by remember { mutableStateOf(false) }
    // 聚焦后展开：输入区变高、操作行换到下面一行
    val expanded = focused
    val inputMinHeight by animateDpAsState(
        targetValue = if (expanded) 116.dp else 44.dp,
        animationSpec = tween(durationMillis = 180),
        label = "inputMinHeight"
    )

    val canSend = value.isNotBlank() && !generating

    Surface(tonalElevation = 3.dp, shadowElevation = 8.dp, color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 14.dp, end = 14.dp, top = 10.dp)
                .padding(bottom = bottomPad + 12.dp)
        ) {
            // ── 输入卡片：大圆角、描边 ──
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(28.dp))
                    .background(MaterialTheme.colorScheme.surface)
                    .border(
                        1.dp,
                        MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f),
                        RoundedCornerShape(28.dp)
                    )
            ) {
                // 已附加的文档：在卡片内顶部回显
                if (docName != null) {
                    Row(
                        modifier = Modifier
                            .padding(start = 14.dp, end = 14.dp, top = 12.dp)
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(AgentBlue.copy(alpha = 0.09f))
                            .padding(horizontal = 11.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Description, null, tint = AgentBlue, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(7.dp))
                        Text(
                            docName,
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f, fill = false)
                        )
                        Text(" · ${docChars} 字", style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.width(4.dp))
                        Box(
                            modifier = Modifier
                                .size(22.dp)
                                .clip(CircleShape)
                                .clickable(onClick = onRemoveDoc),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Close, "移除文档",
                                tint = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.size(14.dp))
                        }
                    }
                }

                // ── 输入区 ──
                BasicTextField(
                    value = value,
                    onValueChange = onValue,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = inputMinHeight)
                        .onFocusChanged {
                            focused = it.isFocused
                            onFocusChange(it.isFocused)
                        },
                    textStyle = MaterialTheme.typography.bodyLarge.copy(
                        color = MaterialTheme.colorScheme.onSurface
                    ),
                    cursorBrush = SolidColor(AgentBlue),
                    maxLines = 8,
                    keyboardOptions = KeyboardOptions(
                        capitalization = KeyboardCapitalization.Sentences,
                        imeAction = ImeAction.Send
                    ),
                    keyboardActions = KeyboardActions(
                        onSend = { if (canSend) onSend() }
                    ),
                    decorationBox = { inner ->
                        Box(
                            modifier = Modifier.padding(horizontal = 18.dp, vertical = 12.dp),
                            contentAlignment = if (expanded) Alignment.TopStart else Alignment.CenterStart
                        ) {
                            if (value.isEmpty()) {
                                Text(
                                    if (docName != null) "针对文档提问…" else "发消息…",
                                    style = MaterialTheme.typography.bodyLarge,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            inner()
                        }
                    }
                )

                // ── 操作行 ──
                Row(
                    modifier = Modifier.padding(start = 10.dp, end = 12.dp, bottom = 10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // 语音输入：按住说话、松开发送
                    if (showMic) {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .scale(if (micActive) 1.14f else 1f)
                                .clip(CircleShape)
                                .background(
                                    if (micActive) Color(0xFFDC2626) else Color.Transparent
                                )
                                .then(micModifier),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Mic,
                                contentDescription = "按住说话",
                                tint = if (micActive) Color.White
                                else MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.size(21.dp)
                            )
                        }
                    }

                    // 快速 / 精致模式：⚡快速 ∨，点击弹出选择
                    if (thinkMode != null && onThinkMode != null) {
                        Box {
                            Row(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(18.dp))
                                    .clickable { modeMenuOpen = true }
                                    .padding(horizontal = 10.dp, vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    imageVector = if (thinkMode == ThinkMode.DEEP) Icons.Default.Psychology
                                    else Icons.Default.Bolt,
                                    contentDescription = null,
                                    tint = if (thinkMode == ThinkMode.DEEP) AgentPurple else AgentBlue,
                                    modifier = Modifier.size(16.dp)
                                )
                                Text(
                                    text = thinkMode.label,
                                    style = MaterialTheme.typography.labelLarge,
                                    fontWeight = FontWeight.SemiBold,
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                                Icon(
                                    imageVector = Icons.Default.ArrowDropDown,
                                    contentDescription = "切换模式",
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                            DropdownMenu(
                                expanded = modeMenuOpen,
                                onDismissRequest = { modeMenuOpen = false }
                            ) {
                                DropdownMenuItem(
                                    text = { Text("快速 · 直接回答") },
                                    leadingIcon = {
                                        Icon(Icons.Default.Bolt, null, tint = AgentBlue, modifier = Modifier.size(18.dp))
                                    },
                                    onClick = {
                                        onThinkMode(ThinkMode.FAST)
                                        modeMenuOpen = false
                                    }
                                )
                                DropdownMenuItem(
                                    text = { Text("精致 · 深度思考") },
                                    leadingIcon = {
                                        Icon(Icons.Default.Psychology, null, tint = AgentPurple, modifier = Modifier.size(18.dp))
                                    },
                                    onClick = {
                                        onThinkMode(ThinkMode.DEEP)
                                        modeMenuOpen = false
                                    }
                                )
                            }
                        }
                    }

                    Spacer(Modifier.weight(1f))

                    // + 上传文档
                    Box(
                        modifier = Modifier
                            .size(40.dp)
                            .clip(CircleShape)
                            .clickable(onClick = onPickDoc),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = "上传文档",
                            tint = if (docName != null) AgentBlue
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                    Spacer(Modifier.width(6.dp))

                    // 发送 / 停止
                    Box(
                        modifier = Modifier
                            .size(42.dp)
                            .clip(CircleShape)
                            // 三个分支统一成 Brush：混用 Color 会让 if 表达式退化成 Any，background 无重载可匹配
                            .background(
                                when {
                                    generating -> Brush.linearGradient(
                                        listOf(
                                            MaterialTheme.colorScheme.errorContainer,
                                            MaterialTheme.colorScheme.errorContainer
                                        )
                                    )
                                    canSend -> Brush.linearGradient(
                                        listOf(AgentBlue, lerp(AgentBlue, Color.White, 0.22f))
                                    )
                                    else -> Brush.linearGradient(
                                        listOf(
                                            MaterialTheme.colorScheme.surfaceVariant,
                                            MaterialTheme.colorScheme.surfaceVariant
                                        )
                                    )
                                }
                            )
                            .clickable { if (generating) onStop() else if (canSend) onSend() },
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            if (generating) Icons.Default.Stop else Icons.Default.ArrowUpward,
                            contentDescription = if (generating) "停止生成" else "发送",
                            tint = when {
                                generating -> MaterialTheme.colorScheme.onErrorContainer
                                canSend -> Color.White
                                else -> MaterialTheme.colorScheme.onSurfaceVariant
                            },
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
internal fun ErrorBanner(text: String, onDismiss: () -> Unit) {
    LaunchedEffect(text) { kotlinx.coroutines.delay(5000); onDismiss() }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 6.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.errorContainer)
            .padding(horizontal = 12.dp, vertical = 9.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onErrorContainer, modifier = Modifier.weight(1f)
        )
        Icon(Icons.Default.Close, "关闭", tint = MaterialTheme.colorScheme.onErrorContainer,
            modifier = Modifier.size(16.dp).clickable(onClick = onDismiss))
    }
}
