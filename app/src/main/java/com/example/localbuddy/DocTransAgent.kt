package com.example.localbuddy

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color as AndroidColor
import android.graphics.Typeface
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.localbuddy.scan.DocxReader
import com.example.localbuddy.scan.DocxWriter
import com.example.localbuddy.scan.OcrLang
import com.example.localbuddy.scan.OcrEngine
import com.example.localbuddy.scan.PdfExporter
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// ──────────────────────────────────────────────────────────────────
// 数据模型
// ──────────────────────────────────────────────────────────────────
enum class DocMode(val label: String) { TRANSLATE("翻译"), OCR("OCR+翻译") }

data class DocTransUiState(
    val mode: DocMode = DocMode.TRANSLATE,
    val targetLang: String = "en",
    val ocrLangDir: String = OcrLang.DEFAULT_DIR,   // OCR 识别语言（rec 模型目录）
    /** OCR+翻译 模式：识别后是否翻译。默认 false=只 OCR，译文区显示识别内容。 */
    val translateAfterOcr: Boolean = false,
    val fileName: String? = null,
    val sourceText: String = "",
    val translatedText: String = "",
    val extracting: Boolean = false,
    val translating: Boolean = false,
    val progress: Float = 0f,             // 翻译进度 0..1
    val statusText: String = "",
    val error: String? = null
) {
    val busy: Boolean get() = extracting || translating
}

// ──────────────────────────────────────────────────────────────────
// ViewModel
// ──────────────────────────────────────────────────────────────────
class DocTransViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(DocTransUiState())
    val state: StateFlow<DocTransUiState> = _state.asStateFlow()

    // 提取 / 翻译两个后台任务句柄：stop() 强制结束用
    private var extractJob: Job? = null
    private var transJob: Job? = null

    init {
        // 预加载 MT 模型（与其它 MT 智能体一致）：否则首次点击「翻译处理」时 mtReady=false，
        // 全部 segment 走占位返回空 → 译文区什么都没出现。幂等，translate() 内还会再兜底 await。
        viewModelScope.launch { SpeechMTEngine.get(getApplication()).warmUp() }
    }

    fun setMode(mode: DocMode) {
        if (_state.value.busy) return
        _state.update { DocTransUiState(mode = mode, targetLang = it.targetLang,
            ocrLangDir = it.ocrLangDir, translateAfterOcr = it.translateAfterOcr) }
    }
    fun setTargetLang(code: String) = _state.update { it.copy(targetLang = code) }
    fun setOcrLang(dir: String) = _state.update { it.copy(ocrLangDir = dir) }
    fun setTranslateAfterOcr(on: Boolean) = _state.update { it.copy(translateAfterOcr = on) }

    /** 强制结束当前任务（提取或翻译）。cancel 是协作式的：正在跑的单次 ONNX 推理会跑完，但循环（逐段翻译/逐页 OCR）立即停。 */
    fun stop() {
        val had = extractJob?.isActive == true || transJob?.isActive == true
        extractJob?.cancel(); transJob?.cancel()
        extractJob = null; transJob = null
        if (had || _state.value.busy) {
            _state.update { it.copy(extracting = false, translating = false, statusText = "已强制结束") }
        }
    }

    /** 根据 mime 抽取源文本。txt/docx 直读；pdf 渲染成图后 OCR；image 直接 OCR。 */
    fun startFromUri(uri: Uri, name: String, mime: String?) {
        _state.update { it.copy(fileName = name, sourceText = "", translatedText = "",
            extracting = true, progress = 0f, statusText = "读取 / 识别中…", error = null) }
        val ctx = getApplication<Application>()
        // 按所选识别语言路由到对应 rec 模型
        OcrEngine.get(ctx).setRecModel(_state.value.ocrLangDir)
        extractJob?.cancel()
        extractJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val isPdf = mime == "application/pdf"
                val isDirectPdf = isPdf && _state.value.mode != DocMode.OCR
                // OCR+翻译 模式（PDF 逐页 OCR / 图片直读）需先加载 OCR 模型：
                // OcrEngine.runOcr 在未 warmUp 时静默返回空串 -> 表现为「未能识别出文字」。
                // 与全能扫描智能体一致：用前 warmUp（已加载则立即返回）。
                val needsOcr = (isPdf && !isDirectPdf) || mime?.startsWith("image/") == true
                if (needsOcr) {
                    // 先等 init 启动的 MT/ASR/VAD 模型加载完再加载/推理 OCR：ORT 多 session
                    // 「加载 + 推理」并发（均注册 XNNPACK）在部分设备原生崩溃（SIGSEGV）。
                    // warmUp 幂等：已加载立即返回，加载中则等首次完成。
                    _state.update { it.copy(statusText = "等待模型就绪…") }
                    SpeechMTEngine.get(ctx).warmUp()
                    val ocr = OcrEngine.get(ctx)
                    if (!ocr.isReady) {
                        _state.update { it.copy(statusText = "加载 OCR 模型中…") }
                        ocr.warmUp()
                    }
                    // 模型加载失败（OOM/缺资产等）：明确报错，不再静默空结果
                    if (!ocr.isReady) throw IllegalStateException("OCR 模型加载失败，无法识别（详见 logcat Tag=OcrEngine）")
                }
                val text = when {
                    // 翻译模式 PDF 直接抽取文字层（不走 OCR）；OCR+翻译 模式才渲染 OCR
                    isPdf && _state.value.mode != DocMode.OCR -> pdfExtractText(ctx, uri)
                    isPdf -> pdfToText(ctx, uri) { !isActive }
                    mime?.contains("word") == true -> readDocx(ctx, uri)
                    mime?.startsWith("image/") == true -> ocrBitmap(ctx, decodeUri(ctx, uri))
                    else -> readText(ctx, uri)     // text/*
                }
                _state.update { it.copy(sourceText = text, extracting = false,
                    statusText = when {
                        text.isNotBlank() -> "已提取 ${text.length} 字"
                        isDirectPdf -> "该 PDF 无可提取文字（可能是扫描件），请切换到「OCR+翻译」"
                        else -> "未能识别出文字"
                    }) }
            } catch (e: CancellationException) {
                _state.update { it.copy(extracting = false, statusText = "已强制结束") }
                throw e
            } catch (e: Throwable) {
                Log.e(TAG, "extract failed", e)
                _state.update { it.copy(extracting = false, statusText = "", error = "处理失败：${e.message ?: "未知"}") }
            }
        }
    }

    /** 分段串行翻译，进度条随段推进。OCR+翻译 模式未勾选「翻译」时只输出 OCR 结果（不走 MT）。 */
    fun translate() {
        if (_state.value.busy) return
        val src = _state.value.sourceText
        // 只 OCR：识别内容直接作为输出（译文区显示），导出也随之可用。无需协程，即时完成。
        if (_state.value.mode == DocMode.OCR && !_state.value.translateAfterOcr) {
            _state.update { it.copy(translatedText = src, progress = 1f,
                statusText = "OCR 识别完成，可导出") }
            return
        }
        if (src.isBlank()) {
            val hint = if (_state.value.mode == DocMode.TRANSLATE)
                "文档未提取到文字：请确认是 txt / .docx / 文字版 PDF（扫描件请改用「OCR+翻译」）"
            else
                "未识别到文字：请重新选择文档"
            Log.w(TAG, "translate skipped: sourceText blank")
            _state.update { it.copy(statusText = hint) }
            return
        }
        val tgt = _state.value.targetLang
        val ctx = getApplication<Application>()
        _state.update { it.copy(translating = true, translatedText = "", progress = 0f, statusText = "翻译中…", error = null) }
        transJob?.cancel()
        transJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val mt = SpeechMTEngine.get(ctx)
                // 首次使用前确保 MT 已加载（warmUp 幂等：未加载则真正加载并等待完成，已加载直接返回）。
                // 不 gate 的话 mtReady=false → 全部段走占位返回空译文，点击看起来毫无反应。
                if (!mt.isLoaded) {
                    _state.update { it.copy(statusText = "加载翻译模型中…") }
                    mt.warmUp()
                    _state.update { it.copy(statusText = "翻译中…") }
                }
                val chunks = chunkify(src)
                val done = StringBuilder()    // 已翻译完成的整段（每段间 \n 分隔）
                // 把「已完成段 + 当前段流式 token」实时写进译文区：hy-mt 逐 token 解码，不再等整段
                fun tick(cur: String, i: Int) {
                    _state.update {
                        it.copy(
                            translatedText = (done.toString() + cur).trimEnd('\n'),
                            progress = (i + 1).toFloat() / chunks.size,
                            statusText = "翻译中… ${i + 1}/${chunks.size}"
                        )
                    }
                }
                chunks.forEachIndexed { i, seg ->
                    // 强制结束：立即跳出（runCatching 会吞掉 mt.translate 内的 CancellationException，须显式检查）
                    if (!isActive) return@launch
                    if (seg.isBlank()) { done.append('\n'); tick("", i); return@forEachIndexed }
                    val live = StringBuilder()   // 当前段的流式译文
                    val tr = runCatching {
                        mt.translate(seg, "zh", tgt) { p ->
                            live.setLength(0); live.append(p)
                            tick(live.toString(), i)
                        }
                    }.getOrDefault("")
                    // 段完成：把整段译文提交到 done（tick 里会即时渲染），并推进进度
                    done.append(tr.trimEnd()).append('\n')
                    tick("", i)
                }
                val out = (done.toString()).trimEnd()
                if (out.isBlank()) {
                    // 给诊断留证据：源文本长度 / 段数 / MT 模型就绪状态
                    Log.w(TAG, "translate empty: srcLen=${src.length} chunks=${chunks.size} " +
                        "mtReady=${mt.mtReady} mtLoaded=${mt.isLoaded}")
                }
                _state.update { it.copy(translating = false,
                    statusText = if (out.isBlank()) "未能生成译文：翻译模型未就绪（详见 logcat Tag=DocTransVM）" else "翻译完成，可导出") }
            } catch (e: CancellationException) {
                _state.update { it.copy(translating = false, statusText = "已强制结束（保留已译部分）") }
                throw e
            } catch (e: Throwable) {
                Log.e(TAG, "translate failed", e)
                _state.update { it.copy(translating = false, statusText = "", error = "翻译失败：${e.message ?: "未知"}") }
            }
        }
    }

    /** 导出：txt/word/pdf。pdf 用位图化（支持中文任意文字）喂 PdfExporter。 */
    suspend fun writeExport(context: Context, uri: Uri, format: String) = withContext(Dispatchers.IO) {
        context.contentResolver.openOutputStream(uri)?.use { out ->
            when (format) {
                "txt" -> out.write(_state.value.translatedText.toByteArray(Charsets.UTF_8))
                "word" -> DocxWriter.write(_state.value.translatedText, out)
                "pdf" -> PdfExporter.export(renderTextToImages(_state.value.translatedText), out)
            }
        }
    }

    // ── 文本抽取 ──
    private fun readText(ctx: Context, uri: Uri): String =
        ctx.contentResolver.openInputStream(uri)?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""

    private fun readDocx(ctx: Context, uri: Uri): String {
        val text = runCatching {
            ctx.contentResolver.openInputStream(uri)?.use { DocxReader.read(it) }
        }.getOrNull().orEmpty()
        if (text.isNotBlank()) return text
        // 提取不出文字：按文件魔数判定真实类型，把确切原因直接展示到界面（不用查 logcat）
        throw IllegalStateException(fileTypeHint(ctx, uri))
    }

    /** 读文件头魔数判断真实类型，给出明确提示。 */
    private fun fileTypeHint(ctx: Context, uri: Uri): String {
        val magic = ctx.contentResolver.openInputStream(uri)?.use {
            val b = ByteArray(8); val n = it.read(b); if (n > 0) b.copyOf(n) else ByteArray(0)
        } ?: ByteArray(0)
        fun isPk(): Boolean = magic.size >= 4 && magic[0] == 'P'.code.toByte() && magic[1] == 'K'.code.toByte()
        fun isOle(): Boolean = magic.size >= 8 && magic[0] == 0xD0.toByte() && magic[1] == 0xCF.toByte() &&
            magic[2] == 0x11.toByte() && magic[3] == 0xE0.toByte()   // D0CF11E0 = OLE 复合文档（旧版 .doc）
        fun isPdf(): Boolean = magic.size >= 4 && magic[0] == '%'.code.toByte() && magic[1] == 'P'.code.toByte() &&
            magic[2] == 'D'.code.toByte() && magic[3] == 'F'.code.toByte()
        return when {
            isPk()   -> "文件是 zip（.docx）但未提取到正文文本：结构特殊或为空文档，请用 Word 打开后另存一份 .docx"
            isOle()  -> "这是旧版 .doc（二进制 Word），不是 .docx：请在 Word 里【另存为….docx】后再试"
            isPdf()  -> "这是 PDF 不是 Word：请选回「文字版 PDF」或改用「OCR+翻译」模式"
            else     -> "无法读取该文件（非标准 .docx 或已损坏）：请用 Word 另存为 .docx 后重试"
        }
    }

    private fun decodeUri(ctx: Context, uri: Uri): Bitmap =
        android.graphics.BitmapFactory.decodeStream(ctx.contentResolver.openInputStream(uri)) ?: throw IllegalStateException("无法解码图片")

    private fun ocrBitmap(ctx: Context, bmp: Bitmap): String {
        val ocr = com.example.localbuddy.scan.OcrEngine.get(ctx)
        val result = ocr.runOcr(bmp)
        return result.ifBlank { "" }
    }

    /** PDF → 用 PdfBox 直接抽取文字层（翻译模式，不做 OCR）。扫描件无文字层则返回空。 */
    private fun pdfExtractText(ctx: Context, uri: Uri): String {
        val inStream = ctx.contentResolver.openInputStream(uri) ?: return ""
        return try {
            // pdfbox-android 必须先初始化资源加载器：glyphlist/字体等资源从 APK assets 读，
            // 未 init 时 GlyphList 静态初始化抛 IOException -> 文字抽取失败（幂等，重复调用无害）
            com.tom_roush.pdfbox.android.PDFBoxResourceLoader.init(ctx)
            val doc = PDDocument.load(inStream)
            val text = PDFTextStripper().getText(doc)
            doc.close()
            text.trim()
        } catch (e: Throwable) {
            Log.e(TAG, "pdf extract text failed", e)
            ""
        } finally { runCatching { inStream.close() } }
    }

    /** PDF → 逐页渲染成 Bitmap → OCR 拼接。对任意 PDF（含中文/扫描件）都稳。[isCancelled] 为 true 时提前停（强制结束）。 */
    private fun pdfToText(ctx: Context, uri: Uri, isCancelled: () -> Boolean = { false }): String {
        val ocr = com.example.localbuddy.scan.OcrEngine.get(ctx)
        var pfd: ParcelFileDescriptor? = null
        try {
            pfd = ctx.contentResolver.openFileDescriptor(uri, "r") ?: return ""
            val sb = StringBuilder()
            PdfRenderer(pfd).use { renderer ->
                val pageRect = android.graphics.Rect().apply {
                    // 固定 A4 比例渲染页，控制内存
                    renderer.openPage(if (renderer.pageCount > 0) 0 else 0).let { p ->
                        this.set(0, 0, maxOf(1, p.width), maxOf(1, p.height)); p.close()
                    }
                }
                val scale = 1240f / maxOf(1, pageRect.width())
                for (i in 0 until renderer.pageCount) {
                    if (isCancelled()) break   // 强制结束：丢弃剩余页
                    val w = (pageRect.width() * scale).toInt().coerceIn(300, 2000)
                    val h = (pageRect.height() * scale).toInt().coerceIn(300, 3000)
                    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    bmp.eraseColor(AndroidColor.WHITE)
                    val p = renderer.openPage(i)
                    val m = android.graphics.Matrix().apply { setScale(w / p.width.toFloat(), h / p.height.toFloat()) }
                    p.render(bmp, null, m, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                    p.close()
                    val t = ocr.runOcr(bmp)
                    if (t.isNotBlank()) { sb.append(t); sb.append('\n') }
                    bmp.recycle()
                }
            }
            return sb.toString()
        } finally {
            runCatching { pfd?.close() }
        }
    }

    /** 把长文本切成适合翻译的 chunk（按空行聚合，超长行按标点硬切到 ~max 字）。 */
    private fun chunkify(text: String, max: Int = 220): List<String> {
        val chunks = ArrayList<String>()
        var cur = StringBuilder()
        fun flush() { if (cur.isNotBlank()) { chunks.add(cur.toString()); cur = StringBuilder() } }
        for (line in text.lines()) {
            if (line.isBlank()) { flush(); continue }
            if (cur.length + line.length + 1 > max && cur.isNotBlank()) flush()
            if (cur.isNotBlank()) cur.append('\n')
            // 单行超长：按字符硬切
            if (line.length > max) {
                var start = 0
                while (start < line.length) {
                    flush()
                    chunks.add(line.substring(start, minOf(start + max, line.length)))
                    start += max
                }
            } else {
                cur.append(line)
            }
        }
        flush()
        return if (chunks.isEmpty()) listOf(text) else chunks
    }

    /** 把文本按页渲染成 Bitmap（A4 比例白底黑字，中文可用），供 PDF 导出。 */
    private fun renderTextToImages(text: String): List<Bitmap> {
        if (text.isBlank()) return emptyList()
        val pageWidth = 1240
        val pageHeight = 1754
        val padX = 64
        val padY = 56
        val textPaint = TextPaint().apply {
            textSize = 30f
            color = AndroidColor.BLACK
            typeface = Typeface.DEFAULT
        }
        val layout = StaticLayout(
            text, textPaint, pageWidth - padX * 2, Layout.Alignment.ALIGN_NORMAL, 1.0f, 8f, false
        )
        val lineHeight = layout.height / maxOf(1, layout.lineCount).toFloat()
        val perPage = ((pageHeight - padY * 2) / lineHeight).toInt().coerceAtLeast(1)
        val pages = (layout.lineCount + perPage - 1) / perPage
        val out = ArrayList<Bitmap>(pages)
        repeat(pages) { pi ->
            val bmp = Bitmap.createBitmap(pageWidth, pageHeight, Bitmap.Config.ARGB_8888)
            bmp.eraseColor(AndroidColor.WHITE)
            val c = Canvas(bmp)
            val start = pi * perPage
            val end = minOf(start + perPage, layout.lineCount)
            var y = padY.toFloat()
            for (li in start until end) {
                val ls = layout.getLineStart(li); val le = layout.getLineEnd(li)
                c.drawText(text, ls, le, padX.toFloat(), y, textPaint)
                y += lineHeight
            }
            out.add(bmp)
        }
        return out
    }

    companion object { private const val TAG = "DocTransVM" }
}

// ──────────────────────────────────────────────────────────────────
// 页面
// ──────────────────────────────────────────────────────────────────
private val DocAccent = Color(0xFF7C3AED)   // 文档翻译智能体 · 紫
// 设置行首标签固定宽度：需容纳 4 个汉字（如「目标语种」）不折行
private val LABEL_WIDTH = 80.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocTransScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val vm: DocTransViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance((context.applicationContext as Application))
    )
    val state by vm.state.collectAsState()

    // 上传：按模式限制可选 mime
    val pickLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            val mime = context.contentResolver.getType(uri)
            val name = runCatching {
                context.contentResolver.query(uri, null, null, null, null)?.use { c ->
                    val i = c.getColumnIndex("_display_name"); if (i >= 0 && c.moveToFirst()) c.getString(i) else null
                }
            }.getOrNull()
            vm.startFromUri(uri, name ?: uri.lastPathSegment ?: "文档", mime)
        }
    }
    val pickMimes = if (state.mode == DocMode.TRANSLATE)
        arrayOf("text/*", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/pdf")
    else
        arrayOf("image/*", "application/pdf")

    val scope = rememberCoroutineScope()
    // 导出：txt / word / pdf 各自的 CreateDocument
    val txtLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
        if (uri != null) scope.launch { vm.writeExport(context, uri, "txt") }
    }
    val docxLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    ) { uri -> if (uri != null) scope.launch { vm.writeExport(context, uri, "word") } }
    val pdfLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { uri ->
        if (uri != null) scope.launch { vm.writeExport(context, uri, "pdf") }
    }
    var showLang by remember { mutableStateOf(false) }
    var showOcrLang by remember { mutableStateOf(false) }
    var showExport by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            AgentHeader(accent = DocAccent, icon = Icons.Default.Description,
                title = "文档识别翻译智能体", subtitle = "文档 · OCR · 翻译 · 格式还原", onBack = onBack)
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(innerPadding).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // 模式 + 目标语种
            ElevatedCard(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("模式", maxLines = 1, style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.width(LABEL_WIDTH))
                        Row(Modifier.clip(RoundedCornerShape(10.dp)).background(MaterialTheme.colorScheme.surfaceVariant).padding(2.dp)) {
                            DocMode.entries.forEach { mode ->
                                val sel = mode == state.mode
                                Box(Modifier.clip(RoundedCornerShape(8.dp))
                                    .background(if (sel) MaterialTheme.colorScheme.surface else Color.Transparent)
                                    .clickable(enabled = !state.busy) { vm.setMode(mode) }
                                    .padding(horizontal = 12.dp, vertical = 8.dp)) {
                                    Text(mode.label, style = MaterialTheme.typography.bodySmall,
                                        fontWeight = if (sel) FontWeight.SemiBold else FontWeight.Normal,
                                        color = if (sel) DocAccent else MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                            }
                        }
                    }
                    // 识别语言在目标语种之前：OCR+翻译 模式先确定识别语言，再决定是否翻译
                    if (state.mode == DocMode.OCR) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("识别语言", maxLines = 1, style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.width(LABEL_WIDTH))
                            AssistChip(onClick = { showOcrLang = true },
                                label = { Text(OcrLang.label(state.ocrLangDir)) })
                        }
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("目标语种", maxLines = 1, style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface, modifier = Modifier.width(LABEL_WIDTH))
                        val cur = languages.firstOrNull { it.code == state.targetLang } ?: languages[1]
                        AssistChip(onClick = { showLang = true }, label = { Text("${cur.flag} ${cur.name}") })
                        // OCR+翻译 模式：复选框控制识别后是否翻译。默认不选=只 OCR，译文区显示识别内容
                        if (state.mode == DocMode.OCR) {
                            Spacer(Modifier.width(6.dp))
                            Row(verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.clip(RoundedCornerShape(8.dp)).clickable { vm.setTranslateAfterOcr(!state.translateAfterOcr) }) {
                                Checkbox(checked = state.translateAfterOcr,
                                    onCheckedChange = { vm.setTranslateAfterOcr(it) })
                                Text("翻译", style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                    Text(if (state.mode == DocMode.TRANSLATE) "支持：txt / word / pdf（文字版 PDF 直接读取；扫描件请用 OCR+翻译）"
                    else "支持：图片 / pdf（默认仅 OCR；勾选「翻译」后识别并翻译）",
                        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }

            // 上传
            ElevatedCard(Modifier.fillMaxWidth(), shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(onClick = { pickLauncher.launch(pickMimes) }, enabled = !state.busy, modifier = Modifier.fillMaxWidth()) {
                        Icon(Icons.Default.UploadFile, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp))
                        Text(if (state.fileName != null) "重新选择文档" else "选择文档")
                    }
                    if (state.fileName != null) {
                        Text(state.fileName!!, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Button(onClick = { vm.translate() },
                        enabled = state.fileName != null && !state.busy,
                        modifier = Modifier.fillMaxWidth(), colors = agentButtonColors(DocAccent)) {
                        Icon(Icons.Default.Translate, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp))
                        Text(when {
                            state.mode == DocMode.OCR && !state.translateAfterOcr -> "仅 OCR 识别"
                            state.mode == DocMode.OCR -> "OCR + 翻译"
                            else -> "翻译处理"
                        })
                    }
                    // 提取/翻译进行中：提供强制结束（两种模式通用）
                    if (state.busy) {
                        OutlinedButton(onClick = { vm.stop() }, modifier = Modifier.fillMaxWidth()) {
                            Icon(Icons.Default.Stop, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp))
                            Text("强制结束")
                        }
                    }
                    if (state.translating) {
                        LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)))
                    }
                    // 状态文案始终显示：翻译中、加载中、以及「未提取到文字/未能生成译文」等终端情况，
                    // 避免失败时界面毫无反馈、看起来像点击没反应。
                    if (state.statusText.isNotBlank()) {
                        Text(state.statusText, style = MaterialTheme.typography.labelSmall,
                            color = if (state.translating) DocAccent else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }

            // 结果 + 导出
            ElevatedCard(Modifier.fillMaxWidth().weight(1f), shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)) {
                Column(Modifier.fillMaxSize().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    val scroll = rememberScrollState()
                    // 逐段/逐 token 翻译时实时显示已译内容，并随新内容自动滚动到底部（流式更新很快，
                    // 用 scrollTo 直接跳到底，避免 animateScrollTo 频繁重启动画抖动）
                    LaunchedEffect(state.translatedText) {
                        if (state.translatedText.isNotEmpty() && scroll.maxValue > 0) scroll.scrollTo(scroll.maxValue)
                    }
                    // 只 OCR 时结果区即识别内容；勾选翻译后才是译文
                    val resultLabel = when {
                        state.translating -> "译文生成中…"
                        state.mode == DocMode.OCR && !state.translateAfterOcr -> "OCR 识别结果"
                        else -> "译文"
                    }
                    // weight(1f)：文本框弹性占位，导出按钮固定在卡片底部，卡片再矮也不会被挤出不可见
                    TextBox(resultLabel, state.translatedText, scroll, Modifier.weight(1f))
                    if (state.error != null) {
                        Text(state.error!!, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                        OutlinedButton(onClick = { showExport = true },
                            enabled = state.translatedText.isNotBlank(),
                            modifier = Modifier.weight(1f)) {
                            Icon(Icons.Default.Download, null, Modifier.size(18.dp)); Spacer(Modifier.width(6.dp)); Text("导出")
                        }
                        if (state.translatedText.isNotBlank()) {
                            Text("${state.translatedText.length} 字", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
    }

    if (showLang) {
        AlertDialog(onDismissRequest = { showLang = false }, title = { Text("目标语种") },
            text = {
                // 固定高度容器内滚动，避免 38 个语种把弹窗顶出屏且无法拖动
                val langScroll = rememberScrollState()
                Box(Modifier.fillMaxWidth().heightIn(max = 320.dp)) {
                    Column(Modifier.fillMaxWidth().verticalScroll(langScroll)) {
                        languages.forEach { l ->
                            Row(Modifier.fillMaxWidth().clickable { vm.setTargetLang(l.code); showLang = false }
                                .padding(horizontal = 8.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(l.flag, fontSize = 18.sp); Spacer(Modifier.width(10.dp)); Text(l.name)
                            }
                        }
                        Spacer(Modifier.height(4.dp))
                    }
                }
            },
            confirmButton = { TextButton(onClick = { showLang = false }) { Text("关闭") } })
    }
    if (showOcrLang) {
        AlertDialog(onDismissRequest = { showOcrLang = false }, title = { Text("识别语言") },
            text = {
                val os = rememberScrollState()
                Box(Modifier.fillMaxWidth().heightIn(max = 320.dp)) {
                    Column(Modifier.fillMaxWidth().verticalScroll(os)) {
                        OcrLang.MODELS.forEach { m ->
                            val sel = m.dir == state.ocrLangDir
                            Row(Modifier.fillMaxWidth().clickable { vm.setOcrLang(m.dir); showOcrLang = false }
                                .padding(horizontal = 8.dp, vertical = 12.dp),
                                verticalAlignment = Alignment.CenterVertically) {
                                Text(if (sel) "● " else "  ", fontSize = 14.sp, color = if (sel) DocAccent else Color.Transparent)
                                Text(m.label, style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = if (sel) FontWeight.SemiBold else FontWeight.Normal)
                            }
                        }
                    }
                }
            },
            confirmButton = { TextButton(onClick = { showOcrLang = false }) { Text("关闭") } })
    }
    if (showExport) {
        AlertDialog(onDismissRequest = { showExport = false }, title = { Text("导出格式") },
            text = { Column {
                Row(Modifier.fillMaxWidth().clickable { showExport = false; txtLauncher.launch("译文.txt") }.padding(vertical = 12.dp)) {
                    Text("纯文本 (.txt)", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                }
                Row(Modifier.fillMaxWidth().clickable { showExport = false; docxLauncher.launch("译文.docx") }.padding(vertical = 12.dp)) {
                    Text("Word (.docx)", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                }
                Row(Modifier.fillMaxWidth().clickable { showExport = false; pdfLauncher.launch("译文.pdf") }.padding(vertical = 12.dp)) {
                    Text("PDF (.pdf)", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                }
            } },
            confirmButton = { TextButton(onClick = { showExport = false }) { Text("取消") } })
    }
}

// 简易只读文本框；modifier 传入 weight(1f) 时弹性占位（导出按钮固定卡底，不被挤出）
@Composable
private fun TextBox(label: String, text: String, scroll: androidx.compose.foundation.ScrollState? = null,
                    modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(4.dp))
        // min 压到 60：结果卡片被上方挤压时文本框收缩让位，导出按钮始终可见
        Box(Modifier.fillMaxWidth().heightIn(min = 60.dp, max = 300.dp)
            .clip(RoundedCornerShape(12.dp)).background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .verticalScroll(scroll ?: rememberScrollState()).padding(10.dp)) {
            Text(if (text.isBlank()) "处理完成后在此显示译文" else text,
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurface)
        }
    }
}