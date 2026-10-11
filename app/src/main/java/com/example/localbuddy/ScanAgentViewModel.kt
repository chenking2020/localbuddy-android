package com.example.localbuddy

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.localbuddy.SpeechMTEngine
import com.example.localbuddy.scan.DocScanner
import com.example.localbuddy.scan.OcrEngine
import android.graphics.BitmapFactory
import android.net.Uri
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.UUID

data class ScanPage(val id: String, val original: Bitmap, val scanned: Bitmap? = null, val scanning: Boolean = false)

data class ScanAgentState(
    val pages: List<ScanPage> = emptyList(),
    val mode: DocScanner.Mode = DocScanner.Mode.DOCUMENT,
    val colorMode: Boolean = false,            // false=黑白（默认），true=彩色扫描
    val processing: Boolean = false,
    val statusText: String = "",
    val ocrText: String = "",
    val ocrProcessing: Boolean = false,
    val translatedText: String = "",
    val targetLang: String = "en",
    val translating: Boolean = false
)

/**
 * 全能扫描智能体状态：图片页列表 + 扫描模式 + 处理状态。
 * 阶段1（扫描+导出 MVP）：拍照/选图 -> OpenCV 扫描 -> 多图合成/图片/PDF 导出。
 */
class ScanAgentViewModel(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(ScanAgentState())
    val state: StateFlow<ScanAgentState> = _state.asStateFlow()

    init { DocScanner.ensureInitialized() }

    fun setMode(mode: DocScanner.Mode) = _state.update { it.copy(mode = mode) }

    /** 切换扫描色彩：黑白 / 彩色（所有格式通用）。 */
    fun setColorMode(on: Boolean) = _state.update { it.copy(colorMode = on) }

    /** 追加一页并**自动即时扫描**：缩略图先显示 loading，扫描结果就绪后写回（照片/选图不再需要单独点扫描）。 */
    fun addPage(bitmap: Bitmap) {
        val scaled = downscale(bitmap, 1600)  // 限最大边，控内存 + 加速 OpenCV 处理
        val id = UUID.randomUUID().toString()
        _state.update { it.copy(pages = it.pages + ScanPage(id, scaled, scanning = true)) }
        viewModelScope.launch(Dispatchers.Default) {
            val mode = _state.value.mode
            val color = _state.value.colorMode
            val scanned = runCatching { DocScanner.scan(scaled, mode, color) }.getOrNull()
            _state.update { s ->
                s.copy(pages = s.pages.map { p -> if (p.id == id) p.copy(scanned = scanned, scanning = false) else p })
            }
        }
    }

    private fun downscale(bmp: Bitmap, maxSide: Int): Bitmap {
        val s = maxOf(bmp.width, bmp.height)
        if (s <= maxSide) return bmp
        val r = maxSide.toFloat() / s
        return Bitmap.createScaledBitmap(bmp, (bmp.width * r).toInt(), (bmp.height * r).toInt(), true)
    }

    fun removePage(id: String) = _state.update { it.copy(pages = it.pages.filterNot { p -> p.id == id }) }

    /** 把 from 位置的页移到 to 位置（拖拽排序；导出/OCR 顺序随之变化）。 */
    fun movePage(from: Int, to: Int) = _state.update { s ->
        val p = s.pages.toMutableList()
        if (from in p.indices && to in p.indices && from != to) {
            val item = p.removeAt(from)
            p.add(to, item)
        }
        s.copy(pages = p)
    }

    fun clear() = _state.update { it.copy(pages = emptyList()) }

    /** 扫描所有页（OpenCV 边缘检测+透视+增强）。在 Default 调度器，避免阻塞 UI。 */
    fun scanAll() {
        val pages = _state.value.pages
        val mode = _state.value.mode
        val color = _state.value.colorMode
        if (pages.isEmpty() || _state.value.processing) return
        _state.update { it.copy(processing = true, statusText = "扫描中…") }
        viewModelScope.launch(Dispatchers.Default) {
            val newPages = pages.map { p -> p.copy(scanned = DocScanner.scan(p.original, mode, color)) }
            _state.update { it.copy(pages = newPages, processing = false, statusText = "") }
        }
    }

    /** OCR 识别所有扫描页（PaddleOCR-VL 端到端）。流式回写 ocrText，识别中即可看到结果逐字出现。 */
    fun runOcr() {
        val images = exportImages()
        if (images.isEmpty() || _state.value.ocrProcessing) return
        _state.update { it.copy(ocrProcessing = true, statusText = "OCR 识别中…") }
        viewModelScope.launch(Dispatchers.Default) {
            val engine = OcrEngine.get(getApplication())
            engine.warmUp()
            val sb = StringBuilder()
            try {
                images.forEachIndexed { i, bmp ->
                    val prior = sb.toString()
                    val part = engine.runOcr(bmp) { partial ->
                        // 流式：前面已识别图的文本 + 当前图的部分结果
                        _state.update { it.copy(ocrText = prior + partial) }
                    }
                    sb.append(part)
                    if (i < images.size - 1) sb.append("\n")
                }
                _state.update { it.copy(ocrText = sb.toString(), statusText = "") }
            } catch (e: Exception) {
                _state.update { it.copy(statusText = "OCR 失败：${e.message ?: "未知错误"}") }
            } finally {
                _state.update { it.copy(ocrProcessing = false) }
            }
        }
    }

    /** 直接对上传图片 URI 做 OCR（不经过扫描）。 */
    fun runOcrOnUri(uri: Uri) {
        if (_state.value.ocrProcessing) return
        _state.update { it.copy(ocrProcessing = true, statusText = "OCR 识别中…") }
        viewModelScope.launch(Dispatchers.Default) {
            val engine = OcrEngine.get(getApplication())
            try {
                val bmp = withContext(Dispatchers.IO) {
                    getApplication<Application>().contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                }
                if (bmp != null) {
                    engine.warmUp()
                    val text = engine.runOcr(bmp) { partial ->
                        _state.update { it.copy(ocrText = partial) }
                    }
                    _state.update { it.copy(ocrText = text, statusText = "") }
                } else {
                    _state.update { it.copy(statusText = "无法读取图片") }
                }
            } catch (e: Exception) {
                _state.update { it.copy(statusText = "OCR 失败：${e.message ?: "未知错误"}") }
            } finally {
                _state.update { it.copy(ocrProcessing = false) }
            }
        }
    }

    fun setOcrText(t: String) = _state.update { it.copy(ocrText = t) }

    fun setTargetLang(lang: String) = _state.update { it.copy(targetLang = lang) }

    /** 翻译 OCR 结果到目标语言（复用 SpeechMTEngine）。srcLang 不影响指令，传 "zh"。 */
    fun translate() {
        val src = _state.value.ocrText
        if (src.isBlank() || _state.value.translating) return
        translateText(src)
    }

    /** 翻译给定文本（上传 Word/Txt 用）。 */
    fun translateText(src: String) {
        if (src.isBlank() || _state.value.translating) return
        val tgt = _state.value.targetLang
        _state.update { it.copy(translating = true, statusText = "翻译中…") }
        viewModelScope.launch(Dispatchers.Default) {
            val engine = SpeechMTEngine.get(getApplication())
            engine.warmUp()
            val result = engine.translate(src, "zh", tgt)
            _state.update { it.copy(translatedText = result, translating = false, statusText = "") }
        }
    }

    fun setTranslatedText(t: String) = _state.update { it.copy(translatedText = t) }

    /** 导出用图：优先扫描结果，无则原图。 */
    fun exportImages(): List<Bitmap> = _state.value.pages.mapNotNull { it.scanned ?: it.original }
}
