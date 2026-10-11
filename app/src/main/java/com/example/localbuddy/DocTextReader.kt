package com.example.localbuddy

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color as AndroidColor
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.util.Log
import com.example.localbuddy.scan.DocxReader
import com.example.localbuddy.scan.OcrEngine
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * 文档 → 纯文本抽取（文本对话智能体的「上传文档问答」用）。
 *
 * 与文档识别翻译智能体的取向不同：那边要**逐字逐段保真**（译文要对齐原文结构），这里只要
 * **语义内容**——抽取结果会先经 [TextExtractor.retrieve] 按问题筛一遍再进 prompt，
 * 所以格式、页眉页脚这类噪声丢掉反而更干净。
 *
 * 支持：txt / .docx / 文字版 PDF（PdfBox 抽文字层）/ 图片与扫描件 PDF（逐页渲染后 OCR）。
 * 图片与扫描件 PDF 需要 OCR 模型，调用前需确保 [OcrEngine.warmUp] 已完成。
 */
internal object DocTextReader {

    private const val TAG = "DocTextReader"

    /**
     * 按 mime 抽取文本。
     *
     * @param mime contentResolver 给出的 MIME；未知时按扩展名兜底
     * @return 抽取出的正文；失败抛出带明确原因的异常（直接展示给用户）
     */
    suspend fun read(ctx: Context, uri: Uri, mime: String?, name: String): String =
        withContext(Dispatchers.IO) {
            val isPdf = mime == "application/pdf" || name.endsWith(".pdf", true)
            val isDocx = mime?.contains("word") == true || name.endsWith(".docx", true)
            val isImage = mime?.startsWith("image/") == true
            when {
                isImage -> ocrBitmap(ctx, decodeUri(ctx, uri))
                isDocx -> readDocx(ctx, uri)
                isPdf -> {
                    // 先试文字层（快、准）；空则判定为扫描件 → 渲染 OCR
                    val direct = pdfExtractText(ctx, uri)
                    if (direct.isNotBlank()) direct
                    else {
                        Log.i(TAG, "pdf has no text layer, fallback to OCR page-by-page")
                        ensureOcr(ctx)
                        pdfToText(ctx, uri) { isActive }
                    }
                }
                else -> readText(ctx, uri)
            }
        }

    private fun readText(ctx: Context, uri: Uri): String =
        ctx.contentResolver.openInputStream(uri)?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }
            ?: throw IllegalStateException("无法读取文件内容")

    private fun readDocx(ctx: Context, uri: Uri): String {
        val text = runCatching {
            ctx.contentResolver.openInputStream(uri)?.use { DocxReader.read(it) }
        }.getOrNull().orEmpty()
        if (text.isNotBlank()) return text
        throw IllegalStateException("未能从 .docx 提取文字：请用 Word 另存为 .docx 后重试")
    }

    private fun decodeUri(ctx: Context, uri: Uri): Bitmap =
        BitmapFactory2.decode(ctx, uri) ?: throw IllegalStateException("无法解码图片")

    private suspend fun ocrBitmap(ctx: Context, bmp: Bitmap): String {
        ensureOcr(ctx)
        val text = OcrEngine.get(ctx).runOcr(bmp)
        if (text.isBlank()) throw IllegalStateException("图片未识别到文字")
        return text
    }

    /**
     * OCR 模型按需加载：OcrEngine.runOcr 在未 warmUp 时静默返回空串（表现为"识别不出文字"而非报错），
     * 这里显式加载并在失败时给出明确错误。幂等：已加载立即返回。
     */
    private suspend fun ensureOcr(ctx: Context) {
        val ocr = OcrEngine.get(ctx)
        if (ocr.isReady) return
        ocr.warmUp()
        if (!ocr.isReady) throw IllegalStateException("OCR 模型加载失败，无法识别图片/扫描件")
    }

    /** PDF → PdfBox 直接抽取文字层（无文字层的扫描件返回空串，由调用方降级 OCR）。 */
    private fun pdfExtractText(ctx: Context, uri: Uri): String {
        val inStream = ctx.contentResolver.openInputStream(uri) ?: return ""
        return try {
            // pdfbox-android 必须先初始化资源加载器：glyphlist/字体等资源从 APK assets 读，
            // 未 init 时 GlyphList 静态初始化抛 IOException -> 文字抽取失败（幂等，重复调用无害）
            PDFBoxResourceLoader.init(ctx)
            val doc = PDDocument.load(inStream)
            val text = PDFTextStripper().getText(doc)
            doc.close()
            text.trim()
        } catch (e: Throwable) {
            Log.e(TAG, "pdf extract text failed", e)
            ""
        } finally {
            runCatching { inStream.close() }
        }
    }

    /** PDF → 逐页渲染成 Bitmap → OCR 拼接（调用前需确保 [ensureOcr] 已完成）。[isCancelled] 为 true 时提前停。 */
    private fun pdfToText(ctx: Context, uri: Uri, isCancelled: () -> Boolean = { false }): String {
        val ocr = OcrEngine.get(ctx)
        var pfd: ParcelFileDescriptor? = null
        try {
            pfd = ctx.contentResolver.openFileDescriptor(uri, "r") ?: return ""
            val sb = StringBuilder()
            PdfRenderer(pfd).use { renderer ->
                // 最多处理前 20 页：问答场景不会用到更后面的内容，且 OCR 逐页很慢
                val limit = renderer.pageCount.coerceAtMost(20)
                for (i in 0 until limit) {
                    if (isCancelled()) break
                    val p = renderer.openPage(i)
                    val w = p.width.coerceIn(300, 2000)
                    val h = p.height.coerceIn(300, 3000)
                    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                    bmp.eraseColor(AndroidColor.WHITE)
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
}

/** BitmapFactory 的空安全包装（decodeStream 可能返回 null 或抛异常）。 */
private object BitmapFactory2 {
    fun decode(ctx: Context, uri: Uri): Bitmap? = runCatching {
        android.graphics.BitmapFactory.decodeStream(ctx.contentResolver.openInputStream(uri))
    }.getOrNull()
}
