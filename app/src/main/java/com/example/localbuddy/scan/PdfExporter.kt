package com.example.localbuddy.scan

import android.graphics.Bitmap
import android.graphics.pdf.PdfDocument
import java.io.OutputStream

/**
 * 多图导出 PDF。用 Android 内置 [PdfDocument]，每张图一页（按原图宽高），无额外依赖。
 */
object PdfExporter {

    fun export(images: List<Bitmap>, out: OutputStream) {
        val doc = PdfDocument()
        images.forEachIndexed { i, bmp ->
            // PdfDocument 页面尺寸上限约 14400，超大图按比例缩放
            val maxSide = 2000
            val scale = if (maxOf(bmp.width, bmp.height) > maxSide) maxSide.toFloat() / maxOf(bmp.width, bmp.height) else 1f
            val w = (bmp.width * scale).toInt().coerceAtLeast(1)
            val h = (bmp.height * scale).toInt().coerceAtLeast(1)
            val pageInfo = PdfDocument.PageInfo.Builder(w, h, i + 1).create()
            val page = doc.startPage(pageInfo)
            if (scale < 1f) {
                val scaled = Bitmap.createScaledBitmap(bmp, w, h, true)
                page.canvas.drawBitmap(scaled, 0f, 0f, null)
            } else {
                page.canvas.drawBitmap(bmp, 0f, 0f, null)
            }
            doc.finishPage(page)
        }
        doc.writeTo(out)
        doc.close()
    }
}