package com.example.localbuddy.scan

import android.graphics.Bitmap
import android.graphics.Canvas

/**
 * 多图合成：等宽竖向拼接成一张长图（每张缩放到统一宽度，按高度顺序排列）。
 */
object ImageComposer {

    fun composeVertical(images: List<Bitmap>, targetWidth: Int = 1000): Bitmap {
        if (images.isEmpty()) return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        val scaled = images.map { bmp ->
            val ratio = targetWidth.toFloat() / bmp.width
            Bitmap.createScaledBitmap(
                bmp, targetWidth,
                (bmp.height * ratio).toInt().coerceAtLeast(1), true
            )
        }
        val totalH = scaled.sumOf { it.height }
        val out = Bitmap.createBitmap(targetWidth, totalH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        var y = 0
        scaled.forEach { bmp ->
            canvas.drawBitmap(bmp, 0f, y.toFloat(), null)
            y += bmp.height
        }
        return out
    }
}