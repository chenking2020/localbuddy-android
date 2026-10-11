package com.example.localbuddy.scan

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import java.io.OutputStream

/**
 * 文件导出封装。导出走 SAF（[androidx.activity.result.contract.ActivityResultContracts.CreateDocument]），
 * 由 UI 层 launcher 拿到 Uri 后调这里的写入函数。无需 FileProvider。
 */
object FileExport {

    fun writeBytes(context: Context, uri: Uri, bytes: ByteArray) {
        context.contentResolver.openOutputStream(uri)?.use { it.write(bytes) }
    }

    fun writeBitmap(
        context: Context, uri: Uri, bitmap: Bitmap,
        format: Bitmap.CompressFormat = Bitmap.CompressFormat.JPEG, quality: Int = 95
    ) {
        context.contentResolver.openOutputStream(uri)?.use { os ->
            bitmap.compress(format, quality, os)
        }
    }

    fun outputStream(context: Context, uri: Uri): OutputStream? =
        context.contentResolver.openOutputStream(uri)
}
