package com.example.localbuddy.scan

import android.util.Log
import org.apache.poi.xwpf.usermodel.XWPFDocument
import java.io.InputStream

/**
 * 从 .docx 提取纯文本：用 Apache POI 的 XWPFDocument 解析真实 Word 文档（命名空间、文本框、
 * 混合 run、AlternateContent 等全部覆盖），按 <w:p> 分段输出文本。
 * 纯流输入（不依赖 File），Android 上直接对接 contentResolver 的 InputStream。
 */
object DocxReader {

    fun read(input: InputStream): String {
        try {
            val doc = XWPFDocument(input)
            val sb = StringBuilder()
            for (para in doc.paragraphs) {
                val text = para.text.trim()
                if (text.isNotEmpty()) {
                    if (sb.isNotEmpty()) sb.append('\n')
                    sb.append(text)
                }
            }
            doc.close()
            val out = sb.toString()
            if (out.isEmpty()) Log.w(TAG, "POI: docx parsed but no text extracted")
            return out
        } catch (e: Exception) {
            Log.w(TAG, "POI read failed", e)
            return ""
        }
    }

    private const val TAG = "DocxReader"
}