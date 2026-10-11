package com.example.localbuddy.scan

import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * 简单 docx 生成器：zip(word/document.xml + [Content_Types].xml + rels)，纯文本段落。
 * 纯 java.util.zip，无外部依赖。Word/WPS 可打开。
 */
object DocxWriter {

    fun write(text: String, out: OutputStream) {
        val body = StringBuilder()
        body.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>")
        body.append("<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>")
        for (p in text.split("\n")) {
            val esc = p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body.append("<w:p><w:r><w:t xml:space=\"preserve\">").append(esc).append("</w:t></w:r></w:p>")
        }
        body.append("</w:body></w:document>")

        ZipOutputStream(out).use { zip ->
            zip.putNextEntry(ZipEntry("[Content_Types].xml")); zip.write(CONTENT_TYPES.toByteArray()); zip.closeEntry()
            zip.putNextEntry(ZipEntry("_rels/.rels")); zip.write(RELS.toByteArray()); zip.closeEntry()
            zip.putNextEntry(ZipEntry("word/document.xml")); zip.write(body.toString().toByteArray()); zip.closeEntry()
        }
    }

    private val CONTENT_TYPES =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">" +
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>" +
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>" +
        "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/></Types>"

    private val RELS =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">" +
        "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>"
}
