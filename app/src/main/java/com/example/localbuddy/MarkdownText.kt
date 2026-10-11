package com.example.localbuddy

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp

/**
 * 轻量 Markdown 渲染器（纯 Compose，无外部依赖）。
 *
 * PaddleOCR-VL 的输出本身是 Markdown（标题/加粗/列表/表格/代码等），用这里把识别结果按结构渲染
 * 成可读视图，而不是一堆带 `#`、`**` 的源码。覆盖常见子集：
 * 标题 #..###### / 段落 / 无序+有序列表 / 代码块 ``` / 引用 > / 表格 | | / 分隔线 --- /
 * 行内 **加粗** *斜体* `代码` ~~删除~~ [链接](url)。
 *
 * 复杂结构（嵌套、数学公式）按普通文本兜底，保证不丢内容。
 */
@Composable
fun MarkdownText(markdown: String, modifier: Modifier = Modifier) {
    val blocks = remember(markdown) { parseBlocks(markdown) }
    if (blocks.isEmpty()) return
    val codeBg = MaterialTheme.colorScheme.surfaceVariant
    val codeText = MaterialTheme.colorScheme.onSurface
    val linkColor = MaterialTheme.colorScheme.primary
    val accent = MaterialTheme.colorScheme.primary
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        blocks.forEach { block ->
            when (block) {
                is MdBlock.Heading -> Text(
                    text = buildInline(block.text, codeBg, codeText, linkColor),
                    style = when (block.level) {
                        1 -> MaterialTheme.typography.titleLarge
                        2 -> MaterialTheme.typography.titleMedium
                        else -> MaterialTheme.typography.titleSmall
                    },
                    fontWeight = FontWeight.Bold
                )

                is MdBlock.Paragraph -> Text(
                    text = buildInline(block.text, codeBg, codeText, linkColor),
                    style = MaterialTheme.typography.bodyMedium
                )

                // 列表项之间的间距要比块间距小，否则条目看着像散落的段落
                is MdBlock.Bullet -> Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    block.items.forEach { item ->
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("•", style = MaterialTheme.typography.bodyMedium, color = accent)
                            Text(
                                text = buildInline(item, codeBg, codeText, linkColor),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }

                is MdBlock.Ordered -> Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    block.items.forEachIndexed { idx, item ->
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("${idx + 1}.", style = MaterialTheme.typography.bodyMedium, color = accent, fontWeight = FontWeight.SemiBold)
                            Text(
                                text = buildInline(item, codeBg, codeText, linkColor),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }

                is MdBlock.Code -> Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(codeBg)
                        .padding(vertical = 8.dp)
                ) {
                    if (block.lang.isNotEmpty()) {
                        Text(
                            block.lang.uppercase(),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(start = 10.dp, bottom = 4.dp)
                        )
                    }
                    Text(
                        text = block.content,
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(rememberScrollState())
                            .padding(horizontal = 10.dp)
                    )
                }

                is MdBlock.Quote -> Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(IntrinsicSize.Min)
                        .clip(RoundedCornerShape(8.dp))
                        .background(codeBg)
                ) {
                    Box(Modifier.width(3.dp).fillMaxHeight().background(accent))
                    Text(
                        text = buildInline(block.text, codeBg, codeText, linkColor),
                        style = MaterialTheme.typography.bodyMedium,
                        fontStyle = FontStyle.Italic,
                        modifier = Modifier.padding(8.dp)
                    )
                }

                is MdBlock.Table -> Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(codeBg)
                        .padding(8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    block.rows.forEachIndexed { ridx, row ->
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            row.forEach { cell ->
                                Text(
                                    text = cell,
                                    style = MaterialTheme.typography.bodySmall,
                                    fontFamily = FontFamily.Monospace,
                                    fontWeight = if (ridx == 0) FontWeight.SemiBold else FontWeight.Normal,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                        if (ridx == 0) HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    }
                }

                MdBlock.Rule -> HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            }
        }
    }
}

// ── 块级结构 ──
private sealed class MdBlock {
    data class Heading(val level: Int, val text: String) : MdBlock()
    data class Paragraph(val text: String) : MdBlock()
    data class Bullet(val items: List<String>) : MdBlock()
    data class Ordered(val items: List<String>) : MdBlock()
    data class Code(val content: String, val lang: String = "") : MdBlock()
    data class Quote(val text: String) : MdBlock()
    data class Table(val rows: List<List<String>>) : MdBlock()
    object Rule : MdBlock()
}

private val HEADING_RE = Regex("^(#{1,6})\\s+(.*)$")
private val RULE_RE = Regex("^(\\*{3,}|-{3,}|_{3,})$")
private val BULLET_RE = Regex("^[-*+]\\s+.+")
private val ORDERED_RE = Regex("^\\d+\\.\\s+.+")
// 表格分隔行：| --- | :---: | --- |
private val TABLE_SEP_RE = Regex("^\\|?\\s*:?-{2,}:?\\s*(\\|\\s*:?-{2,}:?\\s*)+\\|?$")

private fun parseBlocks(md: String): List<MdBlock> {
    val lines = md.split("\n")
    val out = ArrayList<MdBlock>()
    var i = 0
    fun lineEndsBlock(t: String) = t.isEmpty() || t.startsWith("```") || t.startsWith("#") ||
        t.startsWith(">") || BULLET_RE.matches(t) || ORDERED_RE.matches(t) || RULE_RE.matches(t)
    while (i < lines.size) {
        val line = lines[i]
        val t = line.trim()
        if (t.isEmpty()) { i++; continue }

        // 代码块。流式输出时收尾的 ``` 可能还没到，这里把剩余内容整段按代码块渲染，
        // 而不是退化成纯文本——否则整个生成过程都会露出 ``` 源码。
        if (t.startsWith("```")) {
            val lang = t.removePrefix("```").trim().takeWhile { !it.isWhitespace() }
            val sb = StringBuilder()
            i++
            while (i < lines.size && !lines[i].trim().startsWith("```")) { sb.append(lines[i]).append('\n'); i++ }
            i++ // 跳过收尾 ```
            var content = sb.toString().trimEnd('\n')
            // ```kotlin 的语言标记不属于代码内容，首行正好是它时去掉
            if (lang.isNotEmpty() && content.lineSequence().firstOrNull()?.trim().equals(lang, true)) {
                val nl = content.indexOf('\n')
                content = if (nl >= 0) content.substring(nl + 1) else ""
            }
            out.add(MdBlock.Code(content, lang))
            continue
        }
        // 标题
        val h = HEADING_RE.matchEntire(t)
        if (h != null) { out.add(MdBlock.Heading(h.groupValues[1].length, h.groupValues[2].trim())); i++; continue }
        // 分隔线
        if (RULE_RE.matches(t)) { out.add(MdBlock.Rule); i++; continue }
        // 表格：当前行含 |，且下一行是分隔行
        if (t.contains("|") && i + 1 < lines.size && TABLE_SEP_RE.matches(lines[i + 1].trim())) {
            val rows = ArrayList<List<String>>()
            rows.add(splitTableRow(t))
            i += 2
            while (i < lines.size && lines[i].trim().contains("|") && lines[i].trim().isNotEmpty()) {
                rows.add(splitTableRow(lines[i].trim())); i++
            }
            out.add(MdBlock.Table(rows)); continue
        }
        // 引用
        if (t.startsWith(">")) {
            val sb = StringBuilder()
            while (i < lines.size && lines[i].trim().startsWith(">")) {
                sb.append(lines[i].trim().removePrefix(">").trimStart()).append('\n'); i++
            }
            out.add(MdBlock.Quote(sb.toString().trimEnd('\n'))); continue
        }
        // 无序列表
        if (BULLET_RE.matches(t)) {
            val items = ArrayList<String>()
            while (i < lines.size && BULLET_RE.matches(lines[i].trim())) {
                items.add(lines[i].trim().replaceFirst(Regex("^[-*+]\\s+"), "")); i++
            }
            out.add(MdBlock.Bullet(items)); continue
        }
        // 有序列表
        if (ORDERED_RE.matches(t)) {
            val items = ArrayList<String>()
            while (i < lines.size && ORDERED_RE.matches(lines[i].trim())) {
                items.add(lines[i].trim().replaceFirst(Regex("^\\d+\\.\\s+"), "")); i++
            }
            out.add(MdBlock.Ordered(items)); continue
        }
        // 段落：连续收集直到遇到块级边界
        val sb = StringBuilder()
        while (i < lines.size) {
            val lt = lines[i].trim()
            if (lineEndsBlock(lt)) break
            if (sb.isNotEmpty()) sb.append('\n')
            sb.append(lines[i]); i++
        }
        out.add(MdBlock.Paragraph(sb.toString()))
    }
    return out
}

private fun splitTableRow(line: String): List<String> =
    line.trim().trim('|').split("|").map { it.trim() }

// ── 行内格式 ──
// 顺序敏感：**bold** 优先于 *italic*，__bold__ 优先于 _italic_
private val INLINE_RE = Regex(
    """\*\*([^*]+)\*\*""" +          // 1: **加粗**
    """|\*([^*]+)\*""" +             // 2: *斜体*
    """|__([^_]+)__""" +             // 3: __加粗__
    """|_([^_]+)_""" +              // 4: _斜体_
    """|`([^`]+)`""" +              // 5: `行内代码`
    """|~~([^~]+)~~""" +            // 6: ~~删除~~
    """|\[([^\]]+)\]\(([^)]+)\)"""  // 7: 链接文本, 8: url
)

private fun buildInline(text: String, codeBg: Color, codeText: Color, linkColor: Color): AnnotatedString =
    buildAnnotatedString {
        var idx = 0
        for (m in INLINE_RE.findAll(text)) {
            if (m.range.first > idx) append(text.substring(idx, m.range.first))
            val g = m.groupValues
            when {
                g[1].isNotEmpty() -> { pushStyle(SpanStyle(fontWeight = FontWeight.Bold)); append(g[1]); pop() }
                g[2].isNotEmpty() -> { pushStyle(SpanStyle(fontStyle = FontStyle.Italic)); append(g[2]); pop() }
                g[3].isNotEmpty() -> { pushStyle(SpanStyle(fontWeight = FontWeight.Bold)); append(g[3]); pop() }
                g[4].isNotEmpty() -> { pushStyle(SpanStyle(fontStyle = FontStyle.Italic)); append(g[4]); pop() }
                g[5].isNotEmpty() -> { pushStyle(SpanStyle(background = codeBg, color = codeText, fontFamily = FontFamily.Monospace)); append(g[5]); pop() }
                g[6].isNotEmpty() -> { pushStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)); append(g[6]); pop() }
                g[7].isNotEmpty() -> {
                    pushStyle(SpanStyle(color = linkColor, textDecoration = TextDecoration.Underline))
                    append(g[7]); pop()
                }
            }
            idx = m.range.last + 1
        }
        if (idx < text.length) append(text.substring(idx))
    }
