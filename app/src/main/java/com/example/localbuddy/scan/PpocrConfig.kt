package com.example.localbuddy.scan

/**
 * PaddleOCR `inference.yml` 的**定向行扫描解析器**。
 *
 * 这套 ppocrv5 导出文件的 shape 非常规整，无需通用 YAML 解析器：
 *  - `character_dict:` 后紧跟一排 `- 字符` 列表（无引号/带引号/含 \uXXXX 转义），直到非 `- ` 行；
 *  - `RecResizeImg.image_shape: [- 3, - 48, - 320]`（内联数组或换行列表）；
 *  - det 的 `NormalizeImage.mean/std`、`DetResizeForTest.resize_long`、`PostProcess.thresh/box_thresh/unclip_ratio/max_candidates`。
 * 仅提取实现需要的字段，未知区块忽略。
 */
object PpocrConfig {

    /** 提取 `character_dict` 字符表（顺序 = CTC 类别索引 1..N）。 */
    fun extractDict(yml: String): List<String> {
        val lines = yml.lines()
        val start = lines.indexOfFirst { it.trimStart().startsWith("character_dict:") }
        if (start < 0) return emptyList()
        val out = ArrayList<String>()
        for (k in start + 1 until lines.size) {
            val t = lines[k].trim()
            // 列表项形如 `- 一`（dash+空格）或 `-　`（dash+全角空格，CN 首项为全角空格）
            if (!t.startsWith("-")) break
            val item = t.substring(1).trim()
            out.add(unquote(item))
        }
        return out
    }

    /** 提取 rec 的 `RecResizeImg.image_shape` → (C, H, W)。缺省 (3, 48, 320)。 */
    fun extractRecShape(yml: String): Triple<Int, Int, Int> {
        val lines = yml.lines()
        val idx = lines.indexOfFirst { it.trimStart().startsWith("RecResizeImg:") }
        if (idx < 0) return Triple(3, 48, 320)
        // 找随后的 image_shape:
        var shape = lines.subList(idx, lines.size).firstOrNull { it.trimStart().startsWith("image_shape:") }
            ?.trim() ?: return Triple(3, 48, 320)
        val list = ArrayList<Int>()
        val inner = shape.substringAfter("image_shape:").trim()
        if (inner.startsWith("[")) {
            inner.substring(1, inner.length - 1).split(',').forEach { (it.trim().toIntOrNull())?.let { v -> list.add(v) } }
        } else if (inner.isEmpty()) {
            // 换行列表：紧跟若干 `- N`
            val shapeIdx = lines.subList(idx, lines.size).indexOfFirst { it.trimStart().startsWith("image_shape:") } + idx
            var k = shapeIdx + 1
            while (k < lines.size && list.size < 4) {
                val t = lines[k].trim()
                if (t.startsWith("- ")) { (t.substring(2).trim().toIntOrNull())?.let { v -> list.add(v) }; k++ }
                else break
            }
        }
        if (list.size < 3) return Triple(3, 48, 320)
        return Triple(list[0], list[1], list[2])
    }

    /** 提取 det 的 `NormalizeImage.mean/std`（`.resize_long` 由调用方分别取）。 */
    fun extractDetNorm(yml: String): Pair<List<Double>, List<Double>> {
        val mean = extractNumberList(yml, "mean:")
        val std = extractNumberList(yml, "std:")
        return (if (mean.size == 3) mean else listOf(0.485, 0.456, 0.406)) to
                (if (std.size == 3) std else listOf(0.229, 0.224, 0.225))
    }

    fun extractResizeLong(yml: String): Int =
        yml.lines().firstOrNull { it.trimStart().startsWith("resize_long:") }
            ?.substringAfter("resize_long:")?.trim()?.toIntOrNull() ?: 960

    fun extractPostFloat(yml: String, key: String, def: Double): Double =
        yml.lines().firstOrNull { it.trimStart().startsWith("$key:") }
            ?.substringAfter("$key:")?.trim()?.toDoubleOrNull() ?: def

    fun extractPostInt(yml: String, key: String, def: Int): Int =
        yml.lines().firstOrNull { it.trimStart().startsWith("$key:") }
            ?.substringAfter("$key:")?.trim()?.toIntOrNull() ?: def

    private fun extractNumberList(yml: String, key: String): List<Double> {
        val lines = yml.lines()
        val idx = lines.indexOfFirst { it.trimStart().startsWith(key) }
        if (idx < 0) return emptyList()
        val valInline = lines[idx].substringAfter(key).trim()
        if (valInline.startsWith("[")) {
            return valInline.substring(1, valInline.length - 1).split(',')
                .mapNotNull { it.trim().toDoubleOrNull() }
        }
        val out = ArrayList<Double>()
        var k = idx + 1
        while (k < lines.size) {
            val t = lines[k].trim()
            if (t.startsWith("- ")) { (t.substring(2).trim().toDoubleOrNull())?.let { out.add(it) }; k++ }
            else break
        }
        return out
    }

    /** 去掉单项外围引号，并解 \uXXXX 转义（dict 中如 "space" 等）。 */
    private fun unquote(s: String): String {
        var v = s
        if (v.length >= 2 && ((v.startsWith("'") && v.endsWith("'")) || (v.startsWith("\"") && v.endsWith("\""))))
            v = v.substring(1, v.length - 1)
        if (!v.contains("\\u")) return v
        val sb = StringBuilder()
        var i = 0
        while (i < v.length) {
            if (v[i] == '\\' && i + 5 < v.length && v[i + 1] == 'u') {
                sb.append(v.substring(i + 2, i + 6).toInt(16).toChar()); i += 6; continue
            }
            sb.append(v[i]); i++
        }
        return sb.toString()
    }
}