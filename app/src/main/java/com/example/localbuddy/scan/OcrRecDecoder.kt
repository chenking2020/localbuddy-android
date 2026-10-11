package com.example.localbuddy.scan

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * PP-OCR 识别(rec)侧工具：把检测框矫正、切分、缩放到 rec 输入、CTC 贪心解码、按阅读顺序排版。
 *
 * 关键约定（已离线实证）：
 *  - rec 输入为 **BGR, 像素/255（0..1）**, NCHW;高固定 48、宽动态(16 倍数, 上限 maxW=320)；
 *  - 输出 logits [N, T, C]，C = len(dict)+2；CTC：class 0=blank，class i(dict[i-1])，class len(dict)+1=end；
 *
 * buildRecTensors 管线（解决倾斜/挤行/英文连字三类问题）：
 *  1) 透视矫正：检测 quad（minAreaRect 外扩，点序 TL,TR,BR,BL）经 setPolyToPoly 拉成水平条带，
 *     旋转/倾斜文字先回正再识别，否则条带内文字是斜的、AABB 裁剪还混入大量背景；
 *  2) 行切分：条带过高（det 把整段文字框成一个块）时按行投影空白带拆成多行，
 *     否则多行压到 48 高全部糊掉、也丢失分段；
 *  3) 宽行切段：等比换算后宽超 maxW*1.4 的行（英文长行宽可达中文 3-5 倍）切成多段逐段识别，
 *     切点尽量落在空白列（词边界），段间由调用方按 [RecSegment.gapBefore] 补空格。
 */
object OcrRecDecoder {

    /** 一段 rec 输入：NCHW BGR /255 数据 + 宽高 + 拼接标记。 */
    class RecSegment(val nchw: FloatArray, val w: Int, val h: Int,
                     /** 本段是一行的开始（行切分产生或行内首段），拼接时前置换行。 */
                     val lineStart: Boolean,
                     /** 本段前是词间空白（切在空白列），拼接时前置空格。 */
                     val gapBefore: Boolean)

    fun buildRecTensors(bitmap: Bitmap, quad: FloatArray, recH: Int, maxW: Int): List<RecSegment> {
        // 1) 透视矫正：quad -> 水平条带
        val strip = rectify(bitmap, quad)
        // 2) 行切分：整段框拆成单行
        val lines = splitLines(strip)
        // 3) 每行按需切段
        val out = ArrayList<RecSegment>()
        for (line in lines) out.addAll(sliceLine(line, recH, maxW))
        return out
    }

    // ── 1) 透视矫正 ──
    /** quad(TL,TR,BR,BL) -> 水平条带位图（宽=上下边均值、高=左右边均值）。白底，双线性采样。 */
    private fun rectify(bitmap: Bitmap, quad: FloatArray): Bitmap {
        val top = dist(quad[0], quad[1], quad[2], quad[3])
        val bottom = dist(quad[6], quad[7], quad[4], quad[5])
        val left = dist(quad[0], quad[1], quad[6], quad[7])
        val right = dist(quad[2], quad[3], quad[4], quad[5])
        val w = ((top + bottom) / 2).toInt().coerceIn(2, 4096)
        val h = ((left + right) / 2).toInt().coerceIn(2, 512)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.eraseColor(Color.WHITE)
        val m = Matrix()
        val ok = m.setPolyToPoly(quad, 0,
            floatArrayOf(0f, 0f, w.toFloat(), 0f, w.toFloat(), h.toFloat(), 0f, h.toFloat()), 0, 4)
        val c = Canvas(out)
        val p = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        if (ok) {
            c.drawBitmap(bitmap, m, p)
        } else {
            // 退化 quad：回退轴对齐 AABB 裁剪（并 clamp 边界）
            val b = aabb(quad)
            val x0 = b[0].coerceIn(0, bitmap.width - 1); val y0 = b[1].coerceIn(0, bitmap.height - 1)
            val x1 = b[2].coerceIn(x0 + 1, bitmap.width); val y1 = b[3].coerceIn(y0 + 1, bitmap.height)
            val src = Bitmap.createBitmap(bitmap, x0, y0, x1 - x0, y1 - y0)
            val scaled = Bitmap.createScaledBitmap(src, w, h, true)
            if (scaled !== src) src.recycle()
            c.drawBitmap(scaled, 0f, 0f, p)
            if (scaled !== out) scaled.recycle()
        }
        return out
    }

    // ── 2) 行切分 ──
    /** 条带按行投影（每行墨水像素数）找空白带，拆成多个单行位图；单行原样返回。 */
    private fun splitLines(strip: Bitmap): List<Bitmap> {
        val w = strip.width; val h = strip.height
        if (h < 8) return listOf(strip)
        val px = IntArray(w * h)
        strip.getPixels(px, 0, w, 0, 0, w, h)
        val rowInk = IntArray(h)
        for (y in 0 until h) {
            val base = y * w
            var ink = 0
            for (x in 0 until w) if (isInk(px[base + x])) ink++
            rowInk[y] = ink
        }
        val blankThresh = max(1, (w * 0.02).toInt())
        val bands = ArrayList<IntRange>()
        var s = -1
        for (y in 0 until h) {
            if (rowInk[y] > blankThresh) { if (s < 0) s = y }
            else if (s >= 0) { bands.add(s..(y - 1)); s = -1 }
        }
        if (s >= 0) bands.add(s..(h - 1))
        bands.removeAll { it.last - it.first + 1 < 4 }   // 噪声带
        if (bands.size <= 1) return listOf(strip)
        return bands.map { Bitmap.createBitmap(strip, 0, it.first, w, it.last - it.first + 1) }
    }

    // ── 3) 宽行切段 ──
    /** 单行位图 -> 1..N 段。轻度超宽（<1.4 倍）直接挤压（模型对温和挤压鲁棒）；重度切多段，切点优先空白列。 */
    private fun sliceLine(line: Bitmap, recH: Int, maxW: Int): List<RecSegment> {
        val srcW = line.width; val srcH = line.height
        val fullW = srcW.toDouble() * recH / srcH
        if (fullW <= maxW * 1.4) {
            val t = buildSlice(line, 0, 0, srcW, srcH, recH, maxW)
            return listOf(RecSegment(t.nchw, t.w, t.h, lineStart = true, gapBefore = false))
        }
        // 每列墨水像素数：定位词间空白列
        val px = IntArray(srcW * srcH)
        line.getPixels(px, 0, srcW, 0, 0, srcW, srcH)
        val ink = IntArray(srcW)
        for (yy in 0 until srcH) {
            val base = yy * srcW
            for (xx in 0 until srcW) if (isInk(px[base + xx])) ink[xx]++
        }
        val gapThresh = max(1, (srcH * 0.06).toInt())

        val n = ceil(fullW / maxW).toInt().coerceAtLeast(2)
        val ideal = srcW.toDouble() / n
        val cuts = ArrayList<Pair<Int, Boolean>>(n - 1)   // (位置, 是否空白列)
        for (i in 1 until n) {
            val target = (i * ideal).toInt()
            val window = (ideal * 0.35).toInt().coerceAtLeast(1)
            var best = -1; var bestDist = Int.MAX_VALUE
            for (cx in (target - window).coerceAtLeast(1)..(target + window).coerceAtMost(srcW - 1)) {
                if (ink[cx] <= gapThresh) {
                    val d = abs(cx - target)
                    if (d < bestDist) { bestDist = d; best = cx }
                }
            }
            cuts.add(if (best >= 0) best to true else target to false)
        }
        val bounds = (listOf(0) + cuts.map { it.first } + listOf(srcW)).distinct().sorted()

        val out = ArrayList<RecSegment>(bounds.size - 1)
        for (k in 0 until bounds.size - 1) {
            val sx = bounds[k]; val ex = bounds[k + 1]
            if (ex - sx < 4) continue   // 过窄段丢弃
            val t = buildSlice(line, sx, 0, ex, srcH, recH, maxW)
            val isGap = cuts.firstOrNull { it.first == sx }?.second == true
            out.add(RecSegment(t.nchw, t.w, t.h, lineStart = k == 0, gapBefore = isGap))
        }
        if (out.isEmpty()) {
            val t = buildSlice(line, 0, 0, srcW, srcH, recH, maxW)
            return listOf(RecSegment(t.nchw, t.w, t.h, lineStart = true, gapBefore = false))
        }
        return out
    }

    /** 单段裁剪+缩放（双线性）成 rec 输入。高固定 recH；宽按比例取 16 倍数、上限 maxW。 */
    private fun buildSlice(src: Bitmap, x0: Int, y0: Int, x1: Int, y1: Int, recH: Int, maxW: Int): RecSegment {
        val srcW = max(1, x1 - x0); val srcH = max(1, y1 - y0)
        var w = max(16, (srcW.toDouble() * recH / srcH / 16.0).toInt() * 16)
        if (w > maxW) w = maxW
        val h = recH
        val crop = Bitmap.createBitmap(src, x0, y0, srcW, srcH)
        val scaled = Bitmap.createScaledBitmap(crop, w, h, true)
        if (scaled !== crop) crop.recycle()
        val px = IntArray(w * h)
        scaled.getPixels(px, 0, w, 0, 0, w, h)
        if (scaled !== crop) scaled.recycle()

        // NCHW, BGR, /255
        val out = FloatArray(3 * h * w)
        for (y in 0 until h) for (x in 0 until w) {
            val argb = px[y * w + x]
            val r = (argb shr 16) and 0xFF; val g = (argb shr 8) and 0xFF; val b = argb and 0xFF
            // BGR 通道序: ch0=B, ch1=G, ch2=R
            out[(0 * h + y) * w + x] = b / 255f
            out[(1 * h + y) * w + x] = g / 255f
            out[(2 * h + y) * w + x] = r / 255f
        }
        return RecSegment(out, w, h, lineStart = true, gapBefore = false)
    }

    private fun isInk(argb: Int): Boolean {
        val lum = ((argb shr 16 and 0xFF) * 299 + (argb shr 8 and 0xFF) * 587 + (argb and 0xFF) * 114) / 1000
        return lum < 160
    }

    private fun dist(x0: Float, y0: Float, x1: Float, y1: Float): Float {
        val dx = x1 - x0; val dy = y1 - y0
        return sqrt(dx * dx + dy * dy)
    }

    private fun aabb(q: FloatArray): IntArray {
        var x0 = Int.MAX_VALUE; var y0 = Int.MAX_VALUE; var x1 = Int.MIN_VALUE; var y1 = Int.MIN_VALUE
        for (i in 0 until 4) {
            x0 = min(x0, q[i * 2].toInt()); x1 = max(x1, q[i * 2].toInt())
            y0 = min(y0, q[i * 2 + 1].toInt()); y1 = max(y1, q[i * 2 + 1].toInt())
        }
        return intArrayOf(x0, y0, x1, y1)
    }

    /** CTC 贪心解码：按类别 argmax -> 去同字符合并 -> 跳过 blank(0)/end(len+1)。 */
    fun ctcDecode(logits: FloatArray, timeSteps: Int, numClasses: Int, dict: List<String>): String {
        val end = dict.size + 1
        val sb = StringBuilder()
        var prev = -1
        for (ti in 0 until timeSteps) {
            val base = ti * numClasses
            var best = 0; var bv = Float.NEGATIVE_INFINITY
            for (c in 0 until numClasses) {
                val v = logits[base + c]
                if (v > bv) { bv = v; best = c }
            }
            if (best == 0 || best == end) { prev = -1; continue }
            if (best == prev) continue
            if (best - 1 in 0 until dict.size) sb.append(dict[best - 1])
            prev = best
        }
        return sb.toString().trim()
    }

    /**
     * 按阅读顺序排版：框中心 y 接近者并入同一行、行内按 x 排、行间按 y 排。
     * @return 纯文本（行间换行、行内空格）。
     */
    fun layout(boxes: List<Pair<DbBox, String>>): String {
        val placed = boxes.mapNotNull { (b, t) ->
            val s = t.trim()
            if (s.isEmpty()) null
            else Placed(b.cy, b.cx, bboxH(b.quad), s)
        }
        if (placed.isEmpty()) return ""
        val sorted = placed.sortedWith(compareBy({ it.cy }, { it.x }))
        val rows = ArrayList<MutableList<Placed>>()
        for (p in sorted) {
            var target: MutableList<Placed>? = null
            for (r in rows) {
                val ref = r[r.size - 1]
                if (abs(p.cy - ref.cy) < max(p.h, ref.h) * 0.7f) { target = r; break }
            }
            if (target == null) { rows.add(ArrayList(listOf(p))) } else target.add(p)
        }
        return rows.joinToString("\n") { row -> row.sortedBy { it.x }.joinToString(" ") { it.text } }
    }

    private data class Placed(val cy: Float, val x: Float, val h: Float, val text: String)

    private fun bboxH(q: FloatArray): Float {
        var y0 = Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
        for (i in 0 until 4) { y0 = min(y0, q[i * 2 + 1]); y1 = max(y1, q[i * 2 + 1]) }
        return y1 - y0
    }
}
