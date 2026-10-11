package com.example.localbuddy.scan

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * PP-OCR DB 检测后处理：概率图 → 文本块四边形。
 *
 * 步骤（对齐 PaddleOCR `DBPostProcess`）：
 *  1. `prob > thresh` 二值化 → 8 连通域标注；
 *  2. 每块取凸包 → 旋转卡尺求**最小外接矩形**；
 *  3. `unclip`：按 unclipRatio×area/perimeter 将四边形向外扩（miter 偏移）；
 *  4. 过滤：块内平均置信度 > boxThresh、宽高 ≥ 3px、像素数 ≥ 6；
 *  5. 返回图像坐标系四边形的 4 角（8 值）+ 置信度。
 */
class DbBox(val quad: FloatArray, val score: Float) {
    val cx: Float get() = (quad[0] + quad[2] + quad[4] + quad[6]) / 4f
    val cy: Float get() = (quad[1] + quad[3] + quad[5] + quad[7]) / 4f
    fun scaled(sx: Float, sy: Float): DbBox {
        val q = FloatArray(8)
        for (i in 0 until 4) { q[i * 2] = quad[i * 2] * sx; q[i * 2 + 1] = quad[i * 2 + 1] * sy }
        return DbBox(q, score)
    }
}

object DbPostprocess {

    fun run(prob: FloatArray, w: Int, h: Int,
            thresh: Double = 0.3, boxThresh: Double = 0.6, unclipRatio: Double = 1.5): List<DbBox> {
        val n = w * h
        val bin = ByteArray(n)
        for (i in 0 until n) bin[i] = if (prob[i] > thresh) 1 else 0

        val labels = labelComponents(bin, w, h)
        if (labels == null) return emptyList()

        // 聚合每个连通域的像素下标
        val regionMap = HashMap<Int, ArrayList<Int>>()
        for (i in 0 until n) {
            val l = labels[i]
            if (l != 0) regionMap.getOrPut(l) { ArrayList() }.add(i)
        }

        val boxes = ArrayList<DbBox>()
        for (idxs in regionMap.values) {
            if (idxs.size < 6) continue
            var minX = Int.MAX_VALUE; var minY = Int.MAX_VALUE
            var maxX = Int.MIN_VALUE; var maxY = Int.MIN_VALUE
            var scoreSum = 0f
            val px = FloatArray(idxs.size); val py = FloatArray(idxs.size)
            var k = 0
            for (pi in idxs) {
                val y = pi / w; val x = pi - y * w
                px[k] = x.toFloat(); py[k] = y.toFloat(); k++
                if (x < minX) minX = x; if (x > maxX) maxX = x
                if (y < minY) minY = y; if (y > maxY) maxY = y
                scoreSum += prob[pi]
            }
            if (maxX - minX < 3 || maxY - minY < 3) continue
            val score = scoreSum / idxs.size
            if (score < boxThresh) continue

            val hull = convexHull(px, py)
            if (hull.isEmpty()) continue
            val rect = minAreaRect(hull) ?: continue
            val wRect = max(sideLen(rect, 0, 1), sideLen(rect, 2, 3))
            val hRect = max(sideLen(rect, 1, 2), sideLen(rect, 3, 0))
            if (wRect < 3f || hRect < 3f) continue
            val area = polygonArea(rect)
            val p = polygonPerimeter(rect)
            val dist = (unclipRatio * area / max(p, 1e-3f)).toFloat()
            val expanded = offsetPolygon(fromQuad(rect), dist)
            if (expanded == null) continue
            boxes.add(DbBox(expanded, score))
        }
        // 可选：去掉极大噪声框（面积占比过高）
        return boxes
    }

    // ── 8 连通域标注（两遍 + 并查集）──
    private fun labelComponents(bin: ByteArray, w: Int, h: Int): IntArray? {
        val n = w * h
        val labels = IntArray(n)
        val parent = IntArray(n)
        for (i in 0 until n) parent[i] = i
        var next = 1
        fun find(a: Int): Int { var x = a; while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x] }; return x }
        for (y in 0 until h) for (x in 0 until w) {
            val idx = y * w + x
            if (bin[idx] == 0.toByte()) continue
            var root = 0
            for (dy in -1..0) for (dx in -1..1) {
                if (dy == 0 && dx == 0) continue
                val nx = x + dx; val ny = y + dy
                if (nx in 0 until w && ny in 0 until h) {
                    val nidx = ny * w + nx
                    if (labels[nidx] != 0) {
                        val r = find(labels[nidx])
                        root = if (root == 0) r else { val a = find(root); val b = r; if (a != b) { parent[b] = a }; find(root) }
                    }
                }
            }
            labels[idx] = if (root == 0) next++ else root
        }
        // 压缩到 1..k
        val map = HashMap<Int, Int>()
        for (i in 0 until n) if (labels[i] != 0) {
            val r = find(labels[i])
            val v = map.getOrPut(r) { map.size + 1 }
            labels[i] = v
        }
        return labels
    }

    // ── 凸包（Andrew 单调链）→ FloatArray 坐标对 ──
    private fun convexHull(px: FloatArray, py: FloatArray): FloatArray {
        val count = px.size
        val order = (0 until count).sortedWith(compareBy({ px[it] }, { py[it] })).toIntArray()
        fun cross(o: Int, a: Int, b: Int): Float {
            val xa = px[a] - px[o]; val ya = py[a] - py[o]
            val xb = px[b] - px[o]; val yb = py[b] - py[o]
            return xa * yb - ya * xb
        }
        val lower = ArrayList<Int>(); val upper = ArrayList<Int>()
        for (i in order) { while (lower.size >= 2 && cross(lower[lower.size - 2], lower[lower.size - 1], i) <= 0) lower.removeAt(lower.size - 1); lower.add(i) }
        for (j in order.reversed()) { while (upper.size >= 2 && cross(upper[upper.size - 2], upper[upper.size - 1], j) <= 0) upper.removeAt(upper.size - 1); upper.add(j) }
        if (lower.isNotEmpty()) lower.removeAt(lower.size - 1)
        if (upper.isNotEmpty()) upper.removeAt(upper.size - 1)
        val hull = ArrayList<Int>(); hull.addAll(lower); hull.addAll(upper)
        if (hull.size < 3) return FloatArray(0)
        val out = FloatArray(hull.size * 2)
        for (i in hull.indices) { out[i * 2] = px[hull[i]]; out[i * 2 + 1] = py[hull[i]] }
        return out
    }

    /** 旋转卡尺求最小外接矩形，返回 4 角（8 值，顺时针）。 */
    private fun minAreaRect(hull: FloatArray): FloatArray? {
        val n = hull.size / 2
        if (n < 3) return null
        var bestArea = Float.MAX_VALUE
        var best = FloatArray(0)
        for (i in 0 until n) {
            val x0 = hull[i * 2]; val y0 = hull[i * 2 + 1]
            val x1 = hull[((i + 1) % n) * 2]; val y1 = hull[((i + 1) % n) * 2 + 1]
            var exx = x1 - x0; var exy = y1 - y0
            val elen = sqrt(exx * exx + exy * exy)
            if (elen < 1e-4f) continue
            exx /= elen; exy /= elen
            val eyx = -exy; val eyy = exx
            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            for (k in 0 until n) {
                val dx = hull[k * 2] - x0; val dy = hull[k * 2 + 1] - y0
                val px = dx * exx + dy * exy
                val py = dx * eyx + dy * eyy
                if (px < minX) minX = px; if (px > maxX) maxX = px
                if (py < minY) minY = py; if (py > maxY) maxY = py
            }
            val w = maxX - minX; val hh = maxY - minY
            if (w < 1e-3f || hh < 1e-3f) continue
            val area = w * hh
            if (area < bestArea) {
                bestArea = area
                val out = FloatArray(8)
                // 矩形的 4 角（局部 x∈[minX,minX+w], y∈[minY,...]）→ 世界
                for (c in 0 until 4) {
                    val lx = if (c == 0 || c == 3) minX else maxX
                    val ly = if (c < 2) minY else maxY
                    out[c * 2] = x0 + lx * exx + ly * eyx
                    out[c * 2 + 1] = y0 + lx * exy + ly * eyy
                }
                best = out
            }
        }
        return if (best.size > 0) best else null
    }

    private fun fromQuad(q: FloatArray): Array<Pair<Float, Float>> =
        Array(4) { i -> q[i * 2] to q[i * 2 + 1] }

    /** miter 外扩：每顶点沿「相邻两边向内法线之和」偏移 dist｜b·dOut｜。 */
    private fun offsetPolygon(corners: Array<Pair<Float, Float>>, dist: Float): FloatArray? {
        val n = corners.size
        if (dist <= 0f) return FloatArray(8) { i -> if (i % 2 == 0) corners[i / 2].first else corners[i / 2].second }
        val cx = corners.map { it.first }.average().toFloat()
        val cy = corners.map { it.second }.average().toFloat()
        val out = FloatArray(n * 2)
        for (i in 0 until n) {
            val prev = corners[(i + n - 1) % n]; val cur = corners[i]; val nxt = corners[(i + 1) % n]
            var dxIn = cur.first - prev.first; var dyIn = cur.second - prev.second
            var dxOut = nxt.first - cur.first; var dyOut = nxt.second - cur.second
            var il = sqrt(dxIn * dxIn + dyIn * dyIn); var ol = sqrt(dxOut * dxOut + dyOut * dyOut)
            if (il < 1e-5f) il = 1f; if (ol < 1e-5f) ol = 1f
            dxIn /= il; dyIn /= il; dxOut /= ol; dyOut /= ol
            // 内法线：垂直且朝向远离 centroid
            fun outward(nx: Float, ny: Float): Boolean {
                val midx = (cur.first + prev.first) / 2f; val midy = (cur.second + prev.second) / 2f
                return (midx - cx) * nx + (midy - cy) * ny > 0f
            }
            var nInx = -dyIn; var nIny = dxIn
            if (!outward(nInx, nIny)) { nInx = -nInx; nIny = -nIny }
            var nOutx = -dyOut; var nOuty = dxOut
            if (!outwardOf(nOutx, nOuty, cur, nxt, cx, cy)) { nOutx = -nOutx; nOuty = -nOuty }
            var bx = nInx + nOutx; var by = nIny + nOuty
            var bl = sqrt(bx * bx + by * by)
            if (bl < 1e-5f) {
                bx = cur.first - cx; by = cur.second - cy
                bl = sqrt(bx * bx + by * by)
                if (bl < 1e-5f) { out[i * 2] = cur.first; out[i * 2 + 1] = cur.second; continue }
                bx /= bl; by /= bl
            } else { bx /= bl; by /= bl }
            var mdot = bx * dxOut + by * dyOut
            if (abs(mdot) < 0.25f) mdot = if (mdot >= 0) 0.25f else -0.25f
            val off = dist / abs(mdot)
            out[i * 2] = cur.first + bx * off; out[i * 2 + 1] = cur.second + by * off
        }
        return out
    }

    private fun outwardOf(nx: Float, ny: Float, cur: Pair<Float, Float>, nxt: Pair<Float, Float>, cx: Float, cy: Float): Boolean {
        val midx = (cur.first + nxt.first) / 2f; val midy = (cur.second + nxt.second) / 2f
        return (midx - cx) * nx + (midy - cy) * ny > 0f
    }

    private fun sideLen(q: FloatArray, a: Int, b: Int): Float {
        val dx = q[a * 2] - q[b * 2]; val dy = q[a * 2 + 1] - q[b * 2 + 1]
        return sqrt(dx * dx + dy * dy)
    }

    private fun polygonArea(q: FloatArray): Float {
        var s = 0f
        for (i in 0 until 4) { val j = (i + 1) % 4; s += q[i * 2] * q[j * 2 + 1] - q[j * 2] * q[i * 2 + 1] }
        return abs(s) / 2f
    }

    private fun polygonPerimeter(q: FloatArray): Float =
        sideLen(q, 0, 1) + sideLen(q, 1, 2) + sideLen(q, 2, 3) + sideLen(q, 3, 0)
}