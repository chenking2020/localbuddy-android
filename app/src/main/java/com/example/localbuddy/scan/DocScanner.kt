package com.example.localbuddy.scan

import android.graphics.Bitmap
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * 文档扫描（OpenCV）：Canny 边缘检测 -> 最大四边形轮廓 -> 透视矫正 -> 灰度增强/二值化。
 *
 * 检测失败（背景复杂/无明确边缘）时回退为原图增强（不透视），保证总有输出；
 * 后续可加"手动选四点"fallback 提升复杂场景成功率。
 *
 * ⚠️ 使用前必须先调 [ensureInitialized]（OpenCV native 加载）。
 */
object DocScanner {

    /**
     * 扫描样式：对标 WPS / 全能扫描王的模式集。
     * [label] 展示名、[hint] 一句话说明、[category] 用于「更多」弹窗分组。
     * 顶部常用 3 个由 [Mode.COMMON] 引用，其余进「更多」。
     */
    enum class Mode(val label: String, val hint: String, val category: Category) {
        DOCUMENT("文档", "合同、报告、通用文件", Category.DOCUMENT),
        BUSINESS_CARD("名片", "姓名电话一键结构化", Category.DOCUMENT),
        RECEIPT("发票票据", "高对比保留金额税号", Category.FINANCE),
        ID_CARD("证件", "身份证/驾照/护照", Category.IDENTITY),
        TABLE("表格", "保留网格、行列对齐", Category.DOCUMENT),
        BOOK("书籍双页", "去书脊、跨页平整", Category.DOCUMENT),
        WHITEBOARD("白板", "去反光阴影拉直", Category.ENHANCE),
        PHOTO("照片修复", "去噪提亮还原", Category.ENHANCE);

        enum class Category(val label: String) {
            DOCUMENT("常用文档"),
            FINANCE("票据"),
            IDENTITY("证件"),
            ENHANCE("效果增强")
        }

        companion object {
            /** 顶部平铺的常用 3 个（最常用：文档 / 发票 / 证件）；其余模式进「更多」。 */
            val COMMON = listOf(DOCUMENT, RECEIPT, ID_CARD)
        }

        /** 是否为「更多」内的非常用模式。 */
        val isExtra: Boolean get() = this !in COMMON
    }

    /** 初始化 OpenCV native 库。返回是否成功/已加载。 */
    fun ensureInitialized(): Boolean = OpenCVLoader.initLocal()

    /** 扫描一张图：自动检测文档四边形并透视矫正 + 增强。
     *  [color] = true 时保留彩色（逐通道直方图拉伸），false 时按 [mode] 灰度/二值化（默认黑白）。 */
    fun scan(bitmap: Bitmap, mode: Mode = Mode.DOCUMENT, color: Boolean = false): Bitmap {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)  // RGBA
        val quad = findDocumentQuad(src)
        val warped = if (quad != null) fourPointTransform(src, quad) else src.clone()
        val enhanced = enhance(warped, mode, color)
        val out = Bitmap.createBitmap(enhanced.cols(), enhanced.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(enhanced, out)
        src.release(); warped.release(); enhanced.release()
        return out
    }

    /** 找文档四边形：最大面积的 4 顶点近似轮廓。 */
    private fun findDocumentQuad(src: Mat): MatOfPoint2f? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(gray, gray, 75.0, 200.0)
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(gray, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        contours.sortByDescending { Imgproc.contourArea(it) }
        var result: MatOfPoint2f? = null
        for (c in contours) {
            val curve = MatOfPoint2f(*c.toArray())
            val peri = Imgproc.arcLength(curve, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(curve, approx, 0.02 * peri, true)
            if (approx.total() == 4L && Imgproc.contourArea(approx) > src.total() * 0.1) {
                result = approx; break
            }
            approx.release()
        }
        gray.release()
        return result
    }

    /** 四点排序：左上、右上、右下、左下。 */
    private fun orderPoints(pts: List<Point>): List<Point> {
        val bySum = pts.sortedBy { it.x + it.y }
        val tl = bySum.first()
        val br = bySum.last()
        val mid = bySum.drop(1).dropLast(1).sortedBy { it.y - it.x }
        val tr = mid.first()
        val bl = mid.last()
        return listOf(tl, tr, br, bl)
    }

    /** 透视变换：把四边形区域矫正为矩形。 */
    private fun fourPointTransform(src: Mat, quad: MatOfPoint2f): Mat {
        val (tl, tr, br, bl) = orderPoints(quad.toList())
        val width = maxOf(dist(tl, tr), dist(bl, br)).toInt().coerceAtLeast(1)
        val height = maxOf(dist(tl, bl), dist(tr, br)).toInt().coerceAtLeast(1)
        val srcPts = MatOfPoint2f(tl, tr, br, bl)
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0), Point(width.toDouble(), 0.0),
            Point(width.toDouble(), height.toDouble()), Point(0.0, height.toDouble())
        )
        val m = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val out = Mat()
        Imgproc.warpPerspective(src, out, m, Size(width.toDouble(), height.toDouble()))
        return out
    }

    private fun dist(a: Point, b: Point) = Math.hypot(a.x - b.x, a.y - b.y)

    /**
     * 按样式增强：文档/白板用自适应阈值二值化（去阴影反光）、表格用 Otsu（保留网格线）、
     * 名片/书籍锐化文字、发票/证件 CLAHE 提对比、照片修复去噪提亮。
     * 输出统一为 GRAY2RGBA，与既有管线保持一致。
     */
    private fun enhance(mat: Mat, mode: Mode, color: Boolean = false): Mat {
        if (color) return enhanceColor(mat)
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, if (mat.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_RGB2GRAY)
        val processed = when (mode) {
            Mode.DOCUMENT -> adaptiveBinary(gray, blockSize = 21, c = 10.0)
            Mode.WHITEBOARD -> adaptiveBinary(gray, blockSize = 15, c = 8.0)
            Mode.TABLE -> otsuBinary(gray)
            Mode.RECEIPT -> clahe(gray)
            Mode.ID_CARD -> clahe(gray)
            Mode.BUSINESS_CARD -> unsharp(clahe(gray))
            Mode.BOOK -> unsharp(clahe(gray))
            Mode.PHOTO -> normalizeStretch(bilateralDenoise(gray))
        }
        val out = Mat()
        Imgproc.cvtColor(processed, out, Imgproc.COLOR_GRAY2RGBA)
        processed.release()
        gray.release()
        return out
    }

    /** 彩色增强：①灰世界自动白平衡（各通道均值对齐，消除黄/蓝灯偏色）→ ②各通道直方图拉伸提对比，保留色彩。 */
    private fun enhanceColor(mat: Mat): Mat {
        val rgb = Mat()
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_RGBA2RGB)
        val ch = ArrayList<Mat>()
        Core.split(rgb, ch)
        // 1) 灰世界白平衡：三通道均值归一，偏色（如暖黄灯光）据此校正
        val means = DoubleArray(3) { Core.mean(ch[it]).`val`[0] }
        val target = (means[0] + means[1] + means[2]) / 3.0
        for (i in 0 until 3) {
            if (means[i] > 1.0 && target > 1.0) {
                Core.multiply(ch[i], Scalar(target / means[i]), ch[i])
            }
        }
        // 2) 各通道直方图拉伸（auto-levels）提对比
        for (i in 0 until 3) {
            val mm = Core.minMaxLoc(ch[i])
            val lo = mm.minVal; val hi = mm.maxVal
            val range = (hi - lo).coerceAtLeast(1e-6)
            val out = Mat()
            ch[i].convertTo(out, CvType.CV_8U, 255.0 / range, -lo * (255.0 / range))
            ch[i].release(); ch[i] = out
        }
        val merged = Mat()
        Core.merge(ch, merged)
        val out = Mat()
        Imgproc.cvtColor(merged, out, Imgproc.COLOR_RGB2RGBA)
        ch.forEach { runCatching { it.release() } }
        rgb.release(); merged.release()
        return out
    }

    private fun adaptiveBinary(gray: Mat, blockSize: Int, c: Double): Mat {
        // 阈值前的轻量中值去噪：抑制传感器颗粒，避免被自适应阈值放大成背景黑点/断笔
        val denoised = Mat()
        Imgproc.medianBlur(gray, denoised, 3)
        val bin = Mat()
        Imgproc.adaptiveThreshold(denoised, bin, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, c)
        denoised.release()
        cleanupBinary(bin)
        return bin
    }

    /** Otsu 全局阈值：网格线/边界比自适应阈值保留更完整。 */
    private fun otsuBinary(gray: Mat): Mat {
        val bin = Mat()
        Imgproc.threshold(gray, bin, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        cleanupBinary(bin)
        return bin
    }

    /**
     * 形态学清理二值结果：先开运算去掉背景孤立黑点（盐噪），再小闭运算补全笔画断口、
     * 让文字边缘更实，缓解"发糊/发虚"。
     */
    private fun cleanupBinary(bin: Mat) {
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, Size(3.0, 3.0))
        val opened = Mat()
        Imgproc.morphologyEx(bin, opened, Imgproc.MORPH_OPEN, kernel)  // 去除背景孤立黑点
        Imgproc.morphologyEx(opened, bin, Imgproc.MORPH_CLOSE, kernel) // 补全断笔，边缘粘实
        opened.release()
        kernel.release()
    }

    /** CLAHE 局部对比度增强：保留灰度细节（发票/证件）。 */
    private fun clahe(gray: Mat): Mat {
        val out = Mat()
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0)).apply(gray, out)
        return out
    }

    /** 高斯模糊差分锐化：强化文字边缘（名片/书籍）。 */
    private fun unsharp(mat: Mat): Mat {
        val blur = Mat()
        Imgproc.GaussianBlur(mat, blur, Size(0.0, 0.0), 3.0)
        val sharp = Mat()
        Core.addWeighted(mat, 1.6, blur, -0.6, 0.0, sharp)
        blur.release()
        return sharp
    }

    /** 双边滤波去噪：保边降噪，柔和旧照片颗粒。 */
    private fun bilateralDenoise(mat: Mat): Mat {
        val out = Mat()
        Imgproc.bilateralFilter(mat, out, 7, 50.0, 50.0)
        return out
    }

    /** 直方图拉伸：把像素值铺满 [0,255]，提亮/去灰蒙。 */
    private fun normalizeStretch(mat: Mat): Mat {
        val out = Mat()
        val mm = Core.minMaxLoc(mat)
        val lo = mm.minVal
        val hi = mm.maxVal
        val range = (hi - lo).coerceAtLeast(1e-6)
        mat.convertTo(out, CvType.CV_8U, 255.0 / range, -lo * (255.0 / range))
        return out
    }
}
