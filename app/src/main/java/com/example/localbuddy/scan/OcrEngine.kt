package com.example.localbuddy.scan

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.localbuddy.NnapiPref
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.round
import kotlin.math.max
import kotlin.math.min

/**
 * PP-OCRv5 轻量 OCR 引擎（检测 det + 识别 rec，多语言 rec 分模型，复用 onnxruntime）。
 *
 * 管线：det(通用文本块检测) → DB 后处理 → 每块按脚本路由到对应 rec 识别 → CTC 解码 → 阅读顺序排版。
 *  - det:   `PP-OCRv5_mobile_det_onnx`，输入 [N,3,H,W]（resize_long=960，(v/255-mean)/std, BGR）；
 *  - rec:   多语言模型按目录懒加载（见 [OcrLang]），输入 [N,3,48,W]（BGR，/255），CTC 解码。
 * 对外 API（warmUp / runOcr / reload / ready / get）保持不变，仅新增 [setRecModel] 用于切换识别语言。
 */
class OcrEngine private constructor(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    @Volatile private var detSession: OrtSession? = null
    private var detCfg = OcrDetConfig()
    @Volatile private var ready: Boolean = false
    @Volatile private var recSessions = HashMap<String, OrtSession>()
    private var recConfigs = HashMap<String, OcrRecModel>()
    private var recDirs: List<String> = emptyList()
    @Volatile private var recDir: String = OcrLang.DEFAULT_DIR

    /** 切换识别语言（rec 模型目录名）；null/空 恢复默认「中文/英文/日文」。 */
    fun setRecModel(dir: String?) {
        val d = if (dir.isNullOrBlank()) OcrLang.DEFAULT_DIR else dir
        if (recDirs.contains(d) || d == OcrLang.DEFAULT_DIR) recDir = d
    }

    /** 模型是否已加载就绪（warmUp 成功后为 true；runOcr 未就绪时静默返回空串）。 */
    val isReady: Boolean get() = ready

    val recModelLabel: String get() = OcrLang.label(recDir)

    suspend fun warmUp() = withContext(Dispatchers.IO) {
        if (ready) return@withContext
        runCatching {
            val rootDir = extractModels("ocr")      // filesDir/ocr/<modeldir>/...
            val so = sessionOpts(NnapiPref.isEnabled())

            // det
            val detYml = readAssetText("ocr/PP-OCRv5_mobile_det_onnx/inference.yml")
            val (dMean, dStd) = PpocrConfig.extractDetNorm(detYml)
            detCfg = OcrDetConfig(
                mResizeLong = PpocrConfig.extractResizeLong(detYml),
                mMean = dMean, mStd = dStd,
                mThresh = PpocrConfig.extractPostFloat(detYml, "thresh", 0.3),
                mBoxThresh = PpocrConfig.extractPostFloat(detYml, "box_thresh", 0.6),
                mUnclip = PpocrConfig.extractPostFloat(detYml, "unclip_ratio", 1.5),
            )
            detSession = env.createSession("$rootDir/PP-OCRv5_mobile_det_onnx/inference.onnx", so)

            // rec：枚举目录并预解析配置（session 懒加载）
            recDirs = OcrLang.MODELS.map { it.dir }.filter { File(rootDir, it).exists() }
            for (dir in recDirs) {
                val yml = readAssetText("ocr/$dir/inference.yml")
                val (_, rh, rw) = PpocrConfig.extractRecShape(yml)
                recConfigs[dir] = OcrRecModel(dir, PpocrConfig.extractDict(yml), rh, rw)
            }
            ready = (detSession != null)
            diag("warmUp: ready=$ready detResize=${detCfg.mResizeLong} recDirs=$recDirs recConfigs=${recConfigs.keys} " +
                "defaultDict=${recConfigs[OcrLang.DEFAULT_DIR]?.dict?.size} recDir=$recDir")
            Log.i(TAG, "OCR(PaddleOCRv5) ready. rec=${recDirs.size} models, det resize=${detCfg.mResizeLong}")
        }.onFailure { diag("warmUp failed: ${it.javaClass.simpleName}: ${it.message}"); Log.w(TAG, "OCR load failed", it); ready = false }
    }

    /**
     * 对一张图做 OCR，返回识别文本。模型未就绪返回空串。
     * [onPartial]：每识别完一行回调一次已排版的部分文本，供 UI 流式展示。
     */
    fun runOcr(bitmap: Bitmap, onPartial: ((String) -> Unit)? = null): String {
        if (!ready) { diag("runOcr: not ready"); return "" }
        val ds = detSession ?: return ""
        return runCatching { runOcrImpl(bitmap, ds, onPartial) }
            .onFailure { diag("runOcr failed: ${it.javaClass.simpleName}: ${it.message}"); Log.e(TAG, "OCR inference failed", it) }
            .getOrElse { "" }
    }

    /** 诊断日志：同时写 logcat 与 filesDir/ocr_debug.log（HONOR 设备 logcat 被系统加密，只能读文件排查）。超 512KB 截断重开。 */
    private fun diag(msg: String) {
        Log.d(TAG, msg)
        runCatching {
            val f = File(context.filesDir, "ocr_debug.log")
            if (f.length() > 512 * 1024) f.delete()
            f.appendText("[${timeNow()}] $msg\n")
        }
    }

    private fun timeNow(): String =
        java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.US).format(java.util.Date())

    private fun runOcrImpl(bitmap: Bitmap, ds: OrtSession, onPartial: ((String) -> Unit)?): String {
        val bw = bitmap.width; val bh = bitmap.height

        // 1) det 预处理 + 推理 → 概率图
        val (detInput, dw, dh) = detPreprocess(bitmap)
        val inTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(detInput), longArrayOf(1, 3, dh.toLong(), dw.toLong()))
        val res = try { ds.run(mapOf("x" to inTensor)) } finally { inTensor.close() }
        val probTensor = res.get("fetch_name_0").get() as OnnxTensor
        // ONNX 输出维度是动态的，info.shape 可能含 -1；用 det 输入尺寸兜底（本 det 为全分辨率 1:1）。
        val shape = probTensor.info.shape
        var mapW = if (shape.size >= 4 && shape[3] > 0) shape[3].toInt() else -1
        var mapH = if (shape.size >= 4 && shape[2] > 0) shape[2].toInt() else -1
        if (mapW <= 0 || mapH <= 0) { mapW = dw; mapH = dh }
        val prob = FloatArray(mapH * mapW).also { probTensor.floatBuffer.get(it) }
        probTensor.close(); res.close()

        // 2) DB 后处理 → 检测框（map 坐标）,再缩回原图坐标
        val detCfgLoc = detCfg
        val boxesMap = DbPostprocess.run(prob, mapW, mapH,
            detCfgLoc.mThresh, detCfgLoc.mBoxThresh, detCfgLoc.mUnclip)
        val sx = bw.toFloat() / mapW; val sy = bh.toFloat() / mapH
        val boxes = boxesMap.map { it.scaled(sx, sy) }
        val probMax = if (prob.isNotEmpty()) prob.max() else -1f
        diag("det: in=${dw}x${dh} map=${mapW}x${mapH} probMax=$probMax thresh=${detCfgLoc.mThresh} boxes=${boxes.size}")

        // 3) 每框选 rec 模型识别
        val recModel = recConfigs[recDir] ?: recConfigs[OcrLang.DEFAULT_DIR] ?: run {
            diag("rec: no config for dir=$recDir, recConfigs=${recConfigs.keys}"); return ""
        }
        val recSession = getRecSession(recModel.dir) ?: run {
            diag("rec: session load failed dir=${recModel.dir}"); return ""
        }
        val out = ArrayList<Pair<DbBox, String>>()
        var blankRec = 0
        for (box in boxes) {
            // 管线：透视矫正(倾斜回正) -> 行切分(整段框拆行) -> 宽行切段(词边界)
            // 段拼接：lineStart 前置换行（分段）、gapBefore 前置空格（英文词间）
            val segs = OcrRecDecoder.buildRecTensors(bitmap, box.quad, recModel.recH, recModel.maxW)
            val sb = StringBuilder()
            for (s in segs) {
                val part = runRec(recSession, recModel, s.nchw, s.w, s.h)
                if (sb.isNotEmpty()) {
                    when { s.lineStart -> sb.append('\n'); s.gapBefore -> sb.append(' ') }
                }
                sb.append(part)
            }
            val t = sb.toString().replace(Regex(" {2,}"), " ").trim()
            if (t.isBlank()) blankRec++ else out.add(box to t)
            onPartial?.invoke(OcrRecDecoder.layout(out))
        }
        if (blankRec > 0) diag("rec: $blankRec/${boxes.size} boxes decoded blank")

        // 4) 阅读顺序排版
        val text = OcrRecDecoder.layout(out)
        diag("ocr done: textLen=${text.length} sample=${text.take(30)}")
        Log.d(TAG, "OCR: ${boxes.size} boxes, ${text.length} chars")
        return text
    }

    private fun runRec(sess: OrtSession, model: OcrRecModel, nchw: FloatArray, w: Int, h: Int): String {
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(nchw), longArrayOf(1, 3, h.toLong(), w.toLong()))
        val res = try { sess.run(mapOf("x" to tensor)) } finally { tensor.close() }
        val outT = res.get("fetch_name_0").get() as OnnxTensor
        val c = model.dict.size + 2                            // blank + end
        val total = outT.floatBuffer.remaining()
        val t = if (c > 0) total / c else 0                    // 时间步(宽)
        if (t <= 0 || total <= 0) diag("runRec: empty output w=$w h=$h total=$total c=$c")
        val logits = FloatArray(t * c).also { outT.floatBuffer.get(it) }
        outT.close(); res.close()
        return OcrRecDecoder.ctcDecode(logits, t, c, model.dict)
    }

    /** 获取（懒加载 + 缓存）某个 rec 模型的 session。 */
    @Synchronized
    private fun getRecSession(dir: String): OrtSession? {
        recSessions[dir]?.let { return it }
        val cfg = recConfigs[dir] ?: return null
        return runCatching {
            val rootDir = File(context.filesDir, "ocr")
            val sess = env.createSession("$rootDir/$dir/inference.onnx", sessionOpts(NnapiPref.isEnabled()))
            recSessions[dir] = sess
            Log.i(TAG, "loaded rec model $dir (dict=${cfg.dict.size})")
            sess
        }.getOrNull()
    }

    // ── det 预处理：resize_long=960 → BGR → (v/255-mean)/std → NCHW ──
    private fun detPreprocess(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        val w = bitmap.width; val h = bitmap.height
        val longer = max(w, h)
        val scale = if (longer > 960) 960.0 / longer else 1.0
        val dw = max(32, (round(w * scale / 32) * 32).toInt())
        val dh = max(32, (round(h * scale / 32) * 32).toInt())
        val scaled = Bitmap.createScaledBitmap(bitmap, dw, dh, true)
        val px = IntArray(dw * dh)
        scaled.getPixels(px, 0, dw, 0, 0, dw, dh)
        if (scaled !== bitmap) scaled.recycle()
        val mean = detCfg.mMean; val std = detCfg.mStd
        val out = FloatArray(3 * dh * dw)
        for (y in 0 until dh) for (x in 0 until dw) {
            val argb = px[y * dw + x]
            val b = ((argb) and 0xFF) / 255.0
            val g = ((argb shr 8) and 0xFF) / 255.0
            val r = ((argb shr 16) and 0xFF) / 255.0
            out[(0 * dh + y) * dw + x] = ((b - mean[0]) / std[0]).toFloat()
            out[(1 * dh + y) * dw + x] = ((g - mean[1]) / std[1]).toFloat()
            out[(2 * dh + y) * dw + x] = ((r - mean[2]) / std[2]).toFloat()
        }
        return Triple(out, dw, dh)
    }

    private data class OcrDetConfig(val mResizeLong: Int = 960,
                                    val mMean: List<Double> = listOf(0.485, 0.456, 0.406),
                                    val mStd: List<Double> = listOf(0.229, 0.224, 0.225),
                                    val mThresh: Double = 0.3,
                                    val mBoxThresh: Double = 0.6,
                                    val mUnclip: Double = 1.5)

    /** 重建所有 session（切换 NNAPI 后调用）。关闭现有后再 warmUp。 */
    suspend fun reload() = withContext(Dispatchers.IO) {
        runCatching { detSession?.close() }
        recSessions.values.forEach { runCatching { it.close() } }
        detSession = null; recSessions.clear()
        ready = false
        warmUp()
    }

    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(4, 8)

    private fun sessionOpts(useNnapi: Boolean = false): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (useNnapi) {
            runCatching { addNnapi() }.onFailure { Log.w(TAG, "NNAPI EP unavailable, fallback", it) }
        }
        // ⚠️ 不注册 XNNPACK：HONOR Android 16（ARM MTE 开启）真机上 XNNPACK 内核在 rec 推理
        // （动态宽度输入）触发 SIGSEGV（SEGV_ACCERR，MTE 标签违规）。纯 CPU EP 兼容性最好。
        // 如需恢复加速，先在 MTE 设备上验证 rec 不崩。
    }

    // ── 资产：递归取每个模型目录的 inference.onnx + inference.yml ──
    private fun extractModels(dir: String): String {
        val outDir = File(context.filesDir, dir)
        if (!outDir.exists()) outDir.mkdirs()
        val subdirs = context.assets.list(dir)?.filter { it.endsWith("_onnx") } ?: emptyList()
        for (sub in subdirs) {
            val target = File(outDir, sub)
            if (!target.exists()) target.mkdirs()
            for (f in listOf("inference.onnx", "inference.yml")) {
                copyAssetIfChanged("$dir/$sub/$f", File(target, f))
            }
        }
        return outDir.absolutePath
    }

    private fun copyAssetIfChanged(asset: String, dst: File) {
        if (dst.exists() && dst.length() > 0L) return   // 已缓存
        context.assets.open(asset).use { input ->
            FileOutputStream(dst).use { output -> input.copyTo(output) }
        }
    }

    private fun readAssetText(name: String): String =
        context.assets.open(name).bufferedReader(Charsets.UTF_8).use { it.readText() }

    companion object {
        private const val TAG = "OcrEngine"
        @Volatile private var instance: OcrEngine? = null
        fun get(context: Context): OcrEngine =
            instance ?: synchronized(this) { instance ?: OcrEngine(context.applicationContext).also { instance = it } }
    }
}