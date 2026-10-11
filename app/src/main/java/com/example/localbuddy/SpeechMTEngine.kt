package com.example.localbuddy

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Half
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.channels.FileChannel
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.nio.LongBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 录音翻译用 ASR / MT 推理引擎（ONNX Runtime 版本）。
 *
 *   ASR : assets/asr —— Qwen3-Audio thinker 三阶段（conv_frontend → encoder → decoder）
 *   MT  : assets/mt  —— Hunyuan-MT（DeepSeek 架构 causal LM，量化，带 KV cache）
 *
 * 模型随 APK 打包在 assets/，首次使用时整目录解包到 filesDir（ORT 需真实路径以加载
 * MT 的 external data）。加载/推理任一环节失败时回退到 [StubModels]，UI、录音、
 * 历史记录管线照常可用，不抛异常。
 *
 * ⚠️ ASR 多个参数为 best-effort 推测（mel 提取、特殊 token id、prompt 模板、KV cache
 *    scatter 语义）；详见各处注释与 [MelFeatures]。MT 路径可靠。
 */
class SpeechMTEngine private constructor(
    private val context: Context
) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    @Volatile private var asrEncoder: OrtSession? = null        // encoder.int4.onnx（mel 直入）
    @Volatile private var asrDecoderInit: OrtSession? = null    // decoder_init.int4.onnx（prefill）
    @Volatile private var asrDecoderStep: OrtSession? = null    // decoder_step.int4.onnx（逐 token）
    /** embed_tokens.bin 的 float16 权重 [EMBED_ROWS, EMBED_DIM]。用 mmap 短缓冲（native，不进 Java 堆，
     *  省 ~311MB —— 否则 311MB ShortArray 占满 ~384MB 堆导致长输入 prefill 物化 logits 时 OOM）。 */
    @Volatile private var embedBuf: ShortBuffer? = null
    @Volatile private var mtSession: OrtSession? = null
    /** MT 输出 logits 的节点名：int8 导出改名为 logits_Q8，加载时动态取含 "logits" 的输出名，兼容 logits / logits_Q8。 */
    @Volatile private var mtLogitsName: String = "logits"
    /** MT 是否接收 position_ids：int8 散装版需显式传，2bit 融合版（GroupQueryAttention 内部处理位置）不接收。加载时按图输入探测，二者自适应。 */
    @Volatile private var mtNeedsPositionIds: Boolean = false
    /** MT KV cache 是否 fp16：4bit 融合版 fp16，2bit/int8 版 fp32。加载时按 past_key_values.0.key 的 elem_type 探测，建空 past 选对应 dtype。 */
    @Volatile private var mtKvFp16: Boolean = false
    @Volatile private var mtTokenizer: HfBpeTokenizer? = null
    @Volatile private var asrTokenizer: QwenByteTokenizer? = null
    @Volatile private var warmed: Boolean = false
    /** ⚠️ 语义校准旋钮（真机验证用）：首个生成 token 绝对位置（-1=自动=L，从 prefill 之后继续）。
     *  audio_offset 已确定 = prompt 里第一个 audio_pad 的下标（见 runAsrDecoder），不再需调。 */
    @Volatile private var posStartOverride: Int = -1
    /** MT 推理线程数：翻译是瓶颈，尽量多用核（上限 8，超此小核边际递减）；ASR 用默认 4。
     *  实测 intra 4→8 提速约 27%（macOS 10 核），Android 8 核预计 10-20%。 */
    private val mtThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(4, 8)
    private val warmUpMutex = Mutex()
    /** ASR 推理串行锁：OrtSession 非线程安全，多协程并发 runAsr 会数据竞争导致结果错乱。 */
    private val asrMutex = Mutex()
    /** MT 推理串行锁：同上，translate 并发调用会污染 KV cache / 输出。 */
    private val mtMutex = Mutex()
    /** Silero VAD session（视频字幕人声检测）；多 detector 共享 session，推理串行化。 */
    @Volatile private var vadSession: OrtSession? = null
    private val vadLock = Any()

    val asrReady: Boolean get() = asrEncoder != null && asrDecoderInit != null && asrDecoderStep != null &&
        embedBuf != null && asrTokenizer != null
    val mtReady: Boolean get() = mtSession != null && mtTokenizer != null
    val vadReady: Boolean get() = vadSession != null

    /** 模型是否已完成至少一次加载（成功加载或回退到占位均算完成）。 */
    val isLoaded: Boolean get() = warmed

    /** 异步加载两个模型。失败时回退到本地占位，不抛出。多次调用幂等：仅首次真正加载，
     *  其余调用会等待首次加载完成后再返回（而非立即返回），保证上层"开始录音"时模型已就绪。 */
    suspend fun warmUp() = withContext(Dispatchers.IO) {
        if (warmed) return@withContext
        warmUpMutex.withLock {
            if (warmed) return@withLock
            // MT 分词器
            runCatching {
                mtTokenizer = HfBpeTokenizer(readAssetText("mt/tokenizer.json"))
                Log.i(TAG, "MT tokenizer loaded")
            }.onFailure { Log.w(TAG, "MT tokenizer load failed", it) }
            // MT 模型（含 external data）
            runCatching {
                val dir = extractAssetDir("mt")
                // 权重分离在 model.onnx_data（external data），createSession 自动按同名 .onnx_data 加载
                val sess = env.createSession("$dir/model.onnx", sessionOpts(mtThreads, NnapiPref.isEnabled()))
                mtSession = sess
                mtLogitsName = sess.outputNames.firstOrNull { it.contains("logits", ignoreCase = true) } ?: "logits"
                mtNeedsPositionIds = "position_ids" in sess.inputNames
                mtKvFp16 = (sess.inputInfo["past_key_values.0.key"]?.info as? TensorInfo)?.type == OnnxJavaType.FLOAT16
                Log.i(TAG, "MT ready: $dir/model.onnx (out=$mtLogitsName, pos=$mtNeedsPositionIds, kv16=$mtKvFp16)")
            }.onFailure { Log.w(TAG, "MT model load failed, fallback to stub", it) }
            // ASR 分词器 + int4 模型（encoder 直入 mel；decoder 拆 init/step，共享 decoder_weights.int4.data）
            runCatching {
                asrTokenizer = QwenByteTokenizer(readAssetText("asr/tokenizer.json"))
                val dir = extractAssetDir("asr")
                // 分文件加载并各自打日志：任一失败能精确定位是哪个（OOM/解压/ORT 拒载）
                asrEncoder = loadSession("$dir/encoder.int4.onnx")
                asrDecoderInit = loadSession("$dir/decoder_init.int4.onnx")
                asrDecoderStep = loadSession("$dir/decoder_step.int4.onnx")
                embedBuf = loadEmbedFp16(File(dir, "embed_tokens.bin"))
                    ?: throw IllegalStateException("embed_tokens.bin missing after extract")
                Log.i(TAG, "ASR ready(int4): $dir embedRows=${embedBuf!!.capacity() / EMBED_DIM}")
            }.onFailure { Log.w(TAG, "ASR model load failed, fallback to stub", it) }
            // Silero VAD（视频字幕人声检测，神经网络，对背景音/低电平鲁棒）
            runCatching {
                val dir = extractAssetDir("vad")
                val sess = env.createSession("$dir/silero_vad.onnx", sessionOpts(useNnapi = NnapiPref.isEnabled()))
                // dry-run：用真实帧长 256 跑一帧，并读取输出 output/stateN。
                // 既验证输入 shape 又验证输出名——原来只 run 不读，输出名/shape 不符也照常通过，
                // 随后每帧 isVoice() 在 runCatching{...}.getOrElse{false} 里静默失败 → 切出 0 段无日志。
                // 这里任一不符即抛 → 不启用 → 回退能量 VAD。
                val tIn = OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(256)), longArrayOf(1, 256))
                val tSt = OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(256)), longArrayOf(2, 1, 128))
                val tSr = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(16000L)), longArrayOf())
                val r = sess.run(mapOf("input" to tIn, "state" to tSt, "sr" to tSr))
                val probArr = FloatArray(1)
                (r.get("output").get() as OnnxTensor).getFloatBuffer().get(probArr)
                (r.get("stateN").get() as OnnxTensor).getFloatBuffer().get(FloatArray(2 * 1 * 128))
                r.close(); tIn.close(); tSt.close(); tSr.close()
                vadSession = sess
                Log.i(TAG, "VAD ready: $dir/silero_vad.onnx (dry-run prob=${probArr[0]})")
            }.onFailure { Log.w(TAG, "VAD model load failed, fallback to energy VAD", it) }
            warmed = true
        }
    }

    /** 32 位 float PCM，采样率 16k，单通道。返回识别文本（静音/无语音返回空串）。
     *  @param language 强制源语种（如 "zh"/"en"，对齐 qwen_asr `_build_text_prompt`），默认中文，
     *                   与录音翻译智能体一致；传 null 则不强制（模型自行输出语言行，由 [cleanAsrText] 解析）。
     *  @param onPartial token 级流式回调：decoder 每生成一个 token 就以「当前已生成文本」回调一次，
     *  调用方可据此逐字刷新 UI。注意回调在推理线程触发，需自行切回主线程更新 UI。 */
    suspend fun asr(
        pcm: FloatArray,
        language: String? = "zh",
        onPartial: ((String) -> Unit)? = null
    ): String =
        withContext(Dispatchers.Default) {
            if (pcm.isEmpty()) return@withContext ""
            if (!asrReady) return@withContext StubModels.asr(pcm)
            val raw = runAsrRobust(pcm, language, onPartial) ?: return@withContext StubModels.asr(pcm)
            cleanAsrText(raw)
        }

    /** ASR 结果：转写文本 + 自动检测到的源语种代码（如 "zh"/"en"，未知/静音为 null）。 */
    data class AsrResult(val text: String, val language: String?, val raw: String = "")

    /**
     * 多语种 ASR（language=null）。用 <asr_text> 切到纯文本转写模式，模型自行检测语种并稳定转写
     * （实测 zh/en/ja/ko 均正确；不加 <asr_text> 的自由生成模式对部分语种/片段会返回空）。
     * 适合源语种未知/多语种场景（如视频字幕）。不再返回检测到的语种——翻译源语种由 MT 自检
     * （[runMt] 指令只指定目标语种），故 [AsrResult.language] 恒为 null（仅诊断用）。静音/无语音返回 text=""。
     */
    suspend fun asrDetect(
        pcm: FloatArray,
        onPartial: ((String) -> Unit)? = null
    ): AsrResult =
        withContext(Dispatchers.Default) {
            if (pcm.isEmpty()) return@withContext AsrResult("", null)
            if (!asrReady) return@withContext AsrResult(StubModels.asr(pcm), null)
            val raw = runAsrRobust(pcm, null, onPartial) ?: return@withContext AsrResult(StubModels.asr(pcm), null)
            Log.w(TAG, "asrDetect raw=[$raw]")
            val (text, lang) = parseAsrWithLang(raw)
            // 兜底：再扫一遍，确保文本里彻底没有 "language X" 残留（防格式差异）
            val cleaned = stripLanguageLabelEverywhere(text)
            Log.w(TAG, "asrDetect parsed text=[$cleaned] lang=$lang")
            if (cleaned.isEmpty() || cleaned.lowercase() in ASR_NOISE) AsrResult("", null, raw)
            else AsrResult(cleaned, lang, raw)
        }

    /**
     * 解析自动检测模式下的 ASR 原始输出。模型可能输出 "language Chinese\n转写" 或连写
     * "language EnglishHello"（语种名与转写首词无分隔）。返回 (干净转写文本, 检测到的语种代码 or null)。
     * - 匹配开头的 "language: X" 标识，提取语种名 → 代码；保留该行标识之后的剩余文本（连写时不丢转写）；
     * - "language None"/"language Unknown" → 语种为 null（静音）。
     * 最终再过一遍 [stripLanguageLabelEverywhere] 兜底，确保无残留。
     */
    private fun parseAsrWithLang(raw: String): Pair<String, String?> {
        val kept = ArrayList<String>()
        var stripping = true
        var langCode: String? = null
        for (line in raw.split("\n")) {
            if (stripping) {
                val nameMatch = ASR_LANG_NAME.find(line)
                if (nameMatch != null) {
                    langCode = LANG_NAME_TO_CODE[nameMatch.groupValues[1].lowercase()]
                    // 取语言标识之后的剩余文本（连写时如 "language EnglishHello" → "Hello"）
                    val rest = line.substring(nameMatch.range.last + 1).trimStart(' ', ':', '：', ',', '-', '\t')
                    if (rest.isNotBlank()) {
                        kept.add(rest)
                        stripping = false          // 已进入转写，后续行不再剥离
                    }
                    continue
                }
                val prefixMatch = ASR_LANG_PREFIX.find(line)
                if (prefixMatch != null) {
                    // 形如 "language: None" 等无声种名 → 静音标识；同样保留其后剩余（一般无）
                    val rest = line.substring(prefixMatch.range.last + 1).trimStart(' ', ':', '：', ',', '-', '\t')
                    if (rest.isNotBlank()) {
                        kept.add(rest)
                        stripping = false
                    }
                    continue
                }
                if (line.isNotBlank()) stripping = false
            }
            if (line.isNotBlank()) kept.add(line)
        }
        return stripLanguageLabelEverywhere(kept.joinToString("\n").trim()) to langCode
    }


    /**
     * 清洗 ASR 原始输出。Qwen3-Audio ASR 默认在转写前先输出一行语言识别（如
     * "language: Chinese" / "language None"），仅静音时整段就是它。这里：
     *  - 剥离开头的 "language: X" 前缀（同行若带译文则保留后半）；
     *  - 仅剩 None/null/unknown 或空 → 视为静音/无语音，返回空串，让上层 `if (asr.isBlank()) continue`
     *    跳过该段，不再把 "None" 当转写显示。
     */
    /** 清洗 ASR 原始输出（强制语言模式用）。
     *  强制语言时模型直接吐纯转写，一般无语言行；但仍兜底删除任意位置的 "language X" 残留，
     *  并判静音（仅剩 None/空 → 返回空串，让上层跳过该段）。 */
    private fun cleanAsrText(raw: String): String {
        val out = stripLanguageLabelEverywhere(raw)
        if (out.isEmpty() || out.lowercase() in ASR_NOISE) return ""
        return out
    }

    /** 旧版行首剥离，保留给 partial 强制模式用（强制模式输出本就无语言行，此处基本 no-op）。 */
    private fun stripLangPrefix(raw: String): String = stripLanguageLabelEverywhere(raw)

    /**
     * 从文本中删除任意位置的 "language <语种名>" / "language: None" 等语言标识片段。
     * 不依赖行首、不依赖换行——把所有匹配 ASR_LANG_PREFIX 的片段整段删掉，再清理多余空白/空行。
     * 这是 partial 与 final 的统一兜底清洗，确保 UI 与烧录文本里绝不出现 "language English" 之类残留。
     */
    private fun stripLanguageLabelEverywhere(raw: String): String {
        var s = ASR_LANG_PREFIX.replace(raw, "")
        // 剥离拼错的语言标识（模型偶发把 "language" 输出成 "Lingage"/"Langage"…，正则匹配
        // 不上而整段漏进转写）：首词与 "language" 编辑距离 ≤2 视为标识词，连同其后可能的
        // 语种名（含连写，如 "Lingage EnglishHello" → "Hello"）一并剥离。
        while (true) {
            val t = s.trimStart()
            val firstWord = t.takeWhile { it.isLetter() }
            if (firstWord.length >= 5 && levenshtein(firstWord.lowercase(), "language") <= 2) {
                var rest = t.substring(firstWord.length).trimStart()
                val secondWord = rest.takeWhile { it.isLetter() }
                val lw = secondWord.lowercase()
                // 语种名前缀匹配（不用整词：模型常与转写首词连写，同 ASR_LANG_NAME 去掉 \b 的理由）
                val matched = LANG_EN_NAMES.firstOrNull { lw.startsWith(it) && it.length >= 3 }
                if (matched != null) rest = rest.substring(matched.length).trimStart()
                s = rest
            } else break
        }
        // 剥离开头的 chat 模板回声（空白音频时模型回吐 "User"/"Assistant" 模板 token，后常
        // 跟换行或直接连写真实转写如 "User你好"）。仅当整词后跟非拉丁字母（换行/中文/标点/
        // 结束）才剥离，避免误伤正常英文句首 "User interface ..."。
        while (true) {
            val t = s.trimStart()
            val m = TEMPLATE_ECHO.find(t) ?: break
            val rest = t.substring(m.value.length)
            val first = rest.trimStart().firstOrNull()
            if (first == null || !(first in 'a'..'z' || first in 'A'..'Z')) {
                s = rest
            } else break
        }
        // 删掉后可能留下行首多余空格/空行，规整一下
        s = s.split("\n").map { it.trim() }.filter { it.isNotEmpty() }.joinToString("\n")
        return s.trim()
    }

    /** Levenshtein 编辑距离（小串，O(n²) 足够）。 */
    private fun levenshtein(a: String, b: String): Int {
        if (a == b) return 0
        val dp = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            var prev = dp[0]; dp[0] = i
            for (j in 1..b.length) {
                val tmp = dp[j]
                dp[j] = minOf(dp[j] + 1, dp[j - 1] + 1, prev + if (a[i - 1] == b[j - 1]) 0 else 1)
                prev = tmp
            }
        }
        return dp[b.length]
    }

    /**
     * 自动检测模式下，流式 partial 用的宽松剥离：把开头以 "language"/"lang" 起始的整行视为
     * 语言标识行直接丢弃（含不完整态，如 "language"、"language English"）。首个非标识行起照原样保留。
     *
     * 为什么需要它：[stripLangPrefix] 的正则要求 "language" 后跟完整语种名（chinese/english…）才匹配，
     * 但流式期间 token 尚未生成完，会出现 "language"、"language English"（无换行）等中间态，正则匹配不上
     * 就会被原样显示。这里只要行以 language/lang 开头就丢弃，从根上避免语言标识闪现。
     * 安全性：自动检测模式下语言标识行总是输出在最前面，真实转写在其后；故丢弃首个 language 开头行不会误伤转写。
     */
    private fun stripLeadingLangLine(raw: String): String {
        val kept = ArrayList<String>()
        var stripping = true
        for (line in raw.split("\n")) {
            if (stripping) {
                if (LANG_LINE_PREFIX.containsMatchIn(line)) continue   // 语言标识行（含不完整）→ 丢弃
                if (line.isNotBlank()) stripping = false
            }
            if (line.isNotBlank()) kept.add(line)
        }
        // 兜底：再扫一遍删任意位置的 language 片段（防转写中途也出现）
        return stripLanguageLabelEverywhere(kept.joinToString("\n").trim())
    }

    /** 把 src（srcLang 语言码）翻译到 targetLang。
     *  @param onPartial token 级流式回调：每生成一个 token 就以「当前已生成译文」回调一次，
     *                   调用方据此实时刷新译文区；回调在推理线程触发，需自行切回主线程更新 UI（状态流为线程安全）。 */
    suspend fun translate(
        src: String,
        srcLang: String,
        targetLang: String,
        onPartial: ((String) -> Unit)? = null
    ): String =
        withContext(Dispatchers.Default) {
            if (src.isBlank()) return@withContext ""
            if (!mtReady) return@withContext StubModels.translate(src, targetLang)
            // OrtSession 非线程安全：串行化 MT 推理，避免并发 runMt 污染 KV cache / 输出错乱
            mtMutex.withLock {
                runCatching { runMt(src, srcLang, targetLang, onPartial) }.getOrElse {
                    Log.e(TAG, "MT inference failed, falling back to stub", it)
                    StubModels.translate(src, targetLang)
                }
            }
        }

    /** 创建一个独立 state 的 Silero VAD 检测器（每条音频流一个）。模型未加载返回 null，调用方回退能量 VAD。 */
    fun createVadDetector(): VadDetector? {
        val s = vadSession ?: return null
        return SileroVadDetector(env, s, vadLock)
    }

    fun close() {
        runCatching { asrEncoder?.close() }
        runCatching { asrDecoderInit?.close() }
        runCatching { asrDecoderStep?.close() }
        runCatching { mtSession?.close() }
        runCatching { vadSession?.close() }
        asrEncoder = null; asrDecoderInit = null; asrDecoderStep = null
        embedBuf = null; mtSession = null; vadSession = null
    }

    /** 重建所有 OrtSession（切换 NNAPI 等运行时配置后调用）。先 close 再 warmUp；期间正在跑的
     *  推理会失败并回退占位，故应在空闲时调用。不持 warmUpMutex（warmUp 内部自锁，避免重入死锁）。 */
    suspend fun reload() = withContext(Dispatchers.IO) {
        close()
        warmed = false
        warmUp()
    }

    // ──────────────────────────────────────────────────────────────
    // MT (Hunyuan-MT) 推理
    // 标准 causal LM：prefill(空 past) -> 贪心自回归（past/present KV cache 追加）-> EOS 停
    // ──────────────────────────────────────────────────────────────
    private fun runMt(
        src: String,
        srcLang: String,
        targetLang: String,
        onPartial: ((String) -> Unit)? = null
    ): String {
        val tok = mtTokenizer!!
        val sess = mtSession!!
        val srcName = LANG_NAMES[srcLang] ?: srcLang
        val tgtName = LANG_NAMES[targetLang] ?: targetLang
        val instruction = "将以下文本翻译为$tgtName，注意只要翻译后结果，不要额外解释：$src"
        val promptIds = tok.encode(instruction)

        // prompt token 序列：BOS, User, <指令+原文>, Assistant
        val seq = ArrayList<Long>(promptIds.size + 4).apply {
            add(tok.bosId.toLong())
            add(tok.userId.toLong())
            promptIds.forEach { add(it.toLong()) }
            add(tok.assistantId.toLong())
        }

        val maxNew = 96   // 翻译上限：短句译文极少超过 96 token，原 256 偏大、模型偶发跑飞时浪费算力
        val generated = ArrayList<Int>()
        var nextInput: List<Long> = seq
        var posStart = 0L              // 已喂入 token 数 = past 长度
        var prevResult: OrtSession.Result? = null
        var past: List<Pair<OnnxTensor, OnnxTensor>> = emptyList()
        try {
            while (generated.size < maxNew) {
                val curLen = nextInput.size
                val inputs = HashMap<String, OnnxTensor>()
                val owned = ArrayList<OnnxTensor>()
                inputs["input_ids"] = longTensor(nextInput.toLongArray()).also { owned.add(it) }
                val total = posStart + curLen
                inputs["attention_mask"] = longTensor(LongArray(total.toInt()) { 1L }).also { owned.add(it) }
                // position_ids：int8 散装版需显式传（prefill=[0..curLen-1]，decode=[posStart]）；
                //   2bit 融合版（GroupQueryAttention 内部 cumsum 派生位置）不接收此输入，传了会被 ORT 拒。
                if (mtNeedsPositionIds) {
                    inputs["position_ids"] = longTensor(LongArray(curLen) { posStart + it }).also { owned.add(it) }
                }
                if (past.isEmpty()) {
                    // prefill：空 past [1,4,0,128]。
                    // ⚠️ int8 cache 版 KV cache 为 FLOAT32（elem_type=1），与旧 int4 的 FLOAT16 不同：
                    //   用 FloatBuffer 建（三参数 createTensor 自动 FLOAT）；present 亦 FLOAT32，decode 直接回传。
                    for (i in 0 until MT_N_LAYERS) {
                        val k = createEmptyKv()
                        val v = createEmptyKv()
                        inputs["past_key_values.$i.key"] = k
                        inputs["past_key_values.$i.value"] = v
                        owned.add(k); owned.add(v)
                    }
                } else {
                    past.forEachIndexed { i, (k, v) ->
                        inputs["past_key_values.$i.key"] = k
                        inputs["past_key_values.$i.value"] = v
                    }
                }

                val result = sess.run(inputs)
                val logits = readLastLogits(result.get(mtLogitsName).get() as OnnxTensor, curLen)
                val newPast = (0 until MT_N_LAYERS).map {
                    (result.get("present.$it.key").get() as OnnxTensor) to
                        (result.get("present.$it.value").get() as OnnxTensor)
                }
                // 本步自建输入可释放；上一步 Result（持有 borrowed past）也释放
                owned.forEach { runCatching { it.close() } }
                prevResult?.close()
                prevResult = result
                past = newPast

                // 重复惩罚（generation_config 配 1.05，此处 1.1 抑制更强）：对已生成 token 的 logits
                //   降权，避免低 bit 模型贪心解码陷入字重复；penalty 温和，不影响正常译文用词。
                if (generated.isNotEmpty()) applyRepPenaltyInPlace(logits, generated)
                val nextId = argmax(logits)
                if (nextId == tok.eosId) break
                generated.add(nextId)
                // token 级流式：逐 token 把「已生成译文」推给上层实时刷新
                if (onPartial != null) runCatching { onPartial(tok.decode(generated)) }
                nextInput = listOf(nextId.toLong())
                posStart += curLen
            }
        } finally {
            prevResult?.close()
        }
        val out = tok.decode(generated)
        Log.d(TAG, "MT instruction=${instruction.replace("\n", "\\n")}")
        Log.d(TAG, "MT generatedIds(${generated.size})=" + generated.take(40))
        Log.d(TAG, "MT decoded=[$out]")
        return out
    }
    // ──────────────────────────────────────────────────────────────
    /**
     * 抗噪统一入口（三个智能体共用）：**先做 [AudioEnhancer] 前置增强再推理**，并在
     * 「识别为空 + 判定为低 SNR」时用**强降噪档重试一次**。
     *
     * 为什么需要重试：噪声下最常见失败不是"识别错字"，而是模型直接吐空/只吐 "language None"
     * （声学特征被噪声淹没、语言检测也失败）。此时首轮的温和降噪力度不够，换 STRONG 档（更大
     * 过估计因子 + 压掉人声带外）常能把人声结构捞回来。仅在首轮为空时触发，常态零额外开销。
     *
     * @return 模型 raw 输出；null=推理异常（调用方回退占位）。
     */
    private suspend fun runAsrRobust(
        pcm: FloatArray,
        language: String?,
        onPartial: ((String) -> Unit)?
    ): String? {
        val capped = capAudio(pcm)
        val (mild, profile) = AudioEnhancer.prepare(capped, AudioEnhancer.Mode.AUTO)
        val r1 = lockedAsr(mild, language, onPartial)
        if (!isSilenceLike(r1) || !profile.needDenoise) return r1
        Log.i(TAG, "ASR empty on noisy input (snr=${"%.1f".format(profile.snrDb)}dB), retry with STRONG denoise")
        val (strong, _) = AudioEnhancer.prepare(capped, AudioEnhancer.Mode.STRONG)
        return lockedAsr(strong, language, onPartial) ?: r1
    }

    /** 串行化 ASR 推理（OrtSession 非线程安全）；异常返回 null 让上层回退。 */
    private suspend fun lockedAsr(
        pcm: FloatArray,
        language: String?,
        onPartial: ((String) -> Unit)?
    ): String? = asrMutex.withLock {
        runCatching { runAsr(pcm, language, onPartial) }.getOrElse {
            Log.e(TAG, "ASR inference failed", it)
            null
        }
    }

    /** 模型输出是否"什么都没识别出来"（空 / 只剩语言标识 / 只剩 None 之类占位）。 */
    private fun isSilenceLike(raw: String?): Boolean {
        if (raw == null) return true
        val t = stripLanguageLabelEverywhere(raw)
        return t.isEmpty() || t.lowercase() in ASR_NOISE
    }

    /** 输入长度上限截断（12s）：避免 decoder prefill 物化过大 logits。 */
    private fun capAudio(pcm: FloatArray): FloatArray =
        if (pcm.size > MAX_AUDIO_SAMPLES) {
            Log.w(TAG, "audio truncated to ${MAX_AUDIO_SAMPLES / MelFeatures.SAMPLE_RATE}s")
            pcm.copyOf(MAX_AUDIO_SAMPLES)
        } else pcm

    private fun runAsr(
        pcm: FloatArray,
        language: String?,
        onPartial: ((String) -> Unit)? = null
    ): String {
        val tok = asrTokenizer!!
        // 限制音频长度，避免 prefill 爆内存
        val capped = capAudio(pcm)
        val (mel, nFrames) = MelFeatures.compute(capped)
        if (nFrames == 0) return ""

        // 1) mel [nFrames,128] 行优先 → 转置成 encoder 需要的 [1,128,nFrames]（通道在前）
        val melC = FloatArray(MelFeatures.N_MELS * nFrames)
        for (t in 0 until nFrames)
            for (b in 0 until MelFeatures.N_MELS)
                melC[b * nFrames + t] = mel[t * MelFeatures.N_MELS + b]
        val melT = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(melC),
            longArrayOf(1, MelFeatures.N_MELS.toLong(), nFrames.toLong())
        )

        // 2) encoder.int4（自含 conv_frontend）：mel → audio_features [1, T', 1024]
        val encRes = asrEncoder!!.run(mapOf("mel" to melT))
        melT.close()
        val audioFeat = encRes.get("audio_features").get() as OnnxTensor

        // 3) split decoder：decoder_init(prefill) + decoder_step(逐 token)
        val text = runCatching { runAsrDecoder(audioFeat, tok, onPartial, language) }
        encRes.close()                  // audio_features 释放
        return text.getOrElse { throw it }
    }

    private fun runAsrDecoder(
        audioFeat: OnnxTensor, tok: QwenByteTokenizer,
        onPartial: ((String) -> Unit)? = null,
        language: String? = "zh"        // 默认强制中文，消除 "language" 前缀；null=不强制
    ): String {
        val decInit = asrDecoderInit!!
        val decStep = asrDecoderStep!!

        // prompt 模板（Qwen3-Audio chat template；token id 与 tokenizer.json 对齐）：
        //   <|im_start|>system\n{context}<|im_end|>\n
        //   <|im_start|>user\n<|audio_start|>{audio_pad × nAudio}<|audio_end|><|im_end|>\n
        //   <|im_start|>assistant\n{language {Lang}<asr_text>}
        //   —— prefill 时 decoder_init 把 audio_pad 占位与 audio_features 对齐：模型把 audio_features 行
        //   scatter 进序列 [audio_offset .. audio_offset+nAudio) 列，audio_offset 必须 = 第一个
        //   <|audio_pad|> 的序列下标；错位则音频特征覆写 prompt 头部 → 纯乱码。
        val nAudio = audioFeat.info.getShape()[1].toInt()   // audio_features [1, T', 1024]
        val prefix = ArrayList<Long>().apply {
            add(tok.imStartId.toLong())
            addAll(tok.encode("system\n").map { it.toLong() })
            add(tok.imEndId.toLong())
            addAll(tok.encode("\n").map { it.toLong() })
            add(tok.imStartId.toLong())
            addAll(tok.encode("user\n").map { it.toLong() })
            add(tok.audioStartId.toLong())                         // <|audio_start|>
        }
        val padStart = prefix.size.toLong()                          // 第一个 audio_pad 的下标
        val ids = prefix.also { id ->
            repeat(nAudio) { id.add(tok.audioPlaceholderId.toLong()) } // <|audio_pad|> × nAudio
            id.add(tok.audioEndId.toLong())                          // <|audio_end|>
            id.add(tok.imEndId.toLong())
            id.addAll(tok.encode("\n").map { it.toLong() })
            id.add(tok.imStartId.toLong())
            id.addAll(tok.encode("assistant\n").map { it.toLong() })
            // <asr_text> 切纯文本转写（始终加）；强制语言时先补 "language {Lang}" 消除语言前缀。
            // ⚠️ 必须用英文名（"language English"）：模型自己的语言行输出就是英文名格式
            // （"language Chinese"/"language English"），中文名（"language 英语"）模型不认，
            // 短音频会退回默认中文偏置。LANG_NAMES 的中文名只用于 MT 指令。
            val langName = language?.let { ASR_LANG_EN[it] ?: it }
            if (langName != null) id.addAll(tok.encode("language $langName").map { it.toLong() })
            id.add(tok.asrTextId.toLong())                           // <asr_text>
        }
        val L = ids.size
        val maxNew = 96   // 转写上限：10s 段快语速可能超 64 token（byte 分词 1~3 tok/字）被截断，96 更稳；模型遇 im_end/ASR_EOS 即停，上限仅兜底

        // 1) prefill：decoder_init(input_ids, position_ids, audio_features, audio_offset) -> logits + KV
        val inputIds = longTensor(ids.toLongArray())
        val positions = longTensor(LongArray(L) { it.toLong() })    // 文本位置 0..L-1
        val offsetT = longTensor1d(longArrayOf(padStart))
        val initRes = decInit.run(
            mapOf("input_ids" to inputIds, "position_ids" to positions,
                "audio_features" to audioFeat, "audio_offset" to offsetT))
        inputIds.close(); positions.close()   // audioFeat 由调用方（runAsr）持有
        var logits = readLastLogits(initRes.get("logits").get() as OnnxTensor, L)
        var kvK = initRes.get("present_keys").get() as OnnxTensor
        var kvV = initRes.get("present_values").get() as OnnxTensor

        // 2) 逐 token：embed_tokens.bin 取行 fp16→fp32 -> decoder_step(input_embeds, position_ids, past)
        val generated = ArrayList<Int>()
        var prevRes = initRes
        var pos = if (posStartOverride >= 0) posStartOverride else L   // 首个生成 token 绝对位置
        try {
            while (generated.size < maxNew) {
                if (generated.isNotEmpty()) applyRepPenaltyInPlace(logits, generated)
                val nextId = argmax(logits)
                if (nextId == tok.imEndId || nextId == ASR_EOS) break
                generated.add(nextId)
                // token 级流式：逐字刷新 UI（自动检测模式宽松剥离语言行，强制模式直接）
                if (onPartial != null && generated.isNotEmpty()) {
                    val decoded = tok.decode(generated)
                    val partial = if (language == null) stripLeadingLangLine(decoded) else stripLangPrefix(decoded)
                    runCatching { onPartial(partial) }
                }
                val emb = embedRow(nextId)
                val embT = OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), longArrayOf(1, 1, EMBED_DIM.toLong()))
                // decoder_step 的 position_ids 要 rank2 [1,1]（长 tensor 1D 会 ORT_INVALID_ARGUMENT）
                val posT = longTensor(longArrayOf(pos.toLong()))
                val stepRes = decStep.run(
                    mapOf("input_embeds" to embT, "position_ids" to posT,
                        "past_keys" to kvK, "past_values" to kvV))
                embT.close(); posT.close()
                prevRes.close()              // past 已被本步消费，可释放上一步 Result
                prevRes = stepRes
                kvK = stepRes.get("present_keys").get() as OnnxTensor
                kvV = stepRes.get("present_values").get() as OnnxTensor
                logits = readLastLogits(stepRes.get("logits").get() as OnnxTensor, 1)
                pos++
            }
        } finally { prevRes.close() }
        val out = tok.decode(generated)
        Log.d(TAG, "ASR nAudio=$nAudio promptLen=$L generated(${generated.size})=" + generated.take(40))
        Log.d(TAG, "ASR decoded=[$out]")
        return out
    }

    /** 取 embed_tokens.bin（fp16）某行并转成 fp32 输入向量。用 duplicate 只读一行，不复制整表。 */
    private fun embedRow(id: Int): FloatArray {
        val b = embedBuf ?: return FloatArray(EMBED_DIM)
        val row = ShortArray(EMBED_DIM)
        val v = b.duplicate()
        v.position(id * EMBED_DIM)
        v.get(row)
        return FloatArray(EMBED_DIM) { Half.toFloat(row[it]) }
    }

    /** embed_tokens.bin（fp16）mmap 成只读 ShortBuffer：native 内存，不进 Java 堆（避免 311MB 占满
     *  ~384MB 堆导致长输入 prefill 物化 logits 时 OOM）。close channel 不影响已 mmap 的 buffer。 */
    private fun loadEmbedFp16(file: File): ShortBuffer? {
        if (!file.exists()) { Log.w(TAG, "embed_tokens.bin not found"); return null }
        return try {
            FileInputStream(file).channel.use { fc ->
                fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size())
                    .order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            }
        } catch (e: Exception) {
            Log.w(TAG, "embed_tokens.bin load failed", e); null
        }
    }

    /** 建 session 前先打加载日志（文件即外在 data 所在 ONNX）。ONNX 会按同目录 external_data 自动载入。 */
    private fun loadSession(path: String): OrtSession {
        return try {
            val s = env.createSession(path, sessionOpts(useNnapi = NnapiPref.isEnabled()))
            Log.i(TAG, "session OK: ${File(path).name}")
            s
        } catch (e: Exception) {
            Log.w(TAG, "session FAIL: ${File(path).name}", e)
            throw e
        }
    }

    /** 读取 logits [1, seqLen, V] 的最后一行（V 个 float），用 getFloatBuffer 避免整张物化。 */
    private fun readLastLogits(tensor: OnnxTensor, seqLen: Int): FloatArray {
        val fb = tensor.getFloatBuffer()
        val total = fb.remaining()
        if (seqLen <= 0 || total == 0) return FloatArray(0)
        val v = total / seqLen
        fb.position((seqLen - 1) * v)
        val row = FloatArray(v)
        fb.get(row)
        return row
    }

    private fun argmax(a: FloatArray): Int {
        var bi = 0
        var bv = Float.NEGATIVE_INFINITY
        for (i in a.indices) {
            if (a[i] > bv) { bv = a[i]; bi = i }
        }
        return bi
    }

    /** 对已生成 token 的 logits 施加重复惩罚（HF 风格：正值除以 penalty，负值乘 penalty）。原地修改。 */
    private fun applyRepPenaltyInPlace(logits: FloatArray, ids: List<Int>, penalty: Float = REP_PENALTY) {
        val seen = HashSet<Int>()
        for (id in ids) {
            if (id in 0 until logits.size && seen.add(id)) {
                val v = logits[id]
                logits[id] = if (v > 0) v / penalty else v * penalty
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 张量构造
    // ──────────────────────────────────────────────────────────────
    private fun longTensor(arr: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, LongBuffer.wrap(arr), longArrayOf(1, arr.size.toLong()))

    private fun longTensor1d(arr: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, LongBuffer.wrap(arr), longArrayOf(arr.size.toLong()))

    /** 建空 KV cache [1,4,0,128]：fp16 模型用 ShortBuffer+FLOAT16（四参数 createTensor），fp32 用 FloatBuffer（三参数）。 */
    private fun createEmptyKv(): OnnxTensor =
        if (mtKvFp16) OnnxTensor.createTensor(env, ShortBuffer.allocate(0), longArrayOf(1, 4, 0, 128), OnnxJavaType.FLOAT16)
        else OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1, 4, 0, 128))

    /** 构建 session 选项。intra 为线程数：MT（翻译瓶颈）多用线程、独占 CPU 时加速；ASR/VAD 用默认 4。
     *  ASR(4)+MT(mtThreads) 并发时略超核数，但 MT 优先吃核、ASR 让步可接受。 */
    private fun sessionOpts(intra: Int = 4, useNnapi: Boolean = false): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(intra)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // NNAPI（Android NPU/DSP）：开启时优先注册，支持算子走 NPU 可能大幅加速；不支持/失败
            //  的算子自动回退下方 XNNPACK/CPU。GQA 融合算子/dynamic shape 支持参差，可能无效甚至更慢
            //  或精度异常，故默认关、可开关（NnapiPref）。须在 XNNPACK 前注册以优先用 NPU。
            if (useNnapi) {
                runCatching { addNnapi() }
                    .onFailure { Log.w(TAG, "NNAPI EP unavailable, fallback to XNNPACK/CPU", it) }
            }
            // XNNPACK 执行提供者：移动端 ARM CPU 矩阵运算高度优化，显著加速 conv/encoder/decoder/MT
            // 的算子（int8 卷积/矩阵乘尤甚）。不支持的算子自动回退 CPU，安全；开启
            // 失败则继续用纯 CPU。线程数与 intra-op 一致。
            runCatching { addXnnpack(mapOf("intra_op_num_threads" to intra.toString())) }
                .onFailure { Log.w(TAG, "XNNPACK EP unavailable, fallback to CPU", it) }
        }

    // ──────────────────────────────────────────────────────────────
    // 资产解包
    // ──────────────────────────────────────────────────────────────
    private fun readAssetText(name: String): String =
        context.assets.open(name).bufferedReader(Charsets.UTF_8).use { it.readText() }

    /** 把 assets/<dir> 下所有文件解包到 filesDir/<dir>/，返回该目录绝对路径。 */
    private fun extractAssetDir(dir: String): String {
        val outDir = File(context.filesDir, dir)
        val names = context.assets.list(dir) ?: emptyArray()
        if (names.isEmpty()) {
            if (!outDir.exists()) outDir.mkdirs()
            return outDir.absolutePath
        }
        if (!outDir.exists()) outDir.mkdirs()
        // 缓存按「文件大小」校验：同名但内容变化的资产（如迭代中的模型权重）会重新解包，避免复用过期
        // 副本导致 ONNX 图与权重错配（图改了但 .data 还是旧的 -> 加载失败或输出乱码）。
        // 清理已不在 assets 里的残留文件（本目录仅由此函数写入，删除非当前资产安全，如旧 MT 的 model_q4.* / 旧版单文件 model.onnx）
        val nameSet = names.toSet()
        outDir.listFiles()?.forEach { f ->
            if (f.isFile && f.name !in nameSet) runCatching { f.delete() }
        }
        for (name in names) {
            val cached = File(outDir, name)
            val assetSize = assetUncompressedSize("$dir/$name")  // -1 = 压缩资产拿不到 fd，退化为存在性校验
            if (cached.exists() && cached.length() > 0L &&
                (assetSize < 0L || cached.length() == assetSize)) continue
            Log.i(TAG, "extracting $dir/$name → ${cached.absolutePath}")
            context.assets.open("$dir/$name").use { input ->
                FileOutputStream(cached).use { output -> input.copyTo(output) }
            }
        }
        return outDir.absolutePath
    }

    /** 取 assets 下某文件的未压缩大小。模型大文件（onnx/onnx_data/data/json）均在 aaptOptions
     *  noCompress 中，openFd 可拿到真实大小用于缓存校验；压缩资产（如 .jinja）openFd 抛
     *  FileNotFoundException，返回 -1 由调用方退化为「存在即跳过」。 */
    private fun assetUncompressedSize(name: String): Long =
        try {
            context.assets.openFd(name).use { it.length }
        } catch (e: java.io.FileNotFoundException) {
            -1L
        } catch (e: Exception) {
            -1L
        }

    companion object {
        private const val TAG = "SpeechMTEngine"
        private const val SAMPLE_RATE = 16_000

        /** MT(Hunyuan-MT) 层数；与模型图 past_key_values.*.key 数量一致。 */
        private const val MT_N_LAYERS = 32
        /** MT 贪心解码重复惩罚（generation_config 为 1.05，实测 1.1 抑制更好且不影响译文）。 */
        private const val REP_PENALTY = 1.1f
        /** ASR 输入嵌入维度（embed_tokens.bin 行宽 = audio_features 最后一维）。 */
        private const val EMBED_DIM = 1024
        /** decoder_step 停止 token：<|endoftext|>（151643），与 <|im_end|>(151645) 并列判停。 */
        private const val ASR_EOS = 151643
        /** 限制输入音频长度，避免 decoder prefill 物化过大 logits。 */
        private const val MAX_AUDIO_SAMPLES = SAMPLE_RATE * 12

        private val LANG_NAMES = mapOf(
            "zh" to "中文", "en" to "英语", "ja" to "日语", "ko" to "韩语",
            "fr" to "法语", "es" to "西班牙语", "de" to "德语", "ru" to "俄语",
            "pt" to "葡萄牙语", "it" to "意大利语", "hu" to "匈牙利语", "fa" to "波斯语",
            "ar" to "阿拉伯语", "pl" to "波兰语", "cs" to "捷克语", "da" to "丹麦语",
            "sv" to "瑞典语", "el" to "希腊语", "tr" to "土耳其语",
            // 语音识别 30 语种中余下的（[asrLangs] 全量覆盖）
            "yue" to "粤语", "id" to "印尼语", "th" to "泰语", "vi" to "越南语",
            "hi" to "印地语", "ms" to "马来语", "nl" to "荷兰语", "fi" to "芬兰语",
            "fil" to "菲律宾语", "mk" to "马其顿语", "ro" to "罗马尼亚语"
        )

        /** Qwen3-Audio ASR 输出的语言标识前缀（如 "language Chinese"/"language: English"/"language None"）。
         *  语种名取自 assets/asr/config.json 的 support_languages。⚠️ 不用 \b 词边界：模型常把语种名与
         *  转写首词连写（如 "language EnglishHello"），此时 "English" 后无词边界，\b 会匹配失败导致整段残留。
         *  去掉 \b 后正则匹配到 "language English" 为止，后续转写文本保留。 */
        /** 22 种中国方言的英文名（官方 README 命名），自动检测模式下模型可能直接吐方言名。 */
        private const val ASR_DIALECT_NAMES =
            "anhui|dongbei|fujian|gansu|guizhou|hebei|henan|hubei|hunan|jiangxi|ningxia|" +
                "shandong|shaanxi|shanxi|sichuan|tianjin|yunnan|zhejiang|wu|minnan"

        private val ASR_LANG_PREFIX = Regex(
            "(?i)\\s*(?:language|lang)\\s*[:：]?\\s*" +
                "(?:none|null|unknown|nan|nil|chinese|english|cantonese|arabic|german|french|spanish|" +
                "portuguese|indonesian|italian|korean|russian|thai|vietnamese|japanese|turkish|" +
                "hindi|malay|dutch|swedish|danish|finnish|polish|czech|filipino|persian|greek|" +
                "romanian|hungarian|macedonian|$ASR_DIALECT_NAMES)"
        )

        /** 捕获 ASR 语言标识里的语种名（第 1 组），用于自动检测语种。同样不用 \b（见上）。 */
        private val ASR_LANG_NAME = Regex(
            "(?i)\\s*(?:language|lang)\\s*[:：]?\\s*" +
                "(chinese|english|cantonese|arabic|german|french|spanish|portuguese|indonesian|" +
                "italian|korean|russian|thai|vietnamese|japanese|turkish|hindi|malay|dutch|" +
                "swedish|danish|finnish|polish|czech|filipino|persian|greek|romanian|hungarian|" +
                "macedonian|$ASR_DIALECT_NAMES)"
        )

        /** 宽松匹配：以 language/lang 开头的行（含不完整态）。供流式 partial 剥离语言标识行用。 */
        private val LANG_LINE_PREFIX = Regex("(?i)^\\s*(?:language|lang)\\b.*")

        /** ASR 输出的语种英文名 → 语言码（传给 MT 作 srcLang）。 */
        private val LANG_NAME_TO_CODE = mapOf(
            "chinese" to "zh", "cantonese" to "zh", "english" to "en", "japanese" to "ja",
            "korean" to "ko", "french" to "fr", "spanish" to "es", "german" to "de",
            "russian" to "ru", "portuguese" to "pt", "arabic" to "ar", "indonesian" to "id",
            "italian" to "it", "thai" to "th", "vietnamese" to "vi", "turkish" to "tr",
            "hindi" to "hi", "malay" to "ms", "dutch" to "nl", "swedish" to "sv",
            "danish" to "da", "finnish" to "fi", "polish" to "pl", "czech" to "cs",
            "filipino" to "fil", "persian" to "fa", "greek" to "el", "romanian" to "ro",
            "hungarian" to "hu", "macedonian" to "mk",
            // 方言：统一归到中文（粤语口音归到粤语码），供 MT 作 srcLang
            "anhui" to "zh", "dongbei" to "zh", "fujian" to "zh", "gansu" to "zh",
            "guizhou" to "zh", "hebei" to "zh", "henan" to "zh", "hubei" to "zh",
            "hunan" to "zh", "jiangxi" to "zh", "ningxia" to "zh", "shandong" to "zh",
            "shaanxi" to "zh", "shanxi" to "zh", "sichuan" to "zh", "tianjin" to "zh",
            "yunnan" to "zh", "zhejiang" to "zh", "wu" to "zh", "minnan" to "zh"
        )

        /** 剥离语言标识后，若仅剩这些占位则视为静音/无语音。
         *  user/assistant：空白/极短音频时模型回吐 chat 模板 token（整段只有 "User"）。 */
        private val ASR_NOISE = setOf("none", "null", "unknown", "nan", "nil", "user", "assistant")

        /** 开头的 chat 模板回声（"User" / "Assistant:" 等），见 [stripLanguageLabelEverywhere]。 */
        private val TEMPLATE_ECHO = Regex("(?i)^(?:user|assistant)\\s*[:：]?\\s*")

        /**
         * ASR 强制语种注入用的英文名（模型语言行的自有格式），见 [runAsrDecoder]。
         * 覆盖 Qwen3-ASR 官方 30 语种（[asrLangs]）。
         * 方言（[asrDialects]）：官方 qwen-asr 的强制语种参数只接受这 30 种语言（见其
         * SUPPORTED_LANGUAGES 校验），方言不在可强制列表内，故按归口语种注入——
         * 两种粤语口音 → Cantonese，其余方言 → Chinese；方言本身由模型自带 LID 识别。
         */
        private val ASR_LANG_EN = mapOf(
            "zh" to "Chinese", "en" to "English", "ja" to "Japanese", "ko" to "Korean",
            "de" to "German", "es" to "Spanish", "fr" to "French", "it" to "Italian",
            "hu" to "Hungarian", "ru" to "Russian", "fa" to "Persian", "ar" to "Arabic",
            "pl" to "Polish", "pt" to "Portuguese", "cs" to "Czech", "da" to "Danish",
            "sv" to "Swedish", "el" to "Greek", "tr" to "Turkish",
            "yue" to "Cantonese", "id" to "Indonesian", "th" to "Thai", "vi" to "Vietnamese",
            "hi" to "Hindi", "ms" to "Malay", "nl" to "Dutch", "fi" to "Finnish",
            "fil" to "Filipino", "mk" to "Macedonian", "ro" to "Romanian",
            // 方言（码见 [asrDialects]）→ 归口语种
            "anhui" to "Chinese", "dongbei" to "Chinese", "fujian" to "Chinese",
            "gansu" to "Chinese", "guizhou" to "Chinese", "hebei" to "Chinese",
            "henan" to "Chinese", "hubei" to "Chinese", "hunan" to "Chinese",
            "jiangxi" to "Chinese", "ningxia" to "Chinese", "shandong" to "Chinese",
            "shaanxi" to "Chinese", "shanxi" to "Chinese", "sichuan" to "Chinese",
            "tianjin" to "Chinese", "yunnan" to "Chinese", "zhejiang" to "Chinese",
            "yue-hk" to "Cantonese", "yue-gd" to "Cantonese",
            "wu" to "Chinese", "minnan" to "Chinese"
        )

        /** 英文语种名小写集（含 ASR 全部支持名 + 方言名），供拼错标识的剥离匹配。 */
        private val LANG_EN_NAMES = (ASR_LANG_EN.values +
            ASR_DIALECT_NAMES.split("|") + listOf("none", "null", "unknown", "cantonese")
            ).map { it.lowercase() }

        @Volatile private var instance: SpeechMTEngine? = null

        fun get(context: Context): SpeechMTEngine {
            return instance ?: synchronized(this) {
                instance ?: SpeechMTEngine(context.applicationContext).also { instance = it }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 离线占位：ONNX 未加载 / 推理失败时让上层 UI 可用
    // ──────────────────────────────────────────────────────────────
    private object StubModels {
        fun asr(pcm: FloatArray): String {
            val seconds = pcm.size / 16_000.0
            val energy = pcm.map { it * it }.average()
            val rms = sqrt(max(energy, 0.0))
            return when {
                rms < 0.005 -> ""
                seconds < 0.4 -> "(短音频)"
                else -> "（ONNX 模型未加载，占位识别 ${"%.1f".format(seconds)}s 段，rms=${"%.3f".format(rms)}）"
            }
        }

        fun translate(src: String, targetLang: String): String {
            // MT 未就绪/推理失败时不伪造译文：旧实现 "[$targetLang] $src" 会把语种码当译文显示
            // （"翻译结果"里出现 "[en] ..."），且不真正翻译、把原文当译文。返回空串，
            // 让上层对空译文不渲染翻译卡片。
            if (src.isBlank()) return ""
            return ""
        }
    }
}
