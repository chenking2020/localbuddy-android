package com.example.localbuddy

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.Random
import kotlin.coroutines.coroutineContext

/**
 * MOSS-TTS-Nano-100M ONNX 语音合成引擎（assets/tts + assets/tts/codec）。
 *
 * 架构 = Audio Tokenizer + LLM 纯自回归：全局 transformer（12 层，prefill/decode_step 图，
 * KV cache）逐行消费 17 列输入（[文本 token, 16 路音频码]），每帧由局部 transformer 图
 * （moss_tts_local_fixed_sampled_frame，内含 16 通道步 + top-k/top-p/重复惩罚采样，
 * 采样随机数显式注入）产出 16 个音频码；音频码流经 codec（MOSS-Audio-Tokenizer-Nano，
 * decode_full 图）解码为 48kHz 立体声波形。
 *
 * 提示协议（与 browser_poc_manifest.json 对齐，已在 Python 原型上端到端验证）：
 *  user 行: user_prompt_prefix + <|audio_start|> + 音色参考帧(文本列=user_slot) +
 *           <|audio_end|> + user_prompt_after_reference + 文本 token + assistant_prompt_prefix
 *  生成行:  <|audio_start|> 文本行 → 每帧 [assistant_slot, 16 码]（decode_step 喂回），
 *           should_continue=0 即自然收尾。
 *
 * 音色 = 零样本克隆：用说话人自己的录音做音色参考——录音 PCM 经 codec encode 图编码成
 * 音频码帧，替换提示中的参考帧（与内置音色同格式；16k 单声道升采样 48k 后编码，已实测
 * 跨语种克隆可用：中文音色说英语/德语自然合成）。
 */
class TtsEngine private constructor(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var tokenizer: SpBpeTokenizer? = null
    @Volatile private var warmed = false
    private val warmUpMutex = Mutex()
    /** 合成串行锁：OrtSession 非线程安全。 */
    private val inferMutex = Mutex()

    private var prefill: OrtSession? = null
    private var decodeStep: OrtSession? = null
    private var frameGraph: OrtSession? = null
    private var codecDecode: OrtSession? = null
    private var codecEncode: OrtSession? = null
    private var manifest: JSONObject? = null

    val isLoaded: Boolean get() = warmed && prefill != null && codecDecode != null && codecEncode != null
    /** 内置音色合成（[synthesize]）的就绪条件：不需要 codecEncode（参考帧已预编码）。 */
    val canSpeak: Boolean get() = warmed && prefill != null && decodeStep != null &&
        frameGraph != null && codecDecode != null && manifest != null

    /**
     * 异步加载（幂等）。模型目录较大（~780MB），打不进 APK（assets 总量会超 Zip32 4GB 上限），
     * 按以下顺序定位模型（ORT 需真实路径，external data 与 onnx 同目录）：
     *  1. filesDir/tts —— 之前解包/拷贝过的缓存；
     *  2. app 专属外部目录（getExternalFilesDir/tts）—— adb push 侧载，原地直用不拷贝；
     *  3. assets/tts —— 兜底（若某构建重新打包了）→ 解包到 filesDir。
     */
    suspend fun warmUp() = withContext(Dispatchers.IO) {
        if (warmed) return@withContext
        warmUpMutex.withLock {
            if (warmed) return@withLock
            runCatching {
                val mf = locateModelDir() ?: run {
                    Log.w(TAG, "TTS 模型未找到：请将 tts 模型目录 push 到 " +
                        "${File(context.getExternalFilesDir(null), "tts")}（app 专属外部目录，无需存储权限）")
                    return@runCatching
                }
                val codecDir = File(mf, "codec")
                manifest = JSONObject(File(mf, "browser_poc_manifest.json").readText())
                tokenizer = SpBpeTokenizer(File(mf, "tokenizer.model").readBytes())
                prefill = createSession(File(mf, "moss_tts_prefill.onnx"))
                decodeStep = createSession(File(mf, "moss_tts_decode_step.onnx"))
                frameGraph = createSession(File(mf, "moss_tts_local_fixed_sampled_frame.onnx"))
                codecDecode = createSession(File(codecDir, "moss_audio_tokenizer_decode_full.onnx"))
                codecEncode = createSession(File(codecDir, "moss_audio_tokenizer_encode.onnx"))
                Log.i(TAG, "TTS ready: $mf")
            }.onFailure { Log.w(TAG, "TTS model load failed", it) }
            warmed = true
        }
    }

    /** 判断目录是否含 TTS 全套文件。 */
    private fun isModelDir(dir: File): Boolean =
        dir.isDirectory &&
            File(dir, "moss_tts_prefill.onnx").exists() &&
            File(dir, "browser_poc_manifest.json").exists() &&
            File(dir, "tokenizer.model").exists() &&
            File(dir, "codec/moss_audio_tokenizer_decode_full.onnx").exists() &&
            File(dir, "codec/moss_audio_tokenizer_encode.onnx").exists()

    private fun locateModelDir(): File? {
        val cached = File(context.filesDir, "tts")
        if (isModelDir(cached)) return cached
        // app 专属外部目录（/sdcard/Android/data/<pkg>/files/tts）—— adb push 侧载点
        context.getExternalFilesDir(null)?.let {
            val ext = File(it, "tts")
            if (isModelDir(ext)) return ext
        }
        // assets 兜底（当前构建不含 tts，留作以后小模型打包）
        if (!context.assets.list("").orEmpty().contains("tts")) return null
        val mf = extractAssetDir("tts")
        return if (isModelDir(File(mf))) File(mf) else null
    }

    /** 合成结果：48kHz 立体声 PCM（左右声道交错）。 */
    class TtsAudio(val pcmInterleaved: FloatArray, val sampleRate: Int = 48_000) {
        val durationSec: Float get() = pcmInterleaved.size / 2f / sampleRate
    }

    /**
     * 文本 → 语音，**音色克隆自说话人自己的录音**：[refPcm16k]（16k 单声道，本段发言）
     * 升采样 48k 后经 codec encode 编码成音频码帧，作为 TTS 提示的音色参考帧。
     * 流式逐帧生成（每帧 80ms 音频），全程持锁串行；[onFrame] 每帧回调（已生成帧数）。
     */
    suspend fun synthesizeCloned(
        text: String,
        refPcm16k: FloatArray,
        maxFrames: Int = 375,
        onFrame: ((Int) -> Unit)? = null
    ): TtsAudio? = withContext(Dispatchers.Default) {
        if (!isLoaded) return@withContext null
        val mf = manifest ?: return@withContext null
        if (refPcm16k.size < 16_000) return@withContext null   // 参考音频 <1s 无克隆意义
        inferMutex.withLock {
            runCatching {
                val refCodes = encodeRef16k(refPcm16k) ?: return@runCatching null
                runSynthesis(mf, text, refCodes, maxFrames, onFrame)
            }.getOrElse {
                Log.e(TAG, "synthesize failed", it)
                null
            }
        }
    }

    /** 内置音色：manifest 里预编码好的音色提示帧（[voiceIndex] 对应 [builtinVoices] 下标）。 */
    data class BuiltinVoice(val name: String, val displayName: String, val group: String)

    /** 内置音色列表（18 个：中/英/日男女声）。TTS 未加载时返回空。 */
    fun builtinVoices(): List<BuiltinVoice> {
        val mf = manifest ?: return emptyList()
        return runCatching {
            val arr = mf.getJSONArray("builtin_voices")
            List(arr.length()) { i ->
                val o = arr.getJSONObject(i)
                val name = o.optString("voice", "voice$i")
                BuiltinVoice(name, o.optString("display_name", name), o.optString("group", ""))
            }
        }.getOrDefault(emptyList())
    }

    /**
     * 文本 → 语音（**内置音色**，无需用户录音）。
     *
     * 与 [synthesizeCloned] 只有音色参考帧的来源不同：这里直接取 manifest 中
     * [builtinVoices] 第 [voiceIndex] 个预编码的 `prompt_audio_codes`（已是 [frames][16] 的
     * 音频码，与 codec encode 输出同格式），省掉「录音 → 升采样 → encode」一整条链路，
     * 因此不依赖 codecEncode，首句延迟也更低。
     */
    suspend fun synthesize(
        text: String,
        voiceIndex: Int = 0,
        maxFrames: Int = DEFAULT_MAX_FRAMES,
        onFrame: ((Int) -> Unit)? = null
    ): TtsAudio? = withContext(Dispatchers.Default) {
        if (!canSpeak || text.isBlank()) return@withContext null
        val mf = manifest ?: return@withContext null
        val refCodes = builtinRefCodes(mf, voiceIndex) ?: return@withContext null
        inferMutex.withLock {
            runCatching { runSynthesis(mf, text, refCodes, maxFrames, onFrame) }
                .getOrElse { Log.e(TAG, "synthesize failed", it); null }
        }
    }

    /** 取内置音色的参考码帧 [frames][16]（长度不足 16 列的帧按 0 补，避免越界）。 */
    private fun builtinRefCodes(mf: JSONObject, voiceIndex: Int): Array<IntArray>? {
        val arr = mf.optJSONArray("builtin_voices") ?: return null
        if (arr.length() == 0) return null
        val v = arr.getJSONObject(voiceIndex.coerceIn(0, arr.length() - 1))
        val codes = v.optJSONArray("prompt_audio_codes") ?: return null
        val frames = codes.length()
        if (frames <= 0) return null
        return Array(frames) { f ->
            val row = codes.optJSONArray(f) ?: return Array(0) { IntArray(0) }
            IntArray(N_VQ) { c -> row.optInt(c, 0) }
        }
    }

    /**
     * 16k 单声道 PCM → 音色参考码帧（codec encode）。线性插值升采样 48k、复制双声道
     * （encode 输入 [1,2,N] planar float32），参考上限 20s（250 帧，与内置音色量级一致）。
     */
    private fun encodeRef16k(pcm16k: FloatArray): Array<IntArray>? {
        val enc = codecEncode ?: return null
        val n48 = minOf(pcm16k.size * 3, 48_000 * 20)
        // 线性插值升采样 16k → 48k
        val up = FloatArray(n48)
        for (i in 0 until n48) {
            val x = i / 3f
            val i0 = x.toInt()
            val i1 = if (i0 + 1 < pcm16k.size) i0 + 1 else i0
            up[i] = pcm16k[i0] + (pcm16k[i1] - pcm16k[i0]) * (x - i0)
        }
        // planar 双声道 [1,2,N]：前 N 为左声道、后 N 为右声道
        val planar = FloatArray(n48 * 2)
        System.arraycopy(up, 0, planar, 0, n48)
        System.arraycopy(up, 0, planar, n48, n48)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(planar), longArrayOf(1, 2, n48.toLong())).use { w ->
            OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(n48)), longArrayOf(1)).use { l ->
                enc.run(mapOf("waveform" to w, "input_lengths" to l)).use { out ->
                    val t = out.get("audio_codes").get() as OnnxTensor   // [1, frames, 16]
                    val shape = t.info.shape
                    val frames = shape[1].toInt()
                    if (frames <= 0) return null
                    val fb = t.intBuffer
                    val codes = Array(frames) { IntArray(16) }
                    for (f in 0 until frames) for (c in 0 until 16) codes[f][c] = fb.get(f * 16 + c)
                    Log.i(TAG, "voice ref encoded: $frames frames (${frames / 12.5f}s)")
                    return codes
                }
            }
        }
    }

    private suspend fun runSynthesis(
        mf: JSONObject, text: String, refCodes: Array<IntArray>,
        maxFrames: Int, onFrame: ((Int) -> Unit)?
    ): TtsAudio? {
        val ctx = coroutineContext
        val cfg = mf.getJSONObject("tts_config")
        val nVq = cfg.getInt("n_vq")                              // 16
        val rowW = nVq + 1                                        // 17
        val audioPad = cfg.getInt("audio_pad_token_id")            // 1024
        val audioStart = cfg.getInt("audio_start_token_id")        // 6
        val audioEnd = cfg.getInt("audio_end_token_id")            // 7
        val userSlot = cfg.getInt("audio_user_slot_token_id")      // 8
        val asstSlot = cfg.getInt("audio_assistant_slot_token_id") // 9

        // 1) 文本 token（BPE 编码已与官方样本对拍一致）
        val textIds = tokenizer?.encode(text) ?: return null
        if (textIds.isEmpty()) return null

        // 2) 音色参考帧：说话人录音编码出的码帧（克隆音色）
        if (refCodes.isEmpty()) return null

        // 3) 组装提示行（每行 17 列 = [文本 token, 16 音频码]）
        val pt = mf.getJSONObject("prompt_templates")
        val rows = ArrayList<IntArray>(256)
        fun addTextRows(key: String) {
            val ids = pt.getJSONArray(key)
            for (i in 0 until ids.length()) rows.add(textRow(ids.getInt(i), audioPad, nVq))
        }
        addTextRows("user_prompt_prefix_token_ids")
        rows.add(textRow(audioStart, audioPad, nVq))
        refCodes.forEach { codes -> rows.add(intArrayOf(userSlot, *codes)) }
        rows.add(textRow(audioEnd, audioPad, nVq))
        addTextRows("user_prompt_after_reference_token_ids")
        for (id in textIds) rows.add(textRow(id, audioPad, nVq))
        addTextRows("assistant_prompt_prefix_token_ids")
        rows.add(textRow(audioStart, audioPad, nVq))

        // 4) prefill：整段提示 → 末位 hidden + 全局 KV（Result 保持打开，张量归其所有）
        val seqLen = rows.size
        val flat = IntArray(seqLen * rowW)
        rows.forEachIndexed { r, row -> System.arraycopy(row, 0, flat, r * rowW, rowW) }
        val prefillSess = prefill ?: return null
        var kvResult: OrtSession.Result
        var hidden: FloatArray
        OnnxTensor.createTensor(env, IntBuffer.wrap(flat), longArrayOf(1, seqLen.toLong(), rowW.toLong())).use { ids ->
            OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(seqLen) { 1 }), longArrayOf(1, seqLen.toLong())).use { mask ->
                kvResult = prefillSess.run(mapOf("input_ids" to ids, "attention_mask" to mask))
            }
        }
        try {
            hidden = lastHidden(kvResult.get("global_hidden").get() as OnnxTensor, seqLen)

            var validLen = seqLen
            val frames = ArrayList<IntArray>(maxFrames.coerceAtMost(1024))
            val repSeen = IntArray(nVq * 1024)   // [16,1024] 展平
            val rng = Random(text.hashCode().toLong() * 31 + refCodes.size)
            val frameSess = frameGraph ?: return null
            val decodeSess = decodeStep ?: return null

            while (frames.size < maxFrames) {
                ctx.ensureActive()
                // 5) 帧采样：局部 transformer + 注入随机数（top-k/top-p/重复惩罚在图内）
                var stop = false
                frameSess.run(mapOf(
                    "global_hidden" to floatTensor(hidden, 1L, HIDDEN.toLong()),
                    "repetition_seen_mask" to intTensor(repSeen, 1L, nVq.toLong(), 1024L),
                    "assistant_random_u" to floatTensor(floatArrayOf(rng.nextFloat()), 1L),
                    "audio_random_u" to floatTensor(FloatArray(nVq) { rng.nextFloat() }, 1L, nVq.toLong())
                )).use { out ->
                    val cont = (out.get("should_continue").get() as OnnxTensor).intBuffer.get(0)
                    if (cont == 0) {
                        stop = true   // 自然收尾
                    } else {
                        val fb = (out.get("frame_token_ids").get() as OnnxTensor).intBuffer
                        frames.add(IntArray(nVq) { fb.get(it) })
                    }
                }
                if (stop) break
                val fr = frames.last()
                onFrame?.invoke(frames.size)
                for (c in 0 until nVq) repSeen[c * 1024 + fr[c]] = 1

                // 6) 喂回全局 transformer：行 = [assistant_slot, 16 码]
                val decIn = HashMap<String, OnnxTensor>(N_LAYERS * 2 + 2)
                decIn["input_ids"] = intTensor(intArrayOf(asstSlot, *fr), 1L, 1L, rowW.toLong())
                decIn["past_valid_lengths"] = intTensor(intArrayOf(validLen), 1L)
                for (i in 0 until N_LAYERS) {
                    decIn["past_key_$i"] = kvResult.get("present_key_$i").get() as OnnxTensor
                    decIn["past_value_$i"] = kvResult.get("present_value_$i").get() as OnnxTensor
                }
                val newResult = decodeSess.run(decIn)
                // 输入自建张量关闭；KV 输入归旧 Result 所有，由下面 close 一并释放
                decIn["input_ids"]?.close()
                decIn["past_valid_lengths"]?.close()
                kvResult.close()
                kvResult = newResult
                hidden = FloatArray(HIDDEN).also { arr ->
                    (kvResult.get("global_hidden").get() as OnnxTensor).floatBuffer.get(arr)
                }
                validLen++
            }

            if (frames.isEmpty()) return null
            Log.i(TAG, "synthesized ${frames.size} frames (${"%.1f".format(frames.size / 12.5f)}s) for [$text]")

            // 7) codec 解码：音频码 [1, n_frames, 16] → 48k 立体声
            val codec = codecDecode ?: return null
            val codesFlat = IntArray(frames.size * nVq)
            frames.forEachIndexed { f, row -> System.arraycopy(row, 0, codesFlat, f * nVq, nVq) }
            val pcm: FloatArray
            codec.run(mapOf(
                "audio_codes" to intTensor(codesFlat, 1L, frames.size.toLong(), nVq.toLong()),
                "audio_code_lengths" to intTensor(intArrayOf(frames.size), 1L)
            )).use { out ->
                val t = out.get("audio").get() as OnnxTensor   // [1, channels, samples] planar
                val shape = t.info.shape
                val ch = shape[1].toInt()
                val samples = shape[2].toInt()
                val planar = FloatArray(ch * samples).also { arr -> t.floatBuffer.get(arr) }
                // planar（[L...L][R...R]）→ 交错（L,R,L,R,...）：直接把 planar 当交错写 WAV
                // 会让每个声道只取到隔一个采样，播放出来 2 倍速 + 变调
                pcm = FloatArray(ch * samples)
                for (i in 0 until samples) {
                    for (c in 0 until ch) {
                        pcm[i * ch + c] = planar[c * samples + i]
                    }
                }
            }
            return TtsAudio(pcm)
        } finally {
            runCatching { kvResult.close() }
        }
    }

    /** [文本 token, pad×16] 的 17 列文本行。 */
    private fun textRow(token: Int, audioPad: Int, nVq: Int): IntArray {
        val row = IntArray(nVq + 1)
        row[0] = token
        for (i in 1..nVq) row[i] = audioPad
        return row
    }

    /** prefill 输出 hidden [1, seq, 768] 的末位 [768]。 */
    private fun lastHidden(t: OnnxTensor, seqLen: Int): FloatArray {
        val fb = t.floatBuffer
        val total = fb.remaining()
        val v = total / seqLen
        fb.position((seqLen - 1) * v)
        return FloatArray(v).also { fb.get(it) }
    }

    private fun floatTensor(a: FloatArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(a), shape)

    private fun intTensor(a: IntArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(env, IntBuffer.wrap(a), shape)

    private fun createSession(file: File): OrtSession {
        val so = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(4, 8))
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        return env.createSession(file.absolutePath, so)
    }

    /**
     * assets/<dir> 全量解包到 filesDir/<dir>（按未压缩大小校验缓存），返回目录绝对路径。
     * tts 目录下含子目录（codec/），需递归；单个文件失败只告警不中断（aapt 可能塞进
     * 空目录/无法打开的条目，不应拖垮整个解包）。
     */
    private fun extractAssetDir(dir: String): String {
        val outDir = File(context.filesDir, dir)
        extractRec(dir, outDir)
        return outDir.absolutePath
    }

    private fun extractRec(assetDir: String, outDir: File) {
        val names = context.assets.list(assetDir) ?: emptyArray()
        if (!outDir.exists()) outDir.mkdirs()
        val nameSet = names.toSet()
        outDir.listFiles()?.forEach { f ->
            if (f.isFile && f.name !in nameSet) runCatching { f.delete() }
        }
        for (name in names) {
            val cached = File(outDir, name)
            val sub = context.assets.list("$assetDir/$name")
            if (!sub.isNullOrEmpty()) {          // 子目录 → 递归
                extractRec("$assetDir/$name", cached)
                continue
            }
            val assetSize = try {
                context.assets.openFd("$assetDir/$name").use { it.length }
            } catch (e: Exception) { -1L }       // 压缩资产 openFd 抛异常 → 跳过大小校验
            if (cached.exists() && cached.length() > 0L &&
                (assetSize < 0L || cached.length() == assetSize)) continue
            Log.i(TAG, "extracting $assetDir/$name")
            runCatching {
                context.assets.open("$assetDir/$name").use { input ->
                    FileOutputStream(cached).use { output -> input.copyTo(output) }
                }
            }.onFailure { Log.w(TAG, "extract $assetDir/$name failed", it) }
        }
    }

    companion object {
        private const val TAG = "TtsEngine"
        private const val N_LAYERS = 12
        private const val HIDDEN = 768
        /** 音频码本数（tts_config.n_vq），内置音色码帧每行长度。 */
        private const val N_VQ = 16
        /** 默认生成帧上限：12.5 帧/秒 → 375 帧 ≈ 30s 语音。 */
        private const val DEFAULT_MAX_FRAMES = 375

        @Volatile private var instance: TtsEngine? = null
        fun get(context: Context): TtsEngine =
            instance ?: synchronized(this) {
                instance ?: TtsEngine(context.applicationContext).also { instance = it }
            }

        /** 把 48k 立体声交错 PCM 写成 WAV（16bit）。 */
        fun writeWavStereo(path: String, pcmInterleaved: FloatArray, sampleRate: Int = 48_000) {
            val channels = 2
            val dataLen = pcmInterleaved.size * 2
            File(path).parentFile?.mkdirs()
            FileOutputStream(File(path)).use { fos ->
                val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
                header.put("RIFF".toByteArray())
                header.putInt(36 + dataLen)
                header.put("WAVE".toByteArray())
                header.put("fmt ".toByteArray())
                header.putInt(16)
                header.putShort(1)
                header.putShort(channels.toShort())
                header.putInt(sampleRate)
                header.putInt(sampleRate * channels * 2)
                header.putShort((channels * 2).toShort())
                header.putShort(16)
                header.put("data".toByteArray())
                header.putInt(dataLen)
                fos.write(header.array())
                val buf = ByteBuffer.allocate(dataLen).order(ByteOrder.LITTLE_ENDIAN)
                for (s in pcmInterleaved) {
                    buf.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
                }
                fos.write(buf.array())
            }
        }
    }
}

/**
 * SentencePiece BPE 分词器（直接解析 tokenizer.model 的 protobuf wire 格式，无第三方依赖）。
 *
 * 该模型为 BPE（piece 的 score 字段是整数合并优先级，非 unigram 对数概率）——已用
 * browser_poc_manifest.json 中 zh/en 两组官方 token ids 对拍验证：NFKC 归一化 + 空白折叠 +
 * ▁ 转义 + dummy prefix + 单字符起步按 rank 贪心合并，两组样本编码完全一致。
 */
private class SpBpeTokenizer(modelBytes: ByteArray) {
    private val vocab = HashMap<String, Int>(16384 * 2)
    private val rank = HashMap<String, Int>(16384 * 2)
    private val unkId = 0

    init {
        // Model{1: repeated Piece}，Piece{1: string piece, 2: float score, 3: enum type}
        val b = modelBytes
        var i = 0
        while (i < b.size) {
            val (tag, ni) = readVarint(b, i)
            i = ni
            val fnum = tag ushr 3
            val wt = tag and 7
            if (fnum == 1 && wt == 2) {          // 一个 Piece（嵌套消息）
                val (len, li) = readVarint(b, i)
                i = li
                val end = i + len
                var j = i
                var piece: String? = null
                var score = 0f
                while (j < end) {
                    val (t, nj) = readVarint(b, j)
                    j = nj
                    val fn = t ushr 3
                    val w = t and 7
                    when {
                        fn == 1 && w == 2 -> {   // piece（UTF-8 字符串）
                            val (l, lj) = readVarint(b, j)
                            j = lj
                            piece = String(b, j, l, Charsets.UTF_8)
                            j += l
                        }
                        fn == 2 && w == 5 -> {   // score（32 位小端 float）
                            score = floatFromBits(b, j)
                            j += 4
                        }
                        else -> j = skipField(b, j, w)
                    }
                }
                i = end
                if (piece != null) {
                    // pieces 在文件中按 id 顺序出现，顺序编号即 token id（已对拍验证）
                    vocab[piece] = vocab.size
                    rank[piece] = -score.toInt()   // rank 越小越优先合并
                }
            } else {
                i = skipField(b, i, wt)
            }
        }
        Log.i("SpBpeTokenizer", "loaded ${vocab.size} pieces")
    }

    fun encode(text: String): IntArray {
        val norm = normalize(text)
        if (norm.isEmpty()) return IntArray(0)
        val t = "▁" + norm.replace(' ', '▁')
        val syms = ArrayList<String>(t.length)
        for (c in t) syms.add(c.toString())
        // BPE 合并：找 rank 最小的可合并相邻对，合并后重扫
        while (syms.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            var best = ""
            for (k in 0 until syms.size - 1) {
                val cand = syms[k] + syms[k + 1]
                val r = rank[cand] ?: continue
                if (r < bestRank) { bestRank = r; bestIdx = k; best = cand }
            }
            if (bestIdx < 0) break
            syms[bestIdx] = best
            syms.removeAt(bestIdx + 1)
        }
        val ids = IntArray(syms.size)
        for (k in syms.indices) ids[k] = vocab[syms[k]] ?: unkId
        return ids
    }

    /** NFKC + 控制符移除 + 空白折叠（nmt_nfkc 的近似实现，对拍通过）。 */
    private fun normalize(text: String): String {
        val nfkc = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFKC)
        val sb = StringBuilder(nfkc.length)
        var lastSpace = true
        for (c in nfkc) {
            val type = Character.getType(c)
            val isControl = type == Character.CONTROL.toInt() || type == Character.FORMAT.toInt() ||
                type == Character.PRIVATE_USE.toInt() || type == Character.SURROGATE.toInt() ||
                type == Character.UNASSIGNED.toInt()
            if (isControl) continue
            if (c.isWhitespace()) {
                if (!lastSpace) { sb.append(' '); lastSpace = true }
            } else {
                sb.append(c); lastSpace = false
            }
        }
        return sb.toString().trim()
    }

    private fun readVarint(b: ByteArray, i: Int): Pair<Int, Int> {
        var r = 0; var s = 0; var idx = i
        while (true) {
            val x = b[idx].toInt() and 0xFF
            idx++
            r = r or ((x and 0x7F) shl s)
            if (x and 0x80 == 0) return r to idx
            s += 7
        }
    }

    private fun skipField(b: ByteArray, i: Int, wt: Int): Int = when (wt) {
        0 -> readVarint(b, i).second
        1 -> i + 8
        2 -> { val (l, li) = readVarint(b, i); li + l }
        5 -> i + 4
        else -> throw IllegalArgumentException("wire type $wt")
    }

    private fun floatFromBits(b: ByteArray, i: Int): Float =
        java.lang.Float.intBitsToFloat(
            (b[i].toInt() and 0xFF) or ((b[i + 1].toInt() and 0xFF) shl 8) or
                ((b[i + 2].toInt() and 0xFF) shl 16) or ((b[i + 3].toInt() and 0xFF) shl 24)
        )
}
