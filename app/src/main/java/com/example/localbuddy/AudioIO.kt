package com.example.localbuddy

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.projection.MediaProjection
import android.net.Uri
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * VAD 最短有效段（ms）：两个采集器（麦克风 / 回放内录）共用的下限。
 * 短于此的段基本是噪声爆音、键盘声、咳嗽、音乐鼓点触发的碎片——送进 ASR 只会产出乱码
 * （噪声下模型更常"吐错字"而不是"吐空"），丢弃既省一次推理又直接减少错字。
 */
private const val VAD_MIN_SEG_MS = 250

/**
 * 16k 单声道 Float32 PCM 采集器。
 *
 * [stream] 采用 **能量 VAD 句尾分段**：持续读取音频并以 30ms 为一帧计算 RMS 能量，
 * 用状态机按"一句话"切段推送：
 *  - 静音期不推送（环境噪声/无人说话 → 不产生任何段，下游不转写不翻译）；
 *  - 检测到说话声开始累积；
 *  - 说话结束后出现 ≥ [pauseMs] 的静音停顿 → 判定句尾，整段推送一次；
 *  - 持续说话不停顿达 [maxSegMs] → 强制切出一段（避免单段过长拖慢推理）。
 *
 * 这样下游每次收到的都是「一段完整人声」，而非固定时长的混合片段，可显著减少
 * 静音段被误转写/误翻译的问题。
 *
 * ⚠️ 语音起始带 [LEAD_IN_FRAMES] 帧前导：进入说话态时把前导缓冲（最近若干帧，含字头前少量静音）
 * 回补进段，避免能量门限滞后检出把字头切掉（其他分段器 [TimedVadSegmenter]/[StreamingVadSegmenter]
 * 均有此处理，本 recorder 早期版本缺它，每段都丢开头 30~150ms）。
 */
class MicRecorder(
    private val sampleRate: Int = 16_000,
    /** 句尾停顿时长（ms）：说完后连续静音到该时长即判句尾。 */
    private val pauseMs: Int = 600,
    /** 单段最长时长（ms）：持续说话不停顿时，到该时长强制切一段。 */
    private val maxSegMs: Int = 8000,
    /** 帧长（ms）：VAD 能量计算的粒度。 */
    private val frameMs: Int = 30
) {
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    @Volatile private var recording: Boolean = false
    private var audioRecord: AudioRecord? = null

    fun hasPermission(context: Context): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    fun stop() {
        recording = false
    }

    /**
     * 启动录音，按 VAD 句尾分段持续推送 Float32 PCM（[-1,1]）。
     * 静音段不推送。通过 [stop] 退出；退出时若缓冲中有未发出的语音会补发一次。
     */
    fun stream(): Flow<FloatArray> = flow {
        val minBuffer = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        if (minBuffer <= 0) error("AudioRecord.getMinBufferSize returned $minBuffer")
        val frameSamples = sampleRate * frameMs / 1000
        // 读取缓冲稍大于一帧，保证实时性
        val readBuf = ShortArray(frameSamples * 2)
        val bufferSizeBytes = max(minBuffer, readBuf.size * 2 * 4)

        @Suppress("MissingPermission")
        val record = AudioRecord(
            android.media.MediaRecorder.AudioSource.MIC,
            sampleRate, channelConfig, audioFormat, bufferSizeBytes
        )
        audioRecord = record
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            error("AudioRecord init failed")
        }

        // 系统级音频效果（厂商 DSP）：NoiseSuppressor 压稳态背景噪声，AcousticEchoCanceler
        // 消除"手机自己外放的声音被麦克风采回"（看视频/放音乐时录音翻译的典型干扰）。
        // 仅在设备声明支持时启用，且不改采集源——避免 VOICE_COMMUNICATION 的窄带滤波与强 AGC
        // 把语音频谱改坏（ASR 对这类失真很敏感，历史上换源后识别率反而下降）。
        val ns = if (NoiseSuppressor.isAvailable())
            runCatching { NoiseSuppressor.create(record.audioSessionId) }
                .onFailure { Log.w(TAG, "NoiseSuppressor create failed", it) }.getOrNull()
        else null
        val aec = if (AcousticEchoCanceler.isAvailable())
            runCatching { AcousticEchoCanceler.create(record.audioSessionId) }
                .onFailure { Log.w(TAG, "AEC create failed", it) }.getOrNull()
        else null
        if (ns != null || aec != null) Log.i(TAG, "audio fx: ns=${ns != null} aec=${aec != null}")

        val pauseFrames = pauseMs / frameMs
        val maxSegSamples = sampleRate * maxSegMs / 1000
        val minSegSamples = sampleRate * VAD_MIN_SEG_MS / 1000
        // VAD：能量门限（自适应底噪）+ 谱门（拒宽带噪声/音乐伴奏主导帧），见 EnergyVadDetector
        val vad = EnergyVadDetector(frameSamples, ABSOLUTE_VOICE_RMS)

        // 自适应噪声门限：用开头的静音估计环境底噪 RMS，门限 = max(底噪×3, 固定下限)
        val noiseFloor = FloatArray(ESTIMATE_FRAMES)
        var noiseIdx = 0
        var noiseReady = false
        var noiseSum = 0.0
        var threshold = ABSOLUTE_VOICE_RMS

        // 当前正在累积的语音段
        val seg = ArrayList<Float>(maxSegSamples / 2)
        var speaking = false          // 是否处于说话态（已进入语音段）
        var silenceRun = 0           // 说话态下连续静音帧数（用于判句尾）
        var trailingSilence = 0       // 静音态下连续静音帧数（用于估底噪/抑制抖动）

        // 前导缓冲：滚动保留最近 LEAD_IN_FRAMES 帧（无论语音/静音）。语音起始时回补进段，
        // 避免字头被能量门限滞后检出而切掉；句尾停顿切段保留（末几帧是静音），maxSeg 强切则清空避免重复计入。
        val leadIn = ArrayList<Float>()
        val leadCap = frameSamples * LEAD_IN_FRAMES

        recording = true
        record.startRecording()
        try {
            while (recording) {
                var read = 0
                while (read < readBuf.size && recording) {
                    val n = record.read(readBuf, read, readBuf.size - read)
                    if (n <= 0) break
                    read += n
                }
                if (read <= 0) continue

                // 逐帧（frameSamples）计算 RMS
                var off = 0
                while (off + frameSamples <= read && recording) {
                    var sumSq = 0.0
                    for (i in 0 until frameSamples) {
                        val s = readBuf[off + i] / 32768f
                        sumSq += (s * s).toDouble()
                    }
                    val rms = sqrt(sumSq / frameSamples).toFloat()

                    // 静音期更新噪声估计
                    if (!speaking) {
                        if (!noiseReady) {
                            noiseFloor[noiseIdx] = rms
                            noiseSum += rms
                            noiseIdx++
                            if (noiseIdx >= ESTIMATE_FRAMES) {
                                noiseReady = true
                                val mean = (noiseSum / ESTIMATE_FRAMES).toFloat()
                                threshold = max(mean * NOISE_MULTIPLIER, ABSOLUTE_VOICE_RMS)
                                Log.d(TAG, "VAD noise est mean=%.4f thr=%.4f".format(mean, threshold))
                            }
                        } else {
                            // 持续用更低 RMS 的帧更新底噪，适应环境变安静
                            if (rms < noiseFloor.minOrNull() ?: Float.MAX_VALUE) {
                                noiseSum += rms - (noiseFloor[noiseIdx % ESTIMATE_FRAMES])
                                noiseFloor[noiseIdx % ESTIMATE_FRAMES] = rms
                                noiseIdx++
                                val mean = (noiseSum / ESTIMATE_FRAMES).toFloat()
                                threshold = max(mean * NOISE_MULTIPLIER, ABSOLUTE_VOICE_RMS)
                            }
                        }
                    }

                    // 门限随底噪估计变化 → 每帧同步给 VAD；判定含谱门（噪声/伴奏主导帧不算语音）
                    vad.setThreshold(threshold)
                    val isVoice = vad.isVoice(readBuf, off, frameSamples)

                    if (!speaking) {
                        if (isVoice) {
                            // 进入说话态：先回补前导缓冲（含字头的若干前置帧，避免字头因门限滞后被切），
                            // 再开启累积当前帧
                            if (leadIn.isNotEmpty()) {
                                seg.addAll(leadIn)
                                leadIn.clear()
                            }
                            speaking = true
                            silenceRun = 0
                            trailingSilence = 0
                            // 加入当前帧（含字头）
                            appendFrame(seg, readBuf, off, frameSamples)
                        } else {
                            trailingSilence++
                        }
                    } else {
                        // 说话态：累积本帧
                        appendFrame(seg, readBuf, off, frameSamples)
                        if (isVoice) {
                            silenceRun = 0
                            // 单段上限在语音帧分支也要检查（同 StreamingVadSegmenter）：持续说话无
                            // 停顿时若只在静音帧判断，段会一直累积到被 ASR 12s 截断、后半句丢失，
                            // 实时场景下还表现为"说了很久都不出字"。
                            if (seg.size >= maxSegSamples) {
                                emit(seg.toFloatArray())
                                seg.clear()
                                speaking = false
                                silenceRun = 0
                                leadIn.clear()
                            }
                        } else {
                            silenceRun++
                            val maxSegCut = seg.size >= maxSegSamples
                            if (silenceRun >= pauseFrames || maxSegCut) {
                                // 句尾停顿 或 单段过长 → 推送整段
                                emit(seg.toFloatArray())
                                seg.clear()
                                speaking = false
                                silenceRun = 0
                                // maxSeg 强切（非句尾停顿）：本帧是句中语音，leadIn 末几帧是上一段语音
                                // 尾巴，清空避免下一段把它当字头前导重复计入（句尾停顿切则保留，leadIn 末是静音）
                                if (maxSegCut) leadIn.clear()
                            }
                        }
                    }
                    // 滚动维护前导缓冲（无论语音/静音），始终保留最近 LEAD_IN_FRAMES 帧
                    appendFrame(leadIn, readBuf, off, frameSamples)
                    if (leadIn.size > leadCap) leadIn.subList(0, leadIn.size - leadCap).clear()
                    off += frameSamples
                }
            }
            // 停止时若有未发出的语音尾巴，补发一次（不丢已说的话；过短尾巴按噪声丢弃）
            if (speaking && seg.size >= minSegSamples) {
                emit(seg.toFloatArray())
                seg.clear()
            }
        } finally {
            runCatching { record.stop() }
            runCatching { record.release() }
            runCatching { ns?.release() }
            runCatching { aec?.release() }
            audioRecord = null
        }
    }

    private fun appendFrame(seg: ArrayList<Float>, buf: ShortArray, off: Int, n: Int) {
        for (i in 0 until n) seg.add(buf[off + i] / 32768f)
    }

    companion object {
        private const val TAG = "MicRecorder"
        /** 语音起始前导帧数：进入说话态时回补最近若干帧（30ms×5≈150ms），含住字头避免被切。对齐 [TimedVadSegmenter]/[StreamingVadSegmenter]。 */
        private const val LEAD_IN_FRAMES = 5
        /** 底噪估计帧数（约 1s：30ms×33）。 */
        private const val ESTIMATE_FRAMES = 33
        /** 噪声倍数：门限 = 底噪均值 × 此值。 */
        private const val NOISE_MULTIPLIER = 3.0f
        /** 绝对语音 RMS 下限，防止极安静环境下门限过低、把量化噪声当人声。 */
        private const val ABSOLUTE_VOICE_RMS = 0.012f
    }
}

/**
 * 从用户上传的音频 Uri 解码出 16k mono Float32 PCM。
 * 任意 MediaCodec 支持的容器（mp3 / m4a / wav / flac / ogg…）都能处理。
 */
object AudioDecoder {

    private const val TAG = "AudioDecoder"
    private const val TARGET_SR = 16_000

    /** 峰值归一化到 [-0.9, 0.9]：视频音轨经编码整体电平常偏低，低电平段送 ASR 易被判空；
     *  归一化后信噪比不变但绝对电平提升，ASR 更稳。近静音段（峰值过低）原样返回，避免把
     *  量化噪声放大成满幅。在 VAD 切段后、送 ASR 前对每段调用。 */
    fun normalizePeak(pcm: FloatArray): FloatArray {
        var peak = 0f
        for (s in pcm) {
            val a = if (s < 0f) -s else s
            if (a > peak) peak = a
        }
        if (peak < 1e-4f) return pcm
        val gain = 0.9f / peak
        return FloatArray(pcm.size) { pcm[it] * gain }
    }

    /**
     * ⚠️ 已废弃、勿用：整条音频一次性解码持有全部 PCM（30 分钟 ≈ 115MB）会 OOM，
     * 且这里仍用**容器声明**的采样率/声道解释输出（见 [decodeStreamPcm16kMono] 的坑）。
     * 统一用 [decodeStreamPcm16kMono] 流式解码；保留仅为兼容，待无引用后删除。
     */
    @Deprecated("使用 decodeStreamPcm16kMono（流式，且按解码器实际输出格式解释 PCM）")
    fun decodeToPcm16kMono(context: Context, uri: Uri): FloatArray {
        val extractor = MediaExtractor()
        context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { afd ->
            extractor.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
        } ?: extractor.setDataSource(context, uri, null)

        val audioTrack = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("No audio track in $uri")

        extractor.selectTrack(audioTrack)
        val format = extractor.getTrackFormat(audioTrack)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: error("Missing mime")
        val srcSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmShorts = ArrayList<Short>(srcSr * 8) // 经验初始容量
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inIdx, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx >= 0) {
                    if (info.size > 0) {
                        val buf = codec.getOutputBuffer(outIdx)!!
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        val shortBuf = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                        while (shortBuf.hasRemaining()) {
                            pcmShorts.add(shortBuf.get())
                        }
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                }
            }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { extractor.release() }
        }

        // 下混到单声道
        val mono = if (channels <= 1) {
            ShortArray(pcmShorts.size) { pcmShorts[it] }
        } else {
            val n = pcmShorts.size / channels
            ShortArray(n) { i ->
                var sum = 0
                for (c in 0 until channels) sum += pcmShorts[i * channels + c]
                (sum / channels).toShort()
            }
        }

        // 重采样到 16k（线性插值，CPU 友好，足够 ASR 用）
        val resampled = resampleLinear(mono, srcSr, TARGET_SR)
        Log.d(TAG, "decoded ${resampled.size} samples (${resampled.size / TARGET_SR.toFloat()}s)")
        // 转 Float32 [-1,1]
        return FloatArray(resampled.size) { resampled[it] / 32768f }
    }

    private fun resampleLinear(src: ShortArray, srcSr: Int, dstSr: Int): ShortArray {
        if (srcSr == dstSr || src.isEmpty()) return src
        val ratio = srcSr.toDouble() / dstSr
        val outLen = (src.size / ratio).toInt()
        val out = ShortArray(outLen)
        for (i in 0 until outLen) {
            val srcPos = i * ratio
            val i0 = srcPos.toInt()
            val i1 = min(i0 + 1, src.size - 1)
            val frac = srcPos - i0
            val v = src[i0] * (1 - frac) + src[i1] * frac
            out[i] = v.toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return out
    }

    /**
     * 流式解码：把音频 Uri 解码为 16k mono Float32 PCM，分块回调 [onPcm]，**不持有完整 PCM**。
     * 供长视频用——一次性解码整条音频会 OOM（30 分钟 PCM ~115MB）。每块解码后立即下混 + 重采样
     * + 转 float 回调出去，调用方消费完即可释放。重采样跨块保持插值状态（[MonoResampler]）。
     * 回调为 suspend：调用方可在其中做 ASR（慢），解码循环会等待回调返回再继续，天然限流。
     */
    suspend fun decodeStreamPcm16kMono(context: Context, uri: Uri, onPcm: suspend (FloatArray) -> Unit) {
        // ① WAV 先走自解析：MediaExtractor 对 audio/x-wav 的支持随设备/系统版本而异，
        //    不少机器直接报"No audio track"（表现为"上传 wav 没反应/识别不了"）。
        //    RIFF 自解析无兼容问题、也不需要解码器；不是 WAV 或压缩 WAV 时回退 ②。
        if (decodeWavStream(context, uri, onPcm)) return

        // ② 其余容器（mp3 / m4a-aac / flac / ogg-opus / 3gp …）：MediaExtractor + MediaCodec
        val extractor = MediaExtractor()
        context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { afd ->
            extractor.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
        } ?: extractor.setDataSource(context, uri, null)

        val audioTrack = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: error("No audio track in $uri")

        extractor.selectTrack(audioTrack)
        val format = extractor.getTrackFormat(audioTrack)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: error("Missing mime")
        val srcSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        // ⚠️ 用**解码器实际输出**的参数解释 PCM，而不是容器声明值：HE-AAC(SBR/PS)、部分
        //    m4a/ogg 的容器声明采样率/声道数/位深与解码输出不一致，用声明值解释会把数据
        //    整体错位（float 当 16bit 读几乎全是 0 → VAD 判静音 → "m4a 识别不了"）。
        //    实际值由首个输出帧前的 INFO_OUTPUT_FORMAT_CHANGED 给出。
        var outSr = srcSr
        var outCh = channels
        var pcmEnc = AudioFormat.ENCODING_PCM_16BIT
        var resampler = MonoResampler(outSr, TARGET_SR)
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var totalSamples = 0

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val of = codec.outputFormat
                    val nSr = of.getInteger(MediaFormat.KEY_SAMPLE_RATE, outSr)
                    val nCh = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT, outCh)
                    pcmEnc = if (Build.VERSION.SDK_INT >= 24) {
                        of.getInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
                    } else AudioFormat.ENCODING_PCM_16BIT
                    if (nSr != outSr) {
                        // 实际输出采样率与容器声明不同（HE-AAC 常见）→ 冲掉残余后按实际值重建
                        pushTail(resampler.flush(), onPcm) { totalSamples += it }
                        outSr = nSr
                        resampler = MonoResampler(outSr, TARGET_SR)
                    }
                    outCh = nCh
                    Log.i(TAG, "decoder out: mime=$mime sr=$outSr ch=$outCh pcmEnc=$pcmEnc" +
                            " (container sr=$srcSr ch=$channels)")
                    continue
                }
                if (outIdx >= 0) {
                    if (info.size > 0) {
                        val buf = codec.getOutputBuffer(outIdx)!!
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        val interleaved = pcmToShorts(buf, pcmEnc)
                        // 下混到单声道（按实际声道数）
                        val mono = downmix(interleaved, outCh)
                        // 重采样到 16k（跨块保持状态）
                        val resampled = resampler.process(mono)
                        if (resampled.isNotEmpty()) {
                            val f = FloatArray(resampled.size) { resampled[it] / 32768f }
                            onPcm(f)
                            totalSamples += f.size
                        }
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                }
            }
            // flush 重采样残余
            pushTail(resampler.flush(), onPcm) { totalSamples += it }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { extractor.release() }
        }
        if (totalSamples == 0) {
            // 有轨道但一个样本都没解出：通常是容器/编码不被支持或文件损坏。
            // 明确抛错，避免上层静默降级成"未检测到语音"让人以为是音量问题。
            error("音频解码无输出：格式不受支持或文件已损坏 ($uri)")
        }
        Log.i(TAG, "decodeStream: ${totalSamples / TARGET_SR}s from sr=$outSr ch=$outCh")
    }

    /** 冲出重采样残余并回调（复用计数逻辑）。 */
    private suspend fun pushTail(
        tail: ShortArray,
        onPcm: suspend (FloatArray) -> Unit,
        onCount: (Int) -> Unit
    ) {
        if (tail.isEmpty()) return
        val f = FloatArray(tail.size) { tail[it] / 32768f }
        onPcm(f)
        onCount(f.size)
    }

    /** WAV 自解析；不是 WAV / 是不支持的压缩 WAV 时返回 false（调用方回退 MediaCodec）。 */
    private suspend fun decodeWavStream(
        context: Context,
        uri: Uri,
        onPcm: suspend (FloatArray) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        runCatching {
            context.contentResolver.openInputStream(uri)?.use { ins ->
                val seq = WavDecoder.decodeSequence(ins, TARGET_SR) ?: return@runCatching false
                var n = 0
                for (chunk in seq) {
                    onPcm(chunk)
                    n += chunk.size
                }
                Log.i(TAG, "decodeStream(wav): ${n / TARGET_SR}s")
                true
            } ?: false
        }.onFailure { Log.w(TAG, "wav decode failed, fallback to MediaCodec", it) }
            .getOrDefault(false)
    }

    /**
     * 按解码器**实际** PCM 编码把输出缓冲解析为 ShortArray。
     * 多数解码器输出 16bit，但 float（ENCODING_PCM_FLOAT）在 AAC / FLAC / OPUS 上并不少见；
     * 把 float 当 16bit 读会得到几乎全 0 的垃圾数据 → VAD 判静音 → 切 0 段 → 表现为"识别不了"。
     */
    private fun pcmToShorts(buf: ByteBuffer, encoding: Int): ShortArray {
        val le = buf.order(ByteOrder.LITTLE_ENDIAN)
        return when (encoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> {
                val fb = le.asFloatBuffer()
                val out = ShortArray(fb.remaining())
                var i = 0
                while (fb.hasRemaining()) {
                    out[i++] = (fb.get() * 32767f).toInt().coerceIn(-32768, 32767).toShort()
                }
                out
            }
            3 /* ENCODING_PCM_8BIT，无符号，中心 128 */ -> {
                val n = le.remaining()
                val out = ShortArray(n)
                for (i in 0 until n) {
                    out[i] = (((le.get().toInt() and 0xFF) - 128) * 256).toShort()
                }
                out
            }
            6 /* ENCODING_PCM_24BIT packed */ -> {
                val n = le.remaining() / 3
                val out = ShortArray(n)
                for (i in 0 until n) {
                    val b0 = le.get().toInt() and 0xFF
                    val b1 = le.get().toInt() and 0xFF
                    val b2 = le.get().toInt() and 0xFF
                    val v = b0 or (b1 shl 8) or (b2 shl 16)
                    out[i] = ((if (v and 0x800000 != 0) v - 0x1000000 else v) shr 8).toShort()
                }
                out
            }
            7 /* ENCODING_PCM_32BIT */ -> {
                val n = le.remaining() / 4
                val out = ShortArray(n)
                for (i in 0 until n) out[i] = (le.getInt() shr 16).toShort()
                out
            }
            else -> {   // 16bit（默认路径）
                val sb = le.asShortBuffer()
                val out = ShortArray(sb.remaining())
                sb.get(out)
                out
            }
        }
    }

    /** 多声道交错 PCM 下混到单声道。 */
    private fun downmix(interleaved: ShortArray, channels: Int): ShortArray {
        if (channels <= 1) return interleaved
        val n = interleaved.size / channels
        val out = ShortArray(n)
        var idx = 0
        for (i in 0 until n) {
            var sum = 0
            for (c in 0 until channels) sum += interleaved[idx++]
            out[i] = (sum / channels).toShort()
        }
        return out
    }
}

/**
 * 跨块线性重采样器：把源采样率 [srcSr] 的 PCM 重采样到 [dstSr]，分块喂入、分块产出，
 * 保持插值位置状态，避免块边界跳样。供流式解码 / 回放捕获用。
 */
internal class MonoResampler(private val srcSr: Int, private val dstSr: Int = 16_000) {
    private val ratio = srcSr.toDouble() / dstSr
    private var carry = 0.0          // 上一块处理后残余的源采样位置（0 until ratio），跨块延续
    private var lastTail: Short = 0  // 上一块最后一个源样本，供块首插值
    private var hasTail = false

    /** 处理一块源采样，返回重采样后的采样（可能为空）。 */
    fun process(chunk: ShortArray): ShortArray {
        if (ratio == 1.0) return chunk
        if (chunk.isEmpty()) return ShortArray(0)
        val n = chunk.size
        // ⚠️ 用 n / ratio（Double）而不是 n / ratio.toInt()：源采样率低于目标时（8k→16k，
        // 电话录音/窄带 WAV 常见）ratio < 1 → toInt() 为 0 → 整数除零崩溃，整条音频解码失败。
        val out = ArrayList<Short>((n / ratio).toInt() + 2)
        var p = carry  // 当前输出采样在 chunk 中的位置
        while (p < n) {
            val i0 = p.toInt()
            val i1 = (i0 + 1).coerceAtMost(n - 1)
            val frac = p - i0
            val s0 = chunk[i0]
            val s1 = chunk[i1]
            val v = s0 * (1 - frac) + s1 * frac
            out.add(v.toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort())
            p += ratio
        }
        carry = p - n
        lastTail = chunk[n - 1]
        hasTail = true
        return out.toShortArray()
    }

    /** 收尾：用最后样本补一个残余输出（可选，量很小）。 */
    fun flush(): ShortArray {
        if (!hasTail || carry <= 0) return ShortArray(0)
        // 残余不足一个完整输出步，跳过（线性插值无更多数据）
        return ShortArray(0)
    }
}

/**
 * 本机回放音频采集器：通过 MediaProjection + AudioPlaybackCaptureConfiguration 捕获**本机正在播放**
 * 的音频（视频声音/音乐等，匹配 USAGE_MEDIA/USAGE_GAME），48k 单声道捕获后重采样到 16k mono
 * Float32，再用与 [MicRecorder] 相同的能量 VAD 句尾分段，输出一段段完整人声 PCM。供"实时视频听音
 * 智能体"用。
 *
 * 仅 Android 10+（API 29）支持回放捕获；低版本不应构造（由调用方按 [Build.VERSION] 拦截）。
 * 只能捕获允许捕获的应用（USAGE_MEDIA/USAGE_GAME），DRM/受保护/系统内容无法捕获。MediaProjection
 * 需由前台 Service（mediaProjection 类型）持有。
 *
 * 采集与 [MicRecorder] 区别仅在音频源（回放捕获 vs 麦克风）与采样率（48k->重采样 vs 直接 16k），
 * VAD 分段逻辑（自适应噪声底 + 句尾停顿切段）完全一致，复用其经验参数。
 */
@RequiresApi(29)
class PlaybackCaptureRecorder(
    private val mediaProjection: MediaProjection,
    private val pauseMs: Int = 600,
    private val maxSegMs: Int = 8000,
    private val frameMs: Int = 30,
    private val captureSampleRate: Int = 48_000   // 48k 最兼容；再重采样到 16k
) {
    private val targetSr = 16_000

    @Volatile private var recording: Boolean = false

    fun stop() { recording = false }

    /**
     * 启动回放捕获，按 VAD 句尾分段持续推送 Float32 PCM（[-1,1]）。静音段不推送。
     * 通过 [stop] 退出；退出时若缓冲中有未发出的语音会补发一次。与 [MicRecorder.stream] 接口一致。
     */
    fun stream(): Flow<FloatArray> = flow {
        // 公开 API 是 AudioPlaybackCaptureConfiguration.Builder（直接构造器是 @SystemApi 不可用）。
        // 匹配 USAGE_MEDIA/USAGE_GAME：常见视频/游戏回放。
        val playbackConfig = AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .build()
        val captureFormat = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(captureSampleRate)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()
        val minBuf = AudioRecord.getMinBufferSize(
            captureSampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBuf <= 0) error("AudioRecord.getMinBufferSize returned $minBuf")
        val readBuf = ShortArray(max(minBuf, captureSampleRate / 10))   // ~100ms 读取缓冲
        val record = AudioRecord.Builder()
            .setAudioPlaybackCaptureConfig(playbackConfig)
            .setAudioFormat(captureFormat)
            .setBufferSizeInBytes(readBuf.size * 2)
            .build()
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            error("AudioRecord(playback) init failed")
        }

        val resampler = MonoResampler(captureSampleRate, targetSr)
        val frameSamples = targetSr * frameMs / 1000          // 480（16k × 30ms）
        val pauseFrames = pauseMs / frameMs                    // 33（~1s）
        val maxSegSamples = targetSr * maxSegMs / 1000         // 128000

        // 固定低门限判有无声音：回放常有持续背景音，自适应噪声底（前1s平均 / 滚动分位×倍数）
        // 会把对白判静音、全程切不出段。改为只要有声音（rms >= 绝对下限）就累积，定时/句尾切，
        // 段内全保留送 ASR、由 ASR 判空过滤静音/纯背景段（同 TimedVadSegmenter 连续覆盖思路）。
        val threshold = ABSOLUTE_VOICE_RMS
        val minSegSamples = targetSr * VAD_MIN_SEG_MS / 1000
        // VAD：低门限 + 谱门。内录场景背景音乐最常见，谱门能把"纯伴奏/纯噪声"帧判为非语音，
        // 让它们落到句尾停顿里被切成独立段（再由 ASR 判空丢弃），而不是混进对白段里干扰识别。
        val vad = EnergyVadDetector(frameSamples, threshold)
        val frameBuf = FloatArray(frameSamples)
        // 诊断统计：每 ~2s 打一次日志，确认采集有效 + VAD 状态
        var statFrames = 0
        var statMaxRms = 0f
        var statMinRms = 1f
        var segsEmitted = 0

        val seg = ArrayList<Float>(maxSegSamples / 2)
        var speaking = false
        var silenceRun = 0
        var trailingSilence = 0

        val pending = ArrayList<Float>(frameSamples * 4)   // 重采样输出的 16k float，待组 30ms 帧

        recording = true
        record.startRecording()
        try {
            while (recording) {
                val n = record.read(readBuf, 0, readBuf.size)
                if (n <= 0) continue
                // 48k mono int16 -> 16k int16 -> float
                val captured = ShortArray(n) { readBuf[it] }
                val res16k = resampler.process(captured)
                for (s in res16k) pending.add(s / 32768f)
                // 按 30ms 帧处理 VAD（状态机同 MicRecorder）
                var off = 0
                while (off + frameSamples <= pending.size && recording) {
                    var sumSq = 0.0
                    for (i in 0 until frameSamples) {
                        val s = pending[off + i]
                        sumSq += (s * s).toDouble()
                    }
                    val rms = sqrt(sumSq / frameSamples).toFloat()

                    for (i in 0 until frameSamples) frameBuf[i] = pending[off + i]
                    val isVoice = vad.isVoice(frameBuf)
                    // 诊断：每 ~2s 打一次采集 RMS 范围 + VAD 状态，确认采集有效、定位不切段原因
                    statFrames++
                    if (rms > statMaxRms) statMaxRms = rms
                    if (rms < statMinRms) statMinRms = rms
                    if (statFrames % 66 == 0) {
                        Log.i(TAG, "playback rms min=%.4f max=%.4f thr=%.4f speaking=%b segs=%d"
                            .format(statMinRms, statMaxRms, threshold, speaking, segsEmitted))
                        statMaxRms = 0f
                        statMinRms = 1f
                    }
                    if (!speaking) {
                        if (isVoice) {
                            speaking = true
                            seg.clear()
                            silenceRun = 0
                            trailingSilence = 0
                            appendFrame(seg, pending, off, frameSamples)
                        } else {
                            trailingSilence++
                        }
                    } else {
                        appendFrame(seg, pending, off, frameSamples)
                        if (isVoice) {
                            silenceRun = 0
                            // 单段上限在语音帧分支也要检查：回放/对白常长时间无静音帧，只在静音帧
                            // 判断会让段一路累积后被 ASR 12s 截断、后半段对白丢失。
                            if (seg.size >= maxSegSamples) {
                                Log.i(TAG, "seg emit (max) dur=${seg.size * 1000 / targetSr}ms")
                                emit(seg.toFloatArray())
                                segsEmitted++
                                seg.clear()
                                speaking = false
                                silenceRun = 0
                            }
                        } else {
                            silenceRun++
                            if (silenceRun >= pauseFrames || seg.size >= maxSegSamples) {
                                Log.i(TAG, "seg emit dur=${seg.size * 1000 / targetSr}ms")
                                emit(seg.toFloatArray())
                                segsEmitted++
                                seg.clear()
                                speaking = false
                                silenceRun = 0
                            }
                        }
                    }
                    off += frameSamples
                }
                if (off > 0) pending.subList(0, off).clear()   // 移除已消费帧，残余跨 read 续帧
            }
            // 停止时若有未发出的语音尾巴，补发一次（不丢已说的话；过短尾巴按噪声丢弃）
            if (speaking && seg.size >= minSegSamples) {
                emit(seg.toFloatArray())
                seg.clear()
            }
        } finally {
            runCatching { record.stop() }
            runCatching { record.release() }
        }
    }

    private fun appendFrame(seg: ArrayList<Float>, pending: ArrayList<Float>, off: Int, n: Int) {
        for (i in 0 until n) seg.add(pending[off + i])
    }

    companion object {
        private const val TAG = "PlaybackCaptureRecorder"
        private const val ABSOLUTE_VOICE_RMS = 0.012f
    }
}

/**
 * Float32 PCM 播放器（AudioTrack 流式），用于把 TTS 合成的波形直接放出来。
 *
 * 为什么不用 MediaPlayer：TTS 产物是**内存里的 FloatArray**（48k 立体声），没有落盘文件；
 * 走 MediaPlayer 要先写 WAV 再播，白白多一次 I/O 与首字延迟，通话场景里这一段延迟是能被听出来的。
 *
 * 停止语义：AudioTrack.write 会在缓冲满时阻塞，协程取消无法通过 finally 打断它，
 * 所以 [stop] 从外部直接调 `AudioTrack.stop()` —— 被打断的 write 会立即返回（≤0），
 * 循环随即退出，配合 [stopped] 标志避免竞态下继续写。
 */
class PcmPlayer {

    /** AudioTrack 缓冲上限（1MB ≈ 5.4s 的 48k 立体声）：再大申请不到共享内存。 */
    private val maxTrackBytes = 1 shl 20

    @Volatile private var stopped = false
    private var track: AudioTrack? = null

    /** 当前是否正在播放。 */
    val playing: Boolean get() = track != null && !stopped

    /**
     * 播放交错 Float32 PCM（[-1,1]），阻塞到播完或被 [stop]。
     *
     * @param pcm 交错样本（L,R,L,R…）；单声道时 [channels] 传 1
     * @param onProgress 播放进度回调（0..1），可用于驱动 UI 的声波动画
     */
    suspend fun play(
        pcm: FloatArray,
        sampleRate: Int = 48_000,
        channels: Int = 2,
        onProgress: ((Float) -> Unit)? = null
    ) = withContext(Dispatchers.IO) {
        if (pcm.isEmpty()) return@withContext
        stopped = false
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate,
            if (channels == 1) AudioFormat.CHANNEL_OUT_MONO else AudioFormat.CHANNEL_OUT_STEREO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        val t = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(
                        if (channels == 1) AudioFormat.CHANNEL_OUT_MONO
                        else AudioFormat.CHANNEL_OUT_STEREO
                    )
                    .build()
            )
            // 缓冲按整段 PCM 大小申请可避免 write 频繁阻塞，但要设上限：一整句 20s 48k 立体声
            // ≈7.7MB，AudioTrack 申请超大共享内存会初始化失败
            .setBufferSizeInBytes(max(minBuf, min(pcm.size * 4, maxTrackBytes)))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        track = t
        try {
            t.play()
            // 每次只写 ~200ms：既能及时响应 stop()，也不会因写入过小而频繁系统调用
            val chunk = (sampleRate / 5) * channels
            var off = 0
            while (off < pcm.size && !stopped) {
                val n = minOf(chunk, pcm.size - off)
                val w = t.write(pcm, off, n, AudioTrack.WRITE_BLOCKING)
                if (w <= 0) break
                off += w
                onProgress?.invoke(off.toFloat() / pcm.size)
            }
            // 排空：write 返回后缓冲里还有数据，等播放头追上再结束（否则尾音被切）
            val tailMs = ((pcm.size / channels - t.playbackHeadPosition) * 1000L / sampleRate) + 120L
            val deadline = System.currentTimeMillis() + tailMs
            while (!stopped && System.currentTimeMillis() < deadline) {
                delay(60)
            }
        } finally {
            runCatching { t.stop() }
            runCatching { t.release() }
            track = null
        }
    }

    /** 中断播放（线程安全，可从任意协程调用）。 */
    fun stop() {
        stopped = true
        runCatching { track?.stop() }
    }

    fun release() {
        stop()
        runCatching { track?.release() }
        track = null
    }
}
