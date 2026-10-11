package com.example.localbuddy

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 语音活动检测（VAD）抽象：逐帧判定 PCM 是否含人声。
 * - [SileroVadDetector]：Silero VAD 神经网络（ONNX），对 BGM/噪声/电平波动鲁棒，视频字幕主用。
 * - [EnergyVadDetector]：能量 RMS 门限回退，Silero 模型未加载时兜底。
 *
 * 替代 [StreamingVadSegmenter] 原有的纯能量门限——能量门限对视频音轨（背景音/低电平）易误判，
 * 神经网络 VAD 显著提升人声检出率。
 */
interface VadDetector {
    /** 每帧采样数。本仓库的 silero_vad.onnx 实测需 256 样本/帧（16k = 16ms）：
     *  喂 512 样本时模型对任何音频（含清晰人声）输出概率都 < 0.15 → 全判静音 → 切出 0 段。
     *  改 256 后人声概率可达 ~0.96、静音 ~0.02，门限 0.5 清晰可分。 */
    val frameSamples: Int
    /** 门限值（诊断用；Silero=语音概率阈值 0.5，Energy=RMS 阈值）。 */
    val threshold: Float

    /** 判定一帧是否语音。有状态（Silero 跨帧维护 LSTM state）。 */
    fun isVoice(frame: FloatArray): Boolean
}

/**
 * Silero VAD（ONNX）推理。每帧 256 样本（16k mono Float32 = 16ms）输出语音概率，跨帧维护
 * state[2,1,128]。对背景音/低电平人声鲁棒。
 *
 * OrtSession 非线程安全，多 detector 共享 session 时用 [lock] 串行化。每个 detector 实例
 * 持有独立 state，对应一条音频流；不复用跨视频。
 *
 * ⚠️ 帧长必须 256：本仓库的 silero_vad.onnx 实测喂 512 样本时输出概率恒 < 0.15（即便对
 *    清晰人声），导致全帧判静音、切出 0 段。256 样本下人声 ~0.96 / 静音 ~0.02，正常。
 *
 * 输入/输出签名（assets/vad/silero_vad.onnx）：
 *   input  float32[1,256]   state float32[2,1,128]   sr int64 标量(16000)
 *   output float32[1,1]（语音概率）  stateN float32[2,1,128]（新 state，回填）
 */
class SileroVadDetector(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val lock: Any,
    // 0.35：低于 Silero 标准默认 0.5。本仓库的 silero_vad.onnx 对清晰人声输出概率本就偏低
    // 且不稳定（同等人声 max 可能 0.6 也可能 0.98），0.5 会漏检大量语音（实测召回仅 ~35%）。
    // 0.35 实测召回升至 ~72%，而纯静音概率 max ~0.02，远低于门限，无误触发；且视频字幕管线
    // 对空段会丢弃（ASR 空白即移除），故宁可偏低多收、由 ASR 兜底，也不漏人声。
    private val probThreshold: Float = 0.35f
) : VadDetector {
    override val frameSamples = 256
    override val threshold: Float get() = probThreshold

    private val state = FloatArray(2 * 1 * 128)   // 跨帧 LSTM state，初始 0

    override fun isVoice(frame: FloatArray): Boolean {
        if (frame.size != frameSamples) return false
        return runCatching {
            synchronized(lock) {
                val inputT = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(frame), longArrayOf(1, frameSamples.toLong())
                )
                val stateT = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128)
                )
                val srT = OnnxTensor.createTensor(
                    env, LongBuffer.wrap(longArrayOf(16000L)), longArrayOf()   // 标量
                )
                val result = session.run(mapOf("input" to inputT, "state" to stateT, "sr" to srT))
                val prob = FloatArray(1).also {
                    (result.get("output").get() as OnnxTensor).getFloatBuffer().get(it)
                }
                (result.get("stateN").get() as OnnxTensor).getFloatBuffer().get(state)   // 回填 state
                result.close()
                inputT.close(); stateT.close(); srT.close()
                prob[0] >= probThreshold
            }
        }.getOrElse { false }
    }
}

/**
 * 能量 RMS 门限 VAD（固定门限）。视频字幕管线**主用**：调用方先两遍解码扫整条帧 RMS 的 10 分位
 * 作噪声底，门限 = max(噪声底×3, 0.012) 传入（见 [com.example.localbuddy.VideoSubtitleViewModel.generate]）。
 * 不依赖 Silero 神经网络——本仓库的 silero_vad.onnx 对真实视频人声概率偏低且不稳定，会把明显人声
 * 误判静音；能量门限对明显说话声召回 ~100%。音乐/噪声误触发由下游 ASR 空段丢弃兜底。
 */
class EnergyVadDetector(
    override val frameSamples: Int = 512,
    rmsThreshold: Float = 0.01f,
    /** 能量门限之上是否再过一道**谱门**（拒宽带噪声/音乐伴奏主导帧）。默认开。 */
    private val spectralGate: Boolean = true
) : VadDetector {
    override val threshold: Float get() = rmsThreshold

    /** 运行时更新能量门限（麦克风自适应底噪场景：底噪估计会随环境变化而变）。 */
    @Volatile private var rmsThreshold: Float = rmsThreshold
    fun setThreshold(v: Float) { rmsThreshold = v }

    // 谱门自适应兜底计数：能量命中数 / 谱门通过数
    private var energyHits = 0
    private var spectralPass = 0
    @Volatile private var gateDisabled = false

    // 帧谱分析（复用缓冲的 FFT，见 AudioEnhancer.FrameSpectrum）
    private val spectrum = AudioEnhancer.FrameSpectrum()
    /** Short PCM 判定用的复用缓冲（避免每帧分配 FloatArray）。 */
    private val scratch = FloatArray(frameSamples)

    override fun isVoice(frame: FloatArray): Boolean {
        var ss = 0.0
        for (s in frame) ss += (s * s).toDouble()
        return decideFrom(frame, frame.size, sqrt(ss / frame.size).toFloat())
    }

    /** 直接判 Short PCM（麦克风/回放采集缓冲，省掉每帧转 Float 数组的分配）。 */
    fun isVoice(buf: ShortArray, off: Int, n: Int): Boolean {
        val s = scratch
        val m = min(n, s.size)
        var ss = 0.0
        for (i in 0 until m) {
            val v = buf[off + i] / 32768f
            s[i] = v
            ss += (v * v).toDouble()
        }
        return decideFrom(s, m, sqrt(ss / m).toFloat())
    }

    private fun decideFrom(frame: FloatArray, n: Int, rms: Float): Boolean {
        if (rms < rmsThreshold) return false
        if (!spectralGate || gateDisabled) return true

        val feats = spectrum.analyze(frame, n) ?: return true
        val (sfm, voiceRatio) = feats
        energyHits++
        // 谱门判据（参数刻意保守）：
        //  - SFM ≤ 0.75：只拒"几乎无谐波结构"的宽带噪声（白噪声/嘶声接近 1.0），
        //    低 SNR 人声 SFM 也会升高，过严会把人声全拒掉；
        //  - 人声带占比 ≥ 0.35：拒低频隆隆（空调/风噪）与"鼓点+镲片"型音乐伴奏，
        //    这类帧能量很大却是背景声，正是纯能量 VAD 最容易被骗的地方。
        val ok = sfm <= SFM_MAX && voiceRatio >= VOICE_BAND_MIN
        if (ok) {
            spectralPass++
            return true
        }
        // 兜底：若能量过门限的帧被谱门拒掉 ≥95%，说明本条素材不适用（如极低 SNR 人声、
        // 电平异常的音乐），直接关掉谱门退回纯能量——宁可多收噪声段由 ASR 判空丢弃，
        // 也不能整条切出 0 段（历史上"切 0 段"就是这个模块最严重的故障模式）。
        if (energyHits >= 40 && spectralPass * 20 < energyHits) {
            gateDisabled = true
            return true
        }
        return false
    }

    companion object {
        /** 谱平坦度上限（超过判为非人声）。 */
        private const val SFM_MAX = 0.75f
        /** 人声带（120~3600Hz）能量占比下限。 */
        private const val VOICE_BAND_MIN = 0.35f
    }
}
