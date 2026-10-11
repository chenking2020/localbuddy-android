package com.example.localbuddy

import android.util.Log
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin

/**
 * 128-mel log-mel 频谱提取器，匹配 Qwen3-Audio thinker conv_frontend 所需的
 * HuggingFace **WhisperFeatureExtractor** 输入（依据 assets/asr/preprocessor_config.json）。
 *
 *   n_mels=128, sr=16000, n_fft=400, hop=160, win=400, Hann 窗,
 *   center=True(reflect-pad) -> nFrames = N / hop（openai-whisper 对 stft 结果做 [..,:-1] 丢尾帧，等价 N/hop）,
 *   功率谱(去 Nyquist -> 200 频点), log10, Whisper 归一化( max(x, x.max-8) 后 (+4)/4 ),
 *   **Slaney 面积归一化**（每三角滤波器 ÷ 其带宽 2/(right-left)，等价 librosa norm='slaney'，
 *   WhisperFeatureExtractor 训练尺度）。
 *
 *   ⚠️ mel 滤波器组必须用 **Slaney** 尺度（librosa htk=False，WhisperFeatureExtractor 训练尺度），
 *      而非 HTK（2595·log10）。早期 n_fft=1024/ln() 以及后来误用 HTK 尺度，都会让 128 个三角
 *      滤波器的中心频率整体错位，conv_frontend 拿到错误频谱、decoder 输出乱码（如 "language None"）。
 *   ⚠️ 滤波器必须做 **Slaney 面积归一化**。缺归一化时滤波器幅度整体偏大（高频尤甚），mel 尺度错误，
 *      实测 conv_frontend->encoder->decoder 对清晰人声只输出空/特殊 token（ASR 全段返回空，任意语种均失败）。
 *      加归一化后正确转写。等价 librosa norm='slaney'（WhisperFeatureExtractor 默认）。
 *
 *   ⚡ 频谱计算用 **Bluestein FFT**（n_fft=400 非二次幂，Bluestein 转成 1024 点 radix-2 FFT）。
 *      早期实现是朴素 O(n²) 标量 DFT，且每个频点都重算 `sample*hann`（冗余 200×），8s 音频 mel 提取
 *      在手机上要数百 ms~1s+，是实时转写跟不上的主因。FFT 后 DFT 核心算量降 ~40×，mel 提取降到 ~0.1s。
 *   ⚡ FFT 与 mel 累加用 **double** 精度（仅最终输出转 float32）。朴素实现是 float32，DFT 累加误差 ~1e-6；
 *      double FFT 误差 ~1e-12，比朴素更贴近 librosa/训练分布（WhisperFeatureExtractor 内部用 double 算 FFT），
 *      故不会引入转写质量回退（只会更准）。已用 Python 对照 np.fft.fft 验证，并在 [init] 自检兜底移植正确性。
 *
 * 输入：16k 单声道 Float32 PCM（[-1,1]）。输出：行优先 [n_frames, n_mels] 的扁平 FloatArray。
 */
object MelFeatures {

    const val SAMPLE_RATE = 16_000
    const val N_MELS = 128
    const val N_FFT = 400
    const val HOP_LENGTH = 160
    const val WIN_LENGTH = N_FFT                 // Whisper win_length = n_fft
    private const val F_MIN = 0f
    private const val F_MAX = SAMPLE_RATE / 2f   // 8000 Hz
    private const val LOG_FLOOR = 1e-10
    /** 去掉 Nyquist 后的频点数 = n_fft/2 = 200（对应 torch stft 的 spec[..., :-1]）。 */
    const val N_FREQS = N_FFT / 2

    // ──────────────────────────────────────────────────────────────
    // Slaney mel 尺度常量（librosa htk=False）-- 与 WhisperFeatureExtractor 训练时一致。
    // ──────────────────────────────────────────────────────────────
    private const val F_SP = 200f / 3f
    private const val MIN_LOG_HZ = 1000f
    private const val MIN_LOG_MEL = MIN_LOG_HZ / F_SP   // = 15f
    private val LOGSTEP = (ln(6.4) / 27.0).toFloat()

    private val hann = FloatArray(WIN_LENGTH) { i ->
        0.5f * (1f - cos(2f * PI.toFloat() * i / WIN_LENGTH.toFloat()))
    }

    /** N_FREQS(200) × N_MELS(128) 的三角滤波器组（Slaney mel 尺度 + Slaney 面积归一化，匹配 Whisper）。 */
    private val melFilterbank: Array<FloatArray>
    /** 每个 mel 滤波器的非零频点区间 [kStart, kEnd)，累加只遍历此区间（稀疏优化）。 */
    private val melRanges: Array<IntArray>

    // ──────────────────────────────────────────────────────────────
    // Bluestein FFT 预计算（N=N_FFT=400 -> M=1024，下一个 ≥ 2N-1=799 的二次幂），全 double 精度。
    // X[k]=Σ_{n=0}^{N-1} x[n] e^{-2πi kn/N}，k=0..N-1
    //   a[n]=x[n]·w[n]，w[m]=e^{-πi m²/N}
    //   b[m]=e^{+πi m²/N}（m 偶函数，卷积核），放成循环缓冲
    //   X[k]=w[k]·(a⊛b)[k]，线性卷积用 M 点 FFT 求
    // ──────────────────────────────────────────────────────────────
    private const val FFT_M = 1024
    private const val invM = 1.0 / FFT_M
    /** radix-2 蝶形旋转因子 e^{-2πi t/M}，t=0..M/2-1。 */
    private val twCos = DoubleArray(FFT_M / 2)
    private val twSin = DoubleArray(FFT_M / 2)
    /** chirp w[m]=e^{-πi m²/N}（注意虚部取负：e^{-iθ}=cosθ - i·sinθ）。 */
    private val chirpRe = DoubleArray(N_FFT)
    private val chirpIm = DoubleArray(N_FFT)
    /** kern = FFT(b)，b 为卷积核；每帧固定，预计算一次。 */
    private val kernRe = DoubleArray(FFT_M)
    private val kernIm = DoubleArray(FFT_M)

    init {
        // mel 滤波器组 + 非零区间（稀疏累加用）
        val (fb, ranges) = buildMelFilterbank()
        melFilterbank = fb
        melRanges = ranges
        // 旋转因子
        for (t in 0 until FFT_M / 2) {
            val ang = -2.0 * PI * t / FFT_M
            twCos[t] = cos(ang)
            twSin[t] = sin(ang)
        }
        // chirp w[m]=e^{-πi m²/N}：re=cos, im=-sin
        for (m in 0 until N_FFT) {
            val ang = PI * m * m / N_FFT.toDouble()
            chirpRe[m] = cos(ang)
            chirpIm[m] = -sin(ang)
        }
        // 卷积核 b[m]=e^{+πi m²/N}（re=cos, im=+sin），m=0..N-1；负索引按偶函数镜像到 M-m
        val bRe = DoubleArray(FFT_M)
        val bIm = DoubleArray(FFT_M)
        for (m in 0 until N_FFT) {
            val ang = PI * m * m / N_FFT.toDouble()
            bRe[m] = cos(ang)
            bIm[m] = sin(ang)
        }
        for (m in 1 until N_FFT) {
            bRe[FFT_M - m] = bRe[m]
            bIm[FFT_M - m] = bIm[m]
        }
        fftRadix2(bRe, bIm)            // kern = FFT(b)
        for (i in 0 until FFT_M) { kernRe[i] = bRe[i]; kernIm[i] = bIm[i] }
        selfTest()
    }

    /** 迭代 radix-2 FFT（位反转 + 蝶形），in-place 作用于长度为二次幂的 re/im（double）。 */
    private fun fftRadix2(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        // 位反转置换
        var j = 0
        for (i in 1 until n) {
            var bit = n ushr 1
            while ((j and bit) != 0) { j = j xor bit; bit = bit ushr 1 }
            j = j or bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var length = 2
        while (length <= n) {
            val half = length ushr 1
            val stride = n / length          // = M/length，二次幂
            var i = 0
            while (i < n) {
                for (k in 0 until half) {
                    val ti = k * stride
                    val wr = twCos[ti]
                    val wi = twSin[ti]
                    val a = i + k
                    val b = i + k + half
                    val ur = re[a]; val ui = im[a]
                    val vr = re[b] * wr - im[b] * wi
                    val vi = re[b] * wi + im[b] * wr
                    re[a] = ur + vr; im[a] = ui + vi
                    re[b] = ur - vr; im[b] = ui - vi
                }
                i += length
            }
            length = length shl 1
        }
    }

    /**
     * 计算一帧（长度 N_FFT 的实数）的功率谱前 N_FREQS 个频点，via Bluestein（double 精度）。
     * [aRe]/[aIm] 为长度 FFT_M 的复用暂存（调用方分配、跨帧复用，避免每帧分配）。结果写 [out]（double）。
     */
    private fun dftPower(frame: FloatArray, aRe: DoubleArray, aIm: DoubleArray, out: DoubleArray) {
        // a = frame · chirp，补零到 M
        for (m in 0 until N_FFT) {
            val s = frame[m].toDouble()
            aRe[m] = s * chirpRe[m]
            aIm[m] = s * chirpIm[m]
        }
        for (m in N_FFT until FFT_M) { aRe[m] = 0.0; aIm[m] = 0.0 }
        fftRadix2(aRe, aIm)                                  // A = FFT(a)
        // C = A · kern（elementwise 复数乘），就地写回 a
        for (m in 0 until FFT_M) {
            val ar = aRe[m]; val ai = aIm[m]
            val br = kernRe[m]; val bi = kernIm[m]
            aRe[m] = ar * br - ai * bi
            aIm[m] = ar * bi + ai * br
        }
        // conv = IFFT(C) = (1/M)·conj(FFT(conj(C)))
        for (m in 0 until FFT_M) aIm[m] = -aIm[m]
        fftRadix2(aRe, aIm)
        for (m in 0 until FFT_M) {
            aIm[m] = -aIm[m]
            aRe[m] *= invM
            aIm[m] *= invM
        }
        // X[k] = chirp[k] · conv[k]，取功率
        for (k in 0 until N_FREQS) {
            val cr = aRe[k]; val ci = aIm[k]
            val xr = chirpRe[k] * cr - chirpIm[k] * ci
            val xi = chirpRe[k] * ci + chirpIm[k] * cr
            out[k] = xr * xr + xi * xi
        }
    }

    /** 移植自检：Bluestein 功率谱 vs 朴素 DFT（均 double），不一致告警（仅 init 一次，开销可忽略）。 */
    private fun selfTest() {
        val x = FloatArray(N_FFT) { (it + 1) * 0.013f }
        val aRe = DoubleArray(FFT_M); val aIm = DoubleArray(FFT_M)
        val pFft = DoubleArray(N_FREQS)
        dftPower(x, aRe, aIm, pFft)
        var maxRel = 0.0
        for (k in 0 until N_FREQS) {
            var re = 0.0; var im = 0.0
            for (n in 0 until N_FFT) {
                val ang = -2.0 * PI * k * n / N_FFT
                re += x[n] * cos(ang)
                im += x[n] * sin(ang)
            }
            val pn = re * re + im * im
            val rel = abs(pn - pFft[k]) / max(max(pn, pFft[k]), 1e-12)
            if (rel > maxRel) maxRel = rel
        }
        if (maxRel > 1e-9) Log.w(TAG, "FFT self-test FAILED maxRel=$maxRel (expect <1e-9)")
        else Log.i(TAG, "FFT self-test OK maxRel=$maxRel")
    }

    // Slaney mel 尺度（librosa htk=False）-- 与 WhisperFeatureExtractor 训练时一致。
    // ⚠️ 之前用 HTK 尺度（2595·log10(1+h/700)）会让 128 个三角滤波器的中心频率整体错位，
    //    conv_frontend 拿到错误频谱、decoder 输出乱码（如 "language None"）。
    //    Slaney 与 HTK 在 mel(8000) 上相差 45 vs 2840，尺度完全不同。
    private fun mel(hz: Float): Float =
        if (hz >= MIN_LOG_HZ) MIN_LOG_MEL + ln(hz / MIN_LOG_HZ) / LOGSTEP
        else hz / F_SP
    private fun invMel(m: Float): Float =
        if (m >= MIN_LOG_MEL) MIN_LOG_HZ * exp(LOGSTEP * (m - MIN_LOG_MEL))
        else F_SP * m

    private fun buildMelFilterbank(): Pair<Array<FloatArray>, Array<IntArray>> {
        val fb = Array(N_MELS) { FloatArray(N_FREQS) }
        val ranges = Array(N_MELS) { IntArray(2) }
        val melMin = mel(F_MIN)
        val melMax = mel(F_MAX)
        // n_mels+2 个等间距 mel 点
        val melPoints = FloatArray(N_MELS + 2) { i ->
            melMin + (melMax - melMin) * i.toFloat() / (N_MELS + 1).toFloat()
        }
        val hzPoints = FloatArray(N_MELS + 2) { i -> invMel(melPoints[i]) }
        // 频点 k -> k * sr / n_fft
        val fftFreqs = FloatArray(N_FREQS) { k -> k.toFloat() * SAMPLE_RATE.toFloat() / N_FFT.toFloat() }
        for (m in 0 until N_MELS) {
            val left = hzPoints[m]
            val center = hzPoints[m + 1]
            val right = hzPoints[m + 2]
            // Slaney 面积归一化：每个三角滤波器乘 2/(right-left)，等价 librosa norm='slaney'。
            // 缺它则滤波器幅度偏大、高频尤甚，mel 尺度错误 -> ASR 对清晰人声输出空（实测）。
            val enorm = if (right > left) 2.0f / (right - left) else 0f
            var kStart = N_FREQS
            var kEnd = 0
            for (k in 0 until N_FREQS) {
                val f = fftFreqs[k]
                val w = when {
                    center == left || center == right -> 0f
                    f <= left || f >= right -> 0f
                    f <= center -> (f - left) / (center - left)
                    else -> (right - f) / (right - center)
                }
                val v = w * enorm
                fb[m][k] = v
                if (v != 0f) {                          // 三角滤波器非零区间连续 [kStart, kEnd]
                    if (k < kStart) kStart = k
                    if (k > kEnd) kEnd = k
                }
            }
            ranges[m][0] = kStart
            ranges[m][1] = if (kEnd >= kStart) kEnd + 1 else kStart
        }
        return fb to ranges
    }

    /** 返回 [n_frames, N_MELS] 行优先扁平数组 + 帧数。 */
    fun compute(pcm: FloatArray): Pair<FloatArray, Int> {
        // 过短音频补零到一帧，保证 reflect padding 可行（n_fft/2 < N）
        val src = if (pcm.size < WIN_LENGTH) pcm.copyOf(WIN_LENGTH) else pcm
        val n = src.size
        val pad = N_FFT / 2                          // center padding（reflect）

        // center=True：两侧 reflect-pad n_fft/2
        val padded = FloatArray(n + 2 * pad)
        for (j in 0 until pad) padded[j] = src[pad - j]                // 左：src[pad..1]
        System.arraycopy(src, 0, padded, pad, n)
        for (j in 0 until pad) padded[pad + n + j] = src[n - 2 - j]    // 右：src[n-2..n-1-pad]

        val nFrames = n / HOP_LENGTH
        if (nFrames <= 0) return FloatArray(0) to 0

        val out = FloatArray(nFrames * N_MELS)
        // 跨帧复用的暂存（避免每帧分配）；FFT 与 mel 累加走 double，仅最终写回 float
        val winframe = FloatArray(N_FFT)
        val aRe = DoubleArray(FFT_M)
        val aIm = DoubleArray(FFT_M)
        val power = DoubleArray(N_FREQS)
        val outD = DoubleArray(nFrames * N_MELS)
        for (fr in 0 until nFrames) {
            val start = fr * HOP_LENGTH
            // 加窗一次（旧实现在 k 循环里重算 200×）-> FFT 功率谱
            for (idx in 0 until N_FFT) winframe[idx] = padded[start + idx] * hann[idx]
            dftPower(winframe, aRe, aIm, power)
            // mel 滤波 + log10（double 累加）
            val rowBase = fr * N_MELS
            for (m in 0 until N_MELS) {
                val fbRow = melFilterbank[m]
                val r = melRanges[m]
                var e = 0.0
                for (k in r[0] until r[1]) e += power[k] * fbRow[k]   // 只累加非零区间（稀疏优化）
                outD[rowBase + m] = log10(max(e, LOG_FLOOR.toDouble()))
            }
        }

        // Whisper 归一化（per-audio）：max(x, x.max-8) 后 (x+4)/4
        var mx = outD[0]
        for (i in 1 until outD.size) if (outD[i] > mx) mx = outD[i]
        val floor = mx - 8.0
        for (i in outD.indices) {
            val v = if (outD[i] < floor) floor else outD[i]
            out[i] = ((v + 4.0) / 4.0).toFloat()
        }
        return out to nFrames
    }

    private const val TAG = "MelFeatures"
}
