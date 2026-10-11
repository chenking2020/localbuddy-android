package com.example.localbuddy

import android.util.Log
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * ASR 前置音频增强（纯 DSP，不引入额外模型）：**三个智能体（录音翻译 / 视频字幕 / 实时听音）
 * 的识别都走 [com.example.localbuddy.SpeechMTEngine]，在此统一下游增强一次即可全部生效**。
 *
 * 背景音乐、背景说话声、环境噪声对 ASR 的伤害有两类，分别对应这里的两种处理：
 * 1. **平稳/半平稳噪声**（空调、风扇、街道、音乐底噪、设备底噪）：抬高噪声底，把人声的
 *    谐波结构掩埋掉 → 用 STFT 域**维纳/谱减增益**压低噪声主导的时频单元（[denoise]）。
 * 2. **低频隆隆 / 直流漂移 / 风噪**（<70Hz，能量常远大于人声却不携带语义）：会占掉
 *    log-mel 的动态范围（Whisper 归一化按 max-8 裁剪，噪声底一高人声细节就被压到地板）
 *    → 先用一阶高通去掉（[highPass]）。
 *
 * 设计原则（避免"降噪越降越糟"）：
 * - **只在需要时降噪**：先 [analyze] 估 SNR，≥ [DENOISE_SNR_DB]（15dB）的干净音频只做高通
 *   + 峰值归一化，一个频点都不动——谱减对干净语音只会削谐波、引入"音乐噪声"反而降准确率。
 * - **噪声谱从音频自身估计**：取段内能量最低的 20% 帧（句间停顿/气口）的平均功率谱作噪声谱，
 *   不需要额外的静音样本；配合过估计因子 beta 控制力度（SNR 越低压得越狠）。
 * - **维纳增益而非硬谱减**：G = P/(P + beta·N)，自带下限 floor，相位完全保留，
 *   比"硬减"的 P-beta·N 音乐噪声少得多；再沿时间平滑（0.55/0.45）进一步抑制抖动噪声。
 * - **增益有地板、带外才重压**：floor 保证不把语音削成怪声；仅 STRONG 模式（首轮识别为空的
 *   兜底重试）才额外压人声带外（<80Hz / >7.4kHz）。
 *
 * 实现要点：512 点 Hann、hop 256（50% 重叠）STFT；radix-2 FFT（[fft]）在 init 自检与朴素
 * DFT 对照；合成用 WOLA（窗平方和归一化，精确重建）；噪声谱估计与滤波分两遍跑，
 * 避免同时持有整段 STFT（12s 音频的复数谱 ~3MB）→ 峰值内存仅 O(N)。
 */
object AudioEnhancer {

    const val SAMPLE_RATE = 16_000

    private const val TAG = "AudioEnhancer"

    // ── STFT 参数 ──────────────────────────────────────────────
    private const val N = 512                 // 32ms @16k
    private const val HOP = 256               // 16ms，50% 重叠
    private const val PAD = N / 2             // 两端 reflect 填充长度（= HOP）
    private const val BINS = N / 2 + 1        // 257（含 DC 与 Nyquist）

    /** 人声主要能量带（Hz）：谱门（[spectralFeatures]）与 STRONG 模式带外压制都用它。 */
    private const val VOICE_LO = 120f
    private const val VOICE_HI = 3600f
    /** STRONG 模式保留的带宽：上下界之外重压（鼓点低音 / 高频嘶声与镲片）。 */
    private const val KEEP_LO = 80f
    private const val KEEP_HI = 7400f

    /** SNR 低于此值才启用降噪（dB）。15dB 以上人声结构清晰，降噪收益 < 失真代价。 */
    private const val DENOISE_SNR_DB = 15f

    /** 峰值归一化目标（与 AudioDecoder.normalizePeak 一致：0.9）。 */
    private const val PEAK_TARGET = 0.9f

    private val hann = FloatArray(N) { i -> (0.5f * (1f - cos(2.0 * PI * i / N))).toFloat() }

    init {
        selfTestFft()
        selfTestStftRoundTrip()
    }

    /** 降噪力度档位：AUTO=按 SNR 自适应；MILD=首轮；STRONG=空结果兜底重试（更狠 + 带外压制）。 */
    enum class Mode { AUTO, MILD, STRONG }

    /**
     * 噪声画像：从音频自身估出的噪声功率谱 + SNR。
     * [valid]=false 表示帧数太少/估计不可靠，调用方应跳过降噪。
     */
    class Profile(
        val noise: FloatArray,   // 长度 BINS 的噪声功率谱
        val snrDb: Float,
        val frames: Int,
        val valid: Boolean
    ) {
        /** 是否需要降噪：估计有效且 SNR 偏低（[DENOISE_SNR_DB] 以下）。 */
        val needDenoise: Boolean get() = valid && snrDb < DENOISE_SNR_DB
    }

    // ──────────────────────────────────────────────────────────────
    // 对外主入口
    // ──────────────────────────────────────────────────────────────

    /**
     * ASR 前统一预处理：去直流/高通 → 分析噪声 →（按需）维纳降噪 → 峰值归一化。
     * @return (处理后的 PCM, 噪声画像)；画像供调用方决定是否需要更强档位重试。
     */
    fun prepare(pcm: FloatArray, mode: Mode = Mode.AUTO): Pair<FloatArray, Profile> {
        if (pcm.size < N * 2) {
            // 太短（<64ms）：无统计意义，只归一化（含 ASR 空段/极限短段）
            return AudioDecoder.normalizePeak(pcm) to Profile(FloatArray(BINS), 30f, 0, false)
        }
        val hp = highPass(pcm)
        val profile = analyze(hp)
        val out = if (profile.needDenoise) denoise(hp, profile, mode) else hp
        return AudioDecoder.normalizePeak(out) to profile
    }

    /** 峰值归一化到 ±0.9（降噪后整体电平下降，必须重新归一化，否则低电平送 ASR 易被判空）。 */
    fun normalizePeak(pcm: FloatArray): FloatArray = AudioDecoder.normalizePeak(pcm)

    /**
     * 去直流 + 一阶 RC 高通（默认 70Hz）。
     * 去掉不携带语义却常占满动态范围的低频（空调/风噪/手持摩擦/音乐鼓点低音/直流偏置），
     * 让 log-mel 的动态范围留给 120Hz~3.6kHz 的人声。截止频率在男声基频（~85Hz）以下，
     * 对语音内容几乎无损。
     */
    fun highPass(pcm: FloatArray, cutoffHz: Float = 70f): FloatArray {
        if (pcm.size < 4) return pcm
        // 1) 去直流：整体减均值（麦克风/解码常见固定偏置，会让 mel 的 DC bin 爆掉）。
        //    注意不修改入参——调用方可能复用同一段 PCM（如降噪重试会用同一份数据再跑一次）
        var sum = 0.0
        for (s in pcm) sum += s.toDouble()
        val dc = (sum / pcm.size).toFloat()
        // 2) 一阶高通：y[n] = a·(y[n-1] + x[n] - x[n-1])，a = RC/(RC+dt)
        val rc = 1.0 / (2.0 * PI * cutoffHz)
        val dt = 1.0 / SAMPLE_RATE
        val a = (rc / (rc + dt)).toFloat()
        val out = FloatArray(pcm.size)
        var prevX = pcm[0] - dc
        var prevY = 0f
        out[0] = 0f
        for (i in 1 until pcm.size) {
            val x = pcm[i] - dc
            val y = a * (prevY + x - prevX)
            out[i] = y
            prevX = x
            prevY = y
        }
        return out
    }

    /**
     * 噪声分析：逐帧功率谱 → 取能量最低 20% 帧的平均谱作噪声谱，取最高 25% 帧估语音能量，
     * 二者之比即 SNR(dB)。
     *
     * 为什么用"低能量帧"而不是首 N 帧：段是 VAD 切出的一句话，句内必有气口/尾音静音，
     * 这些帧就是当前环境的真实噪声；首 N 帧假设在长音频（上传/视频）上常不成立。
     */
    fun analyze(src: FloatArray): Profile {
        if (src.size < N * 2) return Profile(FloatArray(BINS), 30f, 0, false)
        val padded = padReflect(src)
        val nFrames = (padded.size - N) / HOP + 1
        if (nFrames < 8) return Profile(FloatArray(BINS), 30f, nFrames, false)

        val power = Array(nFrames) { FloatArray(BINS) }
        val energy = FloatArray(nFrames)
        val re = FloatArray(N)
        val im = FloatArray(N)
        for (f in 0 until nFrames) {
            val off = f * HOP
            for (i in 0 until N) {
                re[i] = padded[off + i] * hann[i]
                im[i] = 0f
            }
            fft(re, im)
            var e = 0.0
            for (k in 0 until BINS) {
                val p = (re[k] * re[k] + im[k] * im[k]).toDouble()
                power[f][k] = p.toFloat()
                e += p
            }
            energy[f] = e.toFloat()
        }

        val sorted = energy.copyOf()
        sorted.sort()
        // 噪声帧：能量最低 20%（至少 2 帧）
        val noiseThr = sorted[min(max(2, (nFrames * 0.2f).toInt()), nFrames - 1)]
        val noise = FloatArray(BINS)
        var noiseE = 0.0
        var nCount = 0
        for (f in 0 until nFrames) {
            if (energy[f] <= noiseThr) {
                val p = power[f]
                for (k in 0 until BINS) noise[k] += p[k]
                noiseE += energy[f].toDouble()
                nCount++
            }
        }
        // 语音帧：能量最高 25%
        val speechThr = sorted[(nFrames * 0.75f).toInt().coerceIn(0, nFrames - 1)]
        var speechE = 0.0
        var sCount = 0
        for (f in 0 until nFrames) {
            if (energy[f] >= speechThr) {
                speechE += energy[f].toDouble()
                sCount++
            }
        }
        if (nCount == 0 || sCount == 0 || noiseE <= 1e-12) {
            return Profile(FloatArray(BINS), 30f, nFrames, false)
        }
        for (k in 0 until BINS) {
            noise[k] = (noise[k] / nCount).coerceAtLeast(1e-12f)
        }
        noiseE /= nCount
        speechE /= sCount
        val snrDb = (10.0 * log10(max(speechE - noiseE, 1e-12) / max(noiseE, 1e-12)))
            .toFloat().coerceIn(-20f, 60f)
        return Profile(noise, snrDb, nFrames, true)
    }

    /**
     * 维纳降噪：对每帧算 G(k) = max(P/(P + beta·N), floor)，沿时间平滑后乘到复数谱（保相位），
     * WOLA 合成回时域。
     * @param mode AUTO=按 SNR 自适应；MILD/STRONG 直接指定力度（STRONG 额外压制人声带外）。
     */
    fun denoise(src: FloatArray, profile: Profile, mode: Mode = Mode.AUTO): FloatArray {
        if (!profile.valid || src.size < N * 2) return src
        val beta = betaFor(profile.snrDb, mode)
        if (beta <= 0f) return src
        val floor = if (mode == Mode.STRONG) 0.10f else 0.18f
        val bandKill = mode == Mode.STRONG
        val prevG = FloatArray(BINS) { 1f }

        return stftProcess(src) { re, im, _ ->
            for (k in 0 until BINS) {
                val p = re[k] * re[k] + im[k] * im[k]
                var g = if (p > 1e-12f) p / (p + beta * profile.noise[k]) else floor
                if (g < floor) g = floor
                // 时间平滑：抑制逐帧抖动造成的"音乐噪声"（增益跳变听起来像流水声，也干扰 mel）
                g = 0.55f * prevG[k] + 0.45f * g
                prevG[k] = g
                if (bandKill) {
                    val hz = k * SAMPLE_RATE.toFloat() / N
                    if (hz < KEEP_LO || hz > KEEP_HI) g *= 0.12f
                }
                re[k] *= g
                im[k] *= g
                // 负频率镜像（k=0 与 k=N/2 无镜像），保证 IFFT 后是实信号
                val mk = N - k
                if (k > 0 && k < N / 2) {
                    re[mk] *= g
                    im[mk] *= g
                }
            }
            ifft(re, im)
        }
    }

    /**
     * STFT → 逐帧处理（[block] 收到的 re/im 是**已加窗并做完 FFT 的当前帧**，需自行 ifft）→ WOLA 合成。
     * 降噪与 init 自检共用同一套骨架，避免"降噪用的 OLA"和"验证用的 OLA"两份实现走样。
     *
     * 合成用 WOLA（窗平方和归一化）：Hann + hop=N/2 时 Σw² 非常数，必须逐点除以窗平方和
     * （librosa istft 同款做法）才能精确重建，否则会有 50% 的周期性幅度调制，等于给 mel 加干扰。
     */
    private inline fun stftProcess(
        src: FloatArray,
        block: (re: FloatArray, im: FloatArray, frame: Int) -> Unit
    ): FloatArray {
        val padded = padReflect(src)
        val nFrames = (padded.size - N) / HOP + 1
        val acc = FloatArray(src.size)
        val norm = FloatArray(src.size)
        val re = FloatArray(N)
        val im = FloatArray(N)
        for (f in 0 until nFrames) {
            val off = f * HOP
            for (i in 0 until N) {
                re[i] = padded[off + i] * hann[i]
                im[i] = 0f
            }
            fft(re, im)
            block(re, im, f)
            // 写回位置要减掉前导 pad：帧 f 的样本 i 对应原始索引 (off + i - PAD)。
            // 少了这个偏移，重建波形会整体后移 PAD 个样本（首尾还会被 reflect 填充顶掉）。
            for (i in 0 until N) {
                val j = off + i - PAD
                if (j < 0 || j >= src.size) continue
                val w = hann[i]
                acc[j] += re[i] * w
                norm[j] += w * w
            }
        }
        val out = FloatArray(src.size)
        for (i in out.indices) {
            val nz = norm[i]
            // 窗平方和极小的点（首尾边界的"窗零点"）无信息，直接取原样本，避免 0/0
            out[i] = if (nz > 1e-10f) acc[i] / nz else src[i]
        }
        return out
    }

    /**
     * 帧谱特征，供 VAD 区分「人声」与「宽带噪声 / 音乐伴奏」（能量门限之上再过一道谱门）：
     * @return (谱平坦度 SFM, 人声带能量占比)；null=帧太短/空帧（调用方按能量结果处理）。
     * - **SFM（谱平坦度）** = 几何均值/算术均值 ∈ (0,1]：人声谐波+共振峰结构强 → 低（0.02~0.3）；
     *   白噪声/嘶声 → 高（0.6~1.0）；多数音乐伴奏居中偏高。
     * - **人声带占比** = 120~3600Hz 能量/全带能量：人声通常 ≥0.5；低频隆隆、鼓点、高频镲片
     *   主导的音乐/噪声 → 明显更低。
     */
    fun spectralFeatures(frame: FloatArray): Pair<Float, Float>? = FrameSpectrum().analyze(frame)

    /**
     * 逐帧谱分析器（**复用缓冲**：VAD 每 16~32ms 调一次，不能每帧分配数组）。
     * 单线程使用（每条音频流一个实例）。
     */
    class FrameSpectrum {
        private var re = FloatArray(512)
        private var im = FloatArray(512)

        /**
         * @return (谱平坦度 SFM, 人声带能量占比) 或 null（帧太短/空帧）。
         * 含义与阈值建议见 [spectralFeatures]。
         */
        fun analyze(frame: FloatArray, n: Int = frame.size): Pair<Float, Float>? {
            if (n < 64) return null
            val size = nextPow2(n)
            if (re.size < size) {
                re = FloatArray(size)
                im = FloatArray(size)
            }
            val r = re
            val m = im
            for (i in 0 until size) { r[i] = 0f; m[i] = 0f }
            // 帧长相关的 Hann 窗（帧长由 VAD 决定：256/480/512）
            for (i in 0 until n) {
                r[i] = (frame[i] * (0.5f * (1f - cos(2.0 * PI * i / n)))).toFloat()
            }
            fft(r, m)
            val bins = size / 2
            var total = 0.0
            var voice = 0.0
            var logSum = 0.0
            for (k in 1..bins) {
                val p = (r[k] * r[k] + m[k] * m[k]).toDouble()
                total += p
                val hz = k * SAMPLE_RATE.toFloat() / size
                if (hz >= VOICE_LO && hz <= VOICE_HI) voice += p
                logSum += ln(p + 1e-20)
            }
            if (total <= 1e-20) return null
            val mean = total / bins
            val sfm = (exp(logSum / bins) / mean).toFloat().coerceIn(0f, 1f)
            val ratio = (voice / total).toFloat().coerceIn(0f, 1f)
            return sfm to ratio
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 内部
    // ──────────────────────────────────────────────────────────────

    /** 过估计因子：SNR 越低压得越狠；干净（≥15dB）返回 0（不降噪）。STRONG 额外加力。 */
    private fun betaFor(snrDb: Float, mode: Mode): Float {
        val base = when {
            snrDb >= DENOISE_SNR_DB -> 0f
            snrDb >= 10f -> 1.2f
            snrDb >= 6f -> 1.8f
            snrDb >= 3f -> 2.4f
            snrDb >= 0f -> 3.0f
            else -> 3.6f
        }
        return when (mode) {
            Mode.AUTO -> base
            Mode.MILD -> if (base <= 0f) 0f else min(base, 2.4f)
            Mode.STRONG -> if (base <= 0f) 2.0f else (base * 1.6f).coerceAtMost(5.0f)
        }
    }

    /** 两端对称（reflect）填充 N/2，使 STFT 覆盖首尾样本且 WOLA 能精确重建。 */
    private fun padReflect(src: FloatArray): FloatArray {
        val pad = N / 2
        val out = FloatArray(src.size + 2 * pad)
        for (j in 0 until pad) out[j] = src[pad - j]
        System.arraycopy(src, 0, out, pad, src.size)
        for (j in 0 until pad) out[pad + src.size + j] = src[src.size - 2 - j]
        return out
    }

    private fun nextPow2(n: Int): Int {
        var p = 1
        while (p < n) p = p shl 1
        return p
    }

    /** 原地 radix-2 FFT（要求长度为 2 的幂）。正向变换；[ifft] 用共轭法取逆。 */
    private fun fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n ushr 1
            while ((j and bit) != 0) {
                j = j xor bit
                bit = bit ushr 1
            }
            j = j or bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wr = cos(ang).toFloat()
            val wi = sin(ang).toFloat()
            val half = len shr 1
            var i = 0
            while (i < n) {
                var cr = 1f
                var ci = 0f
                for (k in 0 until half) {
                    val a = i + k
                    val b = a + half
                    val ur = re[a]; val ui = im[a]
                    val vr = re[b] * cr - im[b] * ci
                    val vi = re[b] * ci + im[b] * cr
                    re[a] = ur + vr; im[a] = ui + vi
                    re[b] = ur - vr; im[b] = ui - vi
                    val nr = cr * wr - ci * wi
                    ci = cr * wi + ci * wr
                    cr = nr
                }
                i += len
            }
            len = len shl 1
        }
    }

    /** 逆 FFT：共轭 → 正变换 → 共轭并除以 N。 */
    private fun ifft(re: FloatArray, im: FloatArray) {
        for (i in im.indices) im[i] = -im[i]
        fft(re, im)
        val inv = 1f / re.size
        for (i in re.indices) {
            re[i] *= inv
            im[i] = -im[i] * inv
        }
    }

    /** FFT 自检：与朴素 DFT 对照（仅 init 一次，~0.3ms），不一致则告警（移植错误会导致降噪失真）。 */
    private fun selfTestFft() {
        val n = 256
        val x = FloatArray(n) { (0.37f * (it + 1) % 1.0f) - 0.5f }
        val re = x.copyOf()
        val im = FloatArray(n)
        fft(re, im)
        var maxRel = 0.0
        for (k in 0 until 8) {           // 只校验前 8 个频点，够覆盖实现错误
            var sr = 0.0
            var si = 0.0
            for (t in 0 until n) {
                val ang = -2.0 * PI * k * t / n
                sr += x[t] * cos(ang)
                si += x[t] * sin(ang)
            }
            val d = kotlin.math.abs(sr - re[k]) + kotlin.math.abs(si - im[k])
            val mag = kotlin.math.abs(sr) + kotlin.math.abs(si) + 1e-9
            maxRel = max(maxRel, d / mag)
        }
        if (maxRel > 1e-3) Log.w(TAG, "FFT self-test FAILED maxRel=$maxRel (expect <1e-3)")
        else Log.i(TAG, "FFT self-test OK maxRel=$maxRel")
    }

    /**
     * STFT 往返自检：只做 FFT→IFFT（增益恒 1）应无损还原原始波形。
     * 这一步错了（FFT/镜像/ifft/OLA 任一处）降噪就会把语音削成怪声 → 宁可在 init 就报警。
     */
    private fun selfTestStftRoundTrip() {
        val n = 4096
        val x = FloatArray(n) { (0.37f * (it + 1) % 1.0f) - 0.5f }
        val y = stftProcess(x) { re, im, _ -> ifft(re, im) }
        var maxErr = 0.0
        for (i in 0 until n) maxErr = max(maxErr, kotlin.math.abs(y[i] - x[i]).toDouble())
        if (maxErr > 1e-3) Log.w(TAG, "STFT round-trip FAILED maxErr=$maxErr (expect <1e-3)")
        else Log.i(TAG, "STFT round-trip OK maxErr=$maxErr")
    }
}
