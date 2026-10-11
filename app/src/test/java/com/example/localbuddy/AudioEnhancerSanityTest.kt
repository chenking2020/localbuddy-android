package com.example.localbuddy

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/** AudioEnhancer 数值自检（临时验证用）。 */
class AudioEnhancerSanityTest {

    private fun rms(x: FloatArray, from: Int = 0, to: Int = x.size): Double {
        var s = 0.0
        for (i in from until to) s += (x[i] * x[i]).toDouble()
        return sqrt(s / (to - from))
    }

    @Test
    fun highPass_removesDc_keepsVoiceBand() {
        val dc = FloatArray(16000) { 0.5f }
        val outDc = AudioEnhancer.highPass(dc.copyOf())
        assertTrue("DC removed, rms=${rms(outDc)}", rms(outDc) < 0.02)

        val tone = FloatArray(16000) { i -> (0.5 * sin(2 * PI * 1000.0 * i / 16000.0)).toFloat() }
        val outTone = AudioEnhancer.highPass(tone.copyOf())
        val a = rms(tone, 1000, 15000)
        val b = rms(outTone, 1000, 15000)
        assertTrue("1kHz preserved: $a -> $b", b > a * 0.9 && b < a * 1.1)
    }

    @Test
    fun denoise_dropsNoiseFloor_keepsSpeech() {
        val sr = 16000
        val rnd = Random(42)
        // 噪声 ±0.2（power 0.0133）+ 50% 占空的谐波串（power 0.0475）→ 理论 SNR ≈ 5.5dB
        val x = FloatArray(sr * 2) { (rnd.nextFloat() - 0.5f) * 0.4f }
        for (i in (sr * 0.4).toInt() until (sr * 1.4).toInt()) {
            val t = i / sr.toDouble()
            x[i] += (0.25 * sin(2 * PI * 300 * t) + 0.15 * sin(2 * PI * 600 * t) +
                0.10 * sin(2 * PI * 900 * t)).toFloat()
        }
        val hp = AudioEnhancer.highPass(x.copyOf())
        val profile = AudioEnhancer.analyze(hp)
        assertTrue("snr estimate ≈5.5dB, got ${profile.snrDb}", profile.snrDb in 3f..9f)
        val out = AudioEnhancer.denoise(hp, profile, AudioEnhancer.Mode.AUTO)

        val noiseBefore = rms(hp, 0, (sr * 0.3).toInt())
        val noiseAfter = rms(out, 0, (sr * 0.3).toInt())
        val speechBefore = rms(hp, (sr * 0.5).toInt(), (sr * 1.3).toInt())
        val speechAfter = rms(out, (sr * 0.5).toInt(), (sr * 1.3).toInt())
        assertTrue("noise must drop: $noiseBefore -> $noiseAfter", noiseAfter < noiseBefore * 0.6)
        assertTrue("speech must survive: $speechBefore -> $speechAfter", speechAfter > speechBefore * 0.6)
    }

    @Test
    fun prepare_isStableOnPureSilence() {
        val silence = FloatArray(16000)
        val (out, profile) = AudioEnhancer.prepare(silence.copyOf())
        assertTrue("silence stays silence", rms(out) < 1e-6)
        assertTrue("no crash on profile", profile.frames >= 0)
    }

    @Test
    fun streamingSegmenter_capsSegmentsWhileVoiceNeverStops() {
        // 持续语音（VAD 恒 true，模拟朗读/对白长时间无停顿）：段长上限必须在语音帧分支也生效，
        // 否则段会一路累积 → 送 ASR 被 12s 截断 → 后半句全丢（真实 3 分钟有声书曾切出 26s 段）。
        val always = object : VadDetector {
            override val frameSamples: Int = 256
            override val threshold: Float = 0f
            override fun isVoice(frame: FloatArray) = true
        }
        val seg = StreamingVadSegmenter(always, pauseMs = 1200, maxSegMs = 10000)
        val chunk = FloatArray(16000) { 0.4f }
        val out = ArrayList<TimedVadSegmenter.VoiceSegment>()
        for (i in 0 until 30) out.addAll(seg.feed(chunk))   // 30s 不间断语音
        out.addAll(seg.flush())
        assertTrue("应切出多段, got ${out.size}", out.size >= 3)
        for (s in out) {
            val dur = s.endMs - s.startMs
            assertTrue("段长 ${dur}ms 应 ≤ 上限+一帧", dur <= 10_300)
        }
        val total = out.sumOf { it.endMs - it.startMs }
        assertTrue("不能因切段丢内容: 覆盖 ${total}ms / 30000ms", total >= 29_000)
    }

    @Test
    fun spectralFeatures_voiceVsNoise() {
        val sr = 16000
        val rnd = Random(7)
        val noise = FloatArray(512) { (rnd.nextFloat() - 0.5f) * 2f }
        val voiced = FloatArray(512) { i ->
            val t = i / sr.toDouble()
            (0.6 * sin(2 * PI * 200 * t) + 0.4 * sin(2 * PI * 400 * t)).toFloat()
        }
        val s = AudioEnhancer.FrameSpectrum()
        val nf = s.analyze(noise)!!
        val vf = s.analyze(voiced)!!
        assertTrue("voiced SFM lower than noise: ${vf.first} vs ${nf.first}", vf.first < nf.first)
        assertTrue("voiced band ratio higher: ${vf.second} vs ${nf.second}", vf.second > nf.second)
        assertTrue("noise is broadband-ish: sfm=${nf.first}", nf.first > 0.3f)
        assertTrue("voiced is tonal: sfm=${vf.first}", vf.first < 0.3f)
    }
}
