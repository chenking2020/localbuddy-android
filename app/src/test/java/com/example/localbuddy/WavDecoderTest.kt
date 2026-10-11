package com.example.localbuddy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sin

/**
 * WAV 自解析（[WavDecoder]）验证：设备无关，直接在 JVM 上跑。
 * 覆盖建档要求的常见 WAV 变体：PCM 16/24/32bit、IEEE float 32bit、8bit、立体声下混、
 * 非常规采样率（8k/22.05k/48k）重采样到 16k，以及"不是 WAV 时返回 null 以便回退 MediaCodec"。
 */
class WavDecoderTest {

    private fun sine(n: Int, sr: Int, amp: Float = 0.8f): FloatArray =
        FloatArray(n) { (amp * sin(2 * Math.PI * 440.0 * it / sr)).toFloat() }

    /** 造一个最小合法 WAV（fmt + data），按 [fmtTag]/[bits] 编码样本。 */
    private fun buildWav(
        fmtTag: Int,
        bits: Int,
        pcm: FloatArray,
        sr: Int = 44_100,
        ch: Int = 1
    ): ByteArray {
        val bytesPerSample = bits / 8
        val blockAlign = ch * bytesPerSample
        val data = ByteArray(pcm.size * blockAlign)
        val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        for (v in pcm) {
            val s = (v.coerceIn(-1f, 1f) * 32767f).toInt()
            repeat(ch) {
                when {
                    fmtTag == 3 && bits == 32 -> bb.putFloat(v)
                    bits == 16 -> bb.putShort(s.toShort())
                    bits == 24 -> {
                        val i = s shl 8
                        bb.put((i and 0xFF).toByte())
                        bb.put(((i shr 8) and 0xFF).toByte())
                        bb.put(((i shr 16) and 0xFF).toByte())
                    }
                    bits == 32 -> bb.putInt(s shl 16)
                    bits == 8 -> bb.put((((s shr 8) + 128) and 0xFF).toByte())
                    else -> error("unsupported test bits=$bits")
                }
            }
        }
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + data.size); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16)
            putShort(fmtTag.toShort()); putShort(ch.toShort())
            putInt(sr); putInt(sr * blockAlign); putShort(blockAlign.toShort()); putShort(bits.toShort())
            put("data".toByteArray()); putInt(data.size)
        }.array()
        return header + data
    }

    private fun decodeAll(wav: ByteArray, targetSr: Int = 16_000): FloatArray? {
        val seq = WavDecoder.decodeSequence(ByteArrayInputStream(wav), targetSr) ?: return null
        val chunks = seq.toList()
        val total = chunks.sumOf { it.size }
        val out = FloatArray(total)
        var p = 0
        for (c in chunks) {
            System.arraycopy(c, 0, out, p, c.size)
            p += c.size
        }
        return out
    }

    private fun peak(a: FloatArray): Float {
        var m = 0f
        for (v in a) if (abs(v) > m) m = abs(v)
        return m
    }

    @Test
    fun pcm16_44k_decodesTo16kMono() {
        val sr = 44_100
        val src = sine(sr, sr)                       // 1 秒
        val out = decodeAll(buildWav(1, 16, src, sr))
        assertNotNull(out)
        val n = out!!.size
        assertTrue("1s@44.1k 应重采样到约 16000 样本, got $n", n in 15_950..16_050)
        assertTrue("幅度应保留 (got ${peak(out)})", peak(out) in 0.6f..0.95f)
    }

    @Test
    fun pcm24_and_float32_and_8bit_decode() {
        val sr = 16_000
        val src = sine(sr, sr)
        for ((tag, bits) in listOf(1 to 24, 3 to 32, 1 to 32, 1 to 8)) {
            val out = decodeAll(buildWav(tag, bits, src, sr))
            assertNotNull("fmt=$tag bits=$bits 应解码成功", out)
            assertTrue("fmt=$tag bits=$bits 样本数 ${out!!.size}", out.size in 15_900..16_100)
            assertTrue("fmt=$tag bits=$bits 幅度 ${peak(out)}", peak(out) in 0.6f..1.0f)
        }
    }

    @Test
    fun stereo_downmixesToMono() {
        val sr = 48_000
        val src = sine(sr, sr)
        val out = decodeAll(buildWav(1, 16, src, sr, ch = 2))
        assertNotNull(out)
        assertTrue("立体声下混后应为单声道 16k: ${out!!.size}", out.size in 15_900..16_100)
        assertTrue("下混后幅度不应翻倍: ${peak(out)}", peak(out) in 0.6f..0.95f)
    }

    @Test
    fun oddSampleRates_resampleTo16k() {
        for (sr in intArrayOf(8_000, 22_050, 32_000, 96_000)) {
            val src = sine(sr, sr)
            val out = decodeAll(buildWav(1, 16, src, sr))
            assertNotNull("sr=$sr 应解码", out)
            val expected = sr * 16_000 / 16_000      // 1 秒源 → 目标 1 秒
            assertTrue("sr=$sr → ${out!!.size} 样本（期望约 $expected）",
                out.size in 15_800..16_200)
        }
    }

    @Test
    fun nonWavInput_returnsNull_soCallerFallsBackToMediaCodec() {
        val junk = ByteArray(64) { (it * 7).toByte() }   // 无 RIFF/WAVE 头
        assertNull(WavDecoder.decodeSequence(ByteArrayInputStream(junk), 16_000))
    }
}
