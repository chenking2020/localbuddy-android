package com.example.localbuddy

import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * WAV（RIFF）纯 Kotlin 解码 —— 不依赖 MediaExtractor / MediaCodec。
 *
 * 为什么要自己解：Android 的 MediaExtractor 对 WAV（audio/x-wav、audio/wav）支持**随设备与系统
 * 版本而异**，不少机器直接报 "No audio track in ..."（尤其 24bit / float / 非常规采样率），
 * 表现为"上传 wav 没反应/识别不了"。RIFF 结构极简（fmt + data），自己解析无设备兼容问题、
 * 不需要解码器，冷启动也更快。
 *
 * 支持：PCM 8/16/24/32-bit、IEEE float 32/64-bit、A-law / μ-law、
 * WAVE_FORMAT_EXTENSIBLE（按 SubFormat GUID 还原真实格式）、任意采样率与声道数；
 * 输出 16k 单声道 Float32（复用 [MonoResampler] 做抽取）。
 *
 * 不支持（返回 null，调用方回退 MediaCodec 路径）：ADPCM(2)、MP3-in-WAV(0x55) 等压缩格式。
 */
internal object WavDecoder {

    private const val FMT_PCM = 1
    private const val FMT_FLOAT = 3
    private const val FMT_ALAW = 6
    private const val FMT_MULAW = 7
    private const val FMT_EXTENSIBLE = 0xFFFE

    private class Fmt(
        val audioFormat: Int,
        val channels: Int,
        val sampleRate: Int,
        val blockAlign: Int,
        val bitsPerSample: Int
    ) {
        val bytesPerSample: Int get() = max(1, bitsPerSample / 8)
        /** 本解码器能否处理（不支持的交给 MediaCodec 兜底）。 */
        val supported: Boolean
            get() = when (audioFormat) {
                FMT_PCM -> bitsPerSample == 8 || bitsPerSample == 16 ||
                        bitsPerSample == 24 || bitsPerSample == 32
                FMT_FLOAT -> bitsPerSample == 32 || bitsPerSample == 64
                FMT_ALAW, FMT_MULAW -> bitsPerSample == 8
                else -> false
            }
    }

    /**
     * 惰性解码：返回逐块的 16k 单声道 Float32 序列（调用方遍历时才读流，不占内存）。
     * @return null = 不是 WAV / 格式本解码器不支持 → 调用方应回退 MediaCodec 路径。
     */
    fun decodeSequence(input: InputStream, targetSr: Int = 16_000): Sequence<FloatArray>? {
        val src = input.buffered(64 * 1024)
        val head = ByteArray(12)
        if (readFully(src, head) < 12) return null
        if (ascii(head, 0, 4) != "RIFF" || ascii(head, 8, 4) != "WAVE") return null

        return sequence {
            var fmt: Fmt? = null
            while (true) {
                val idBytes = ByteArray(4)
                if (readFully(src, idBytes) < 4) break
                val id = ascii(idBytes, 0, 4)
                val size = readLe32(src) ?: break
                if (id == "fmt ") {
                    val body = ByteArray(size.toInt().coerceIn(0, 4096))
                    readFully(src, body)
                    if (size > body.size) skip(src, size - body.size)
                    fmt = parseFmt(body, size.toInt())
                } else if (id == "data") {
                    val f = fmt
                    if (f == null || !f.supported || f.channels <= 0) break
                    val dec = WavStreamDecoder(f, targetSr)
                    // size 可能为 0（流式写入未回填）→ 一直读到 EOF
                    val dataLen = if (size > 0) size.toLong() else Long.MAX_VALUE
                    yieldAll(dec.pumpSequence(src, dataLen))
                    val tail = dec.flush()
                    if (tail.isNotEmpty()) yield(tail)
                    break
                } else {
                    skip(src, size)      // LIST / fact / 自定义 chunk 一律跳过
                }
            }
        }
    }

    /** 解析 fmt chunk；EXTENSIBLE 时按 SubFormat GUID 前 2 字节还原真实格式。 */
    private fun parseFmt(d: ByteArray, size: Int): Fmt? {
        if (size < 16) return null
        var audioFormat = le16(d, 0)
        val channels = le16(d, 2)
        val sampleRate = le32(d, 4)
        val blockAlign = le16(d, 12)
        val bits = le16(d, 14)
        // EXTENSIBLE 布局：基础 fmt 16B + cbSize(2) + validBits(2) + channelMask(4) = 24，
        // 第 24 字节起是 SubFormat GUID，其前 2 字节即真实 format tag
        if (audioFormat == FMT_EXTENSIBLE && size >= 40) audioFormat = le16(d, 24)
        return Fmt(audioFormat, channels, sampleRate, blockAlign, bits)
    }

    private class WavStreamDecoder(f: Fmt, targetSr: Int) {
        private val fmt = f
        private val frameBytes = if (f.blockAlign > 0) f.blockAlign
        else max(1, f.channels * f.bytesPerSample)
        private val resampler = MonoResampler(f.sampleRate, targetSr)
        private val buf = ByteArray(frameBytes * 8192)

        fun pumpSequence(input: InputStream, dataLen: Long): Sequence<FloatArray> = sequence {
            var remaining = dataLen
            while (remaining > 0) {
                val want = min(buf.size.toLong(), remaining).toInt()
                val got = readFullyUpTo(input, buf, want)
                if (got <= 0) break
                remaining -= got
                val frames = got / frameBytes
                if (frames > 0) yield(decodeFrames(frames))
                // 尾部不足一帧的残字节无法回退，直接丢弃（最多 frameBytes-1 字节）
            }
        }

        private fun decodeFrames(frames: Int): FloatArray {
            val f = fmt
            val mono = ShortArray(frames)
            var p = 0
            for (i in 0 until frames) {
                var sum = 0
                for (c in 0 until f.channels) {
                    sum += sampleAt(p)
                    p += f.bytesPerSample
                }
                mono[i] = (sum / f.channels).coerceIn(-32768, 32767).toShort()
            }
            val rs = resampler.process(mono)
            return FloatArray(rs.size) { rs[it] / 32768f }
        }

        private fun sampleAt(o: Int): Int = when (fmt.audioFormat) {
            FMT_FLOAT -> if (fmt.bitsPerSample == 64) {
                (bitsToDouble(o) * 32767.0).toInt().coerceIn(-32768, 32767)
            } else {
                (bitsToFloat(o) * 32767f).toInt().coerceIn(-32768, 32767)
            }
            FMT_ALAW -> alawToLinear(buf[o].toInt() and 0xFF)
            FMT_MULAW -> ulawToLinear(buf[o].toInt() and 0xFF)
            else -> when (fmt.bitsPerSample) {
                8 -> ((buf[o].toInt() and 0xFF) - 128) * 256        // 8bit 无符号，中心值 128
                16 -> {
                    val v = (buf[o].toInt() and 0xFF) or ((buf[o + 1].toInt() and 0xFF) shl 8)
                    if (v >= 0x8000) v - 0x10000 else v              // 符号扩展
                }
                24 -> {
                    val v = (buf[o].toInt() and 0xFF) or
                            ((buf[o + 1].toInt() and 0xFF) shl 8) or
                            ((buf[o + 2].toInt() and 0xFF) shl 16)
                    (if (v and 0x800000 != 0) v - 0x1000000 else v) shr 8
                }
                32 -> le32(buf, o) shr 16
                else -> 0
            }
        }

        private fun bitsToFloat(o: Int): Float = Float.fromBits(le32(buf, o))

        private fun bitsToDouble(o: Int): Double {
            var bits = 0L
            for (i in 0 until 8) bits = bits or ((buf[o + i].toLong() and 0xFF) shl (8 * i))
            return Double.fromBits(bits)
        }

        fun flush(): FloatArray {
            val t = resampler.flush()
            return FloatArray(t.size) { t[it] / 32768f }
        }
    }

    // ── G.711（A-law / μ-law）────────────────────────────────────────
    private fun alawToLinear(a: Int): Int {
        val t = (a xor 0x55) and 0x7F
        val seg = (t and 0x70) shr 4
        val mant = (t and 0x0F) shl 4
        val v = when (seg) {
            0 -> mant + 8
            1 -> mant + 0x108
            else -> (mant + 0x108) shl (seg - 1)
        }
        return if (a and 0x80 == 0) v else -v
    }

    private fun ulawToLinear(u: Int): Int {
        val t0 = u.inv() and 0xFF
        val seg = (t0 and 0x70) shr 4
        val t = (((t0 and 0x0F) shl 3) + 0x84) shl seg
        return if (t0 and 0x80 != 0) 0x84 - t else t - 0x84
    }

    // ── 字节工具 ──────────────────────────────────────────────────────
    private fun ascii(b: ByteArray, off: Int, len: Int) = String(b, off, len, Charsets.US_ASCII)

    private fun le16(b: ByteArray, o: Int) =
        (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, o: Int) =
        (b[o].toInt() and 0xFF) or ((b[o + 1].toInt() and 0xFF) shl 8) or
                ((b[o + 2].toInt() and 0xFF) shl 16) or ((b[o + 3].toInt() and 0xFF) shl 24)

    private fun readFully(src: InputStream, dst: ByteArray): Int {
        var off = 0
        while (off < dst.size) {
            val n = src.read(dst, off, dst.size - off)
            if (n < 0) break
            off += n
        }
        return off
    }

    private fun readFullyUpTo(src: InputStream, dst: ByteArray, want: Int): Int {
        var off = 0
        while (off < want) {
            val n = src.read(dst, off, want - off)
            if (n < 0) break
            off += n
        }
        return off
    }

    private fun readLe32(src: InputStream): Long? {
        val b = ByteArray(4)
        if (readFully(src, b) < 4) return null
        return le32(b, 0).toLong() and 0xFFFFFFFFL
    }

    private fun skip(src: InputStream, bytes: Long) {
        var left = bytes
        while (left > 0) {
            val n = src.skip(left)
            if (n <= 0) break
            left -= n
        }
    }
}
