package com.example.localbuddy

import android.util.Log
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 16k mono PCM16 WAV 读写与拼接。标准 44 字节头（PCM/mono/16bit）。
 *
 * 用途：把录音每段 PCM 落盘成独立 WAV（[writeWav]），停止时把各段 WAV 流式拼接成
 * 完整音频文件（[concatWavs]），供"原始音频展示"与"下载完整音频"使用。所有文件统一
 * 16k mono 16bit，拼接后格式一致、可直接播放/下载。
 */
object WavIo {
    private const val TAG = "WavIo"
    const val SAMPLE_RATE = 16_000
    private const val CHANNELS = 1
    private const val BITS_PER_SAMPLE = 16

    /** 把 float[-1,1] PCM 写成 WAV 文件（16k mono 16bit）。 */
    fun writeWav(path: String, pcm: FloatArray, sampleRate: Int = SAMPLE_RATE) {
        val file = File(path)
        file.parentFile?.mkdirs()
        val dataLen = pcm.size * 2
        FileOutputStream(file).use { fos ->
            fos.write(makeHeader(dataLen, sampleRate))
            val samples = ByteBuffer.allocate(dataLen).order(ByteOrder.LITTLE_ENDIAN)
            for (s in pcm) {
                val v = (s * 32767f).toInt().coerceIn(-32768, 32767)
                samples.putShort(v.toShort())
            }
            fos.write(samples.array())
        }
    }

    /**
     * 把多个 WAV 的 data 区流式拼接成一个完整 WAV。逐段读取、不一次性载入内存，
     * 适合长录音。各段须同为 16k mono 16bit（本仓库写入的都是此格式）。
     */
    fun concatWavs(outPath: String, paths: List<String>, sampleRate: Int = SAMPLE_RATE) {
        if (paths.isEmpty()) return
        val file = File(outPath)
        file.parentFile?.mkdirs()
        RandomAccessFile(file, "rw").use { raf ->
            raf.setLength(0)
            // 先写占位 header（size=0），追加完 data 后回填
            raf.write(makeHeader(0, sampleRate))
            var dataLen = 0L
            for (p in paths) {
                dataLen += appendData(raf, p)
            }
            raf.seek(4)
            writeLEInt(raf, (36 + dataLen).toInt())
            raf.seek(40)
            writeLEInt(raf, dataLen.toInt())
        }
    }

    /** 构造 44 字节 WAV header（dataLen 为 data 区字节数；可传 0 后回填）。 */
    private fun makeHeader(dataLen: Int, sampleRate: Int): ByteArray {
        val buf = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray())
        buf.putInt(36 + dataLen)
        buf.put("WAVE".toByteArray())
        buf.put("fmt ".toByteArray())
        buf.putInt(16)                                   // fmt chunk size
        buf.putShort(1)                                  // PCM
        buf.putShort(CHANNELS.toShort())
        buf.putInt(sampleRate)
        buf.putInt(sampleRate * CHANNELS * BITS_PER_SAMPLE / 8) // byte rate
        buf.putShort((CHANNELS * BITS_PER_SAMPLE / 8).toShort()) // block align
        buf.putShort(BITS_PER_SAMPLE.toShort())
        buf.put("data".toByteArray())
        buf.putInt(dataLen)
        return buf.array()
    }

    /** 读 path 的 data chunk，原样追加写入 raf，返回写入字节数。跳过非 data 块。 */
    private fun appendData(raf: RandomAccessFile, path: String): Long {
        var written = 0L
        FileInputStream(path).use { fis ->
            val dis = DataInputStream(fis)
            // 跳过 RIFF header（RIFF + size + WAVE = 12 字节）
            dis.skipBytes(12)
            while (true) {
                val idBytes = ByteArray(4)
                if (dis.read(idBytes) < 4) break
                val size = readLEInt(dis).toLong()
                if (String(idBytes) == "data") {
                    var remaining = size
                    val buf = ByteArray(16 * 1024)
                    while (remaining > 0) {
                        val toRead = minOf(buf.size.toLong(), remaining).toInt()
                        val n = dis.read(buf, 0, toRead)
                        if (n < 0) break
                        raf.write(buf, 0, n)
                        written += n
                        remaining -= n
                    }
                    return written
                } else {
                    var skip = size
                    while (skip > 0) {
                        val s = dis.skip(skip)
                        if (s <= 0) break
                        skip -= s
                    }
                    if (size % 2 != 0L) dis.read()   // 块对齐填充字节
                }
            }
        }
        Log.w(TAG, "no data chunk in $path")
        return written
    }

    private fun writeLEInt(raf: RandomAccessFile, v: Int) {
        raf.write(v and 0xFF)
        raf.write((v shr 8) and 0xFF)
        raf.write((v shr 16) and 0xFF)
        raf.write((v shr 24) and 0xFF)
    }

    private fun readLEInt(dis: DataInputStream): Int {
        val b0 = dis.read(); val b1 = dis.read(); val b2 = dis.read(); val b3 = dis.read()
        return (b0 and 0xFF) or ((b1 and 0xFF) shl 8) or
            ((b2 and 0xFF) shl 16) or ((b3 and 0xFF) shl 24)
    }
}
