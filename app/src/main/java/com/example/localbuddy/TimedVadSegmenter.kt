package com.example.localbuddy

import android.util.Log
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 离线语音分段器：把整条 16k mono Float32 PCM 切成带时间戳的片段，供视频字幕按时间轴对齐。
 *
 * 设计目标：**不漏语音**。采用「连续覆盖」策略——
 *  - 段从「语音起始」开始（含少量前导，避免字头被切），到「真正的句尾停顿」结束；
 *  - 段内所有帧都保留送 ASR，**即使某些帧被能量门限误判为静音也不丢**（只要它夹在语音段里）；
 *  - 仅跳过「纯静音区」（从无语音出现的长静音），避免把纯静音送 ASR 浪费时间。
 *
 * 这比「逐帧判定语音/静音、静音即切段」的传统 VAD 鲁棒：后者一旦门限偏高就把轻声/
 * 换气误判静音、把一句话切碎甚至漏掉；本方案只要段里有响亮语音，中间的轻声帧一并送
 * ASR，由 ASR 决定有无内容，不靠能量门限丢帧。
 *
 * 每段最长 12s（≤ ASR 输入上限），句尾停顿 ≥ [PAUSE_MS] 即切段。
 */
object TimedVadSegmenter {

    /** 句尾停顿时长（ms）：段内出现连续静音到该时长即判句尾、切段。对齐 MicRecorder（录音翻译）。 */
    private const val PAUSE_MS = 1000
    /** 单段最长时长（ms）：持续说话不停顿时强制切段。对齐 MicRecorder，且 ≤ ASR 12s 上限。 */
    private const val MAX_SEG_MS = 8000
    /** 帧长（ms）：能量计算粒度。 */
    private const val FRAME_MS = 30
    /** 噪声倍数：语音/静音门限 = 底噪分位 × 此值。对齐 MicRecorder=3.0，门限清晰区分语音/静音，
     *  段在真正句尾停顿处切断（而非被强切混入静音导致 ASR 判空丢段）。 */
    private const val NOISE_MULTIPLIER = 3.0f
    /** 绝对语音 RMS 下限。对齐 MicRecorder=0.012。 */
    private const val ABSOLUTE_VOICE_RMS = 0.012f
    /** 语音起始前导帧数（×FRAME_MS）：段起点比首个语音帧再往前一点，含住字头。 */
    private const val LEAD_IN_FRAMES = 5

    private const val SAMPLE_RATE = 16_000
    private const val TAG = "TimedVadSegmenter"

    /** 一段带时间戳的语音片段。 */
    data class VoiceSegment(val startMs: Long, val endMs: Long, val pcm: FloatArray)

    /**
     * 连续覆盖分段。返回按时间顺序的语音片段列表（纯静音区被跳过，不产出空段）。
     */
    fun segment(pcm: FloatArray): List<VoiceSegment> {
        if (pcm.isEmpty()) return emptyList()
        val frameSamples = SAMPLE_RATE * FRAME_MS / 1000
        val pauseFrames = PAUSE_MS / FRAME_MS
        val maxSegFrames = SAMPLE_RATE * MAX_SEG_MS / 1000 / frameSamples

        val frameCount = pcm.size / frameSamples
        if (frameCount == 0) return emptyList()
        // 各帧 RMS
        val rmsArr = FloatArray(frameCount) { fi ->
            var sumSq = 0.0
            val base = fi * frameSamples
            for (i in 0 until frameSamples) {
                val s = pcm[base + i]
                sumSq += (s * s).toDouble()
            }
            sqrt(sumSq / frameSamples).toFloat()
        }

        // 底噪 = RMS 分布 10 分位；门限仅用于识别「句尾停顿」（静音段），不用于丢帧
        val sorted = rmsArr.sortedArray()
        val noiseLevel = sorted[(sorted.size * 0.10f).toInt().coerceIn(0, sorted.size - 1)]
        val threshold = max(noiseLevel * NOISE_MULTIPLIER, ABSOLUTE_VOICE_RMS)
        Log.i(TAG, "VAD frames=$frameCount noise=%.4f thr=%.4f".format(noiseLevel, threshold))

        val segments = ArrayList<VoiceSegment>()
        var segStartFrame = 0      // 当前段的起始帧（含前导）
        var silenceRun = 0         // 当前段内连续静音帧数
        var hasVoice = false       // 当前段是否已出现过语音帧
        for (fi in 0 until frameCount) {
            val isVoice = rmsArr[fi] >= threshold
            if (isVoice) {
                if (!hasVoice) {
                    // 语音起始：段起点设为略往前（含字头），丢弃前面长静音
                    segStartFrame = maxOf(segStartFrame, fi - LEAD_IN_FRAMES)
                    hasVoice = true
                }
                silenceRun = 0
            } else {
                silenceRun++
            }

            val segFrames = fi + 1 - segStartFrame
            // 句尾停顿（段内已有语音）或 单段过长 → 切段。段内含末尾静音，连续不漏。
            if (hasVoice && (silenceRun >= pauseFrames || segFrames >= maxSegFrames)) {
                emitRange(segments, pcm, segStartFrame, fi + 1, frameSamples)
                segStartFrame = fi + 1
                silenceRun = 0
                hasVoice = false
            }
        }
        // 末尾段：若有语音则发出（纯静音尾巴丢弃）
        if (hasVoice && segStartFrame < frameCount) {
            emitRange(segments, pcm, segStartFrame, frameCount, frameSamples)
        }
        Log.i(TAG, "VAD produced ${segments.size} segments")
        return segments
    }

    /** 把 [startFrame, endFrame) 范围的帧固化为一个 [VoiceSegment]。 */
    private fun emitRange(
        out: ArrayList<VoiceSegment>,
        pcm: FloatArray,
        startFrame: Int,
        endFrame: Int,
        frameSamples: Int
    ) {
        val s = (startFrame * frameSamples).coerceAtLeast(0)
        val e = (endFrame * frameSamples).coerceAtMost(pcm.size)
        if (e <= s) return
        val startMs = s * 1000L / SAMPLE_RATE
        val endMs = e * 1000L / SAMPLE_RATE
        out.add(VoiceSegment(startMs, endMs, pcm.copyOfRange(s, e)))
    }
}
