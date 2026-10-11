package com.example.localbuddy

/**
 * 流式增量 VAD 分段器：分块喂入 16k mono Float32 PCM，用 [VadDetector] 逐帧判定语音，
 * 增量输出带**绝对时间戳**的语音片段。供长视频用——避免一次性持有整条 PCM 导致 OOM。
 * 调用方持续 [feed] PCM 块，每当检测到句尾停顿（或单段达上限）就吐出已完成的 [VoiceSegment]；
 * 结束时调 [flush] 取尾巴。
 *
 * 分段策略（连续覆盖）：段从首个语音帧开始，到真正的句尾停顿结束；段内所有帧（含被门限
 * 误判的轻声帧）都保留送 ASR，不丢帧；仅跳过纯静音区。句尾停顿/单段上限由构造参数控制，
 * 默认 1000ms / 8000ms；调用方可按场景调整——视频字幕以语言表达完整性为主传更粗的值。
 *
 * VAD 检测器由构造方注入：视频字幕用 [SileroVadDetector]（神经网络，对背景音/低电平鲁棒），
 * 模型未加载时回退 [EnergyVadDetector]。帧长由检测器决定（Silero 16k = 256 样本 = 16ms）。
 *
 * 内存峰值：当前段缓冲 ≤ maxSegMs(10s≈640KB) + 帧缓冲，不随音频总长增长。
 */
class StreamingVadSegmenter(
    private val detector: VadDetector,
    /** 句尾停顿时长（ms）：段内连续静音到该时长即判句尾、切段。 */
    private val pauseMs: Int = PAUSE_MS,
    /** 单段最长时长（ms）：持续说话不停顿时强制切段（应 ≤ ASR 12s 输入上限）。 */
    private val maxSegMs: Int = MAX_SEG_MS,
    /** 最短有效段（ms）：短于此的段直接丢弃——噪声爆音/键盘声/咳嗽/音乐鼓点触发的碎片，
     *  送进 ASR 只会产出乱码（噪声下模型常吐错字而非空），丢弃既提速又减少错字。 */
    private val minSegMs: Int = MIN_SEG_MS
) {

    private val frameSamples = detector.frameSamples              // 256（Silero）
    private val pauseFrames = pauseMs / (frameSamples * 1000 / SAMPLE_RATE)
    private val maxSegSamples = SAMPLE_RATE * maxSegMs / 1000
    private val minSegSamples = SAMPLE_RATE * minSegMs / 1000

    /** 当前 VAD 门限（诊断用；Silero=概率阈值，Energy=RMS 阈值）。 */
    val currentThreshold: Float get() = detector.threshold

    // 当前段累积
    private val segBuf = ArrayList<Float>(maxSegSamples / 2)
    private var segStartGlobalSample = 0L   // 段起始在整条音频中的绝对采样下标
    private var hasVoice = false
    private var silenceRun = 0

    // 帧缓冲（块可能不对齐帧长）
    private val pending = ArrayList<Float>(frameSamples * 2)
    private var globalSample = 0L           // 已消费的总采样数（绝对位置）

    // 前导缓冲：滚动保留最近 LEAD_IN_FRAMES 帧。语音起始时回补进段，避免字头被切——
    // 门限略高或人声渐入时，首个被判为语音的帧往往滞后于字头几十~几百毫秒，无前导会丢字头。
    private val leadInSamples = frameSamples * LEAD_IN_FRAMES
    private val recent = ArrayDeque<Float>(leadInSamples)

    /** 喂入一块 PCM，返回本块处理期间产出的已完成语音片段。 */
    fun feed(chunk: FloatArray): List<TimedVadSegmenter.VoiceSegment> {
        val out = ArrayList<TimedVadSegmenter.VoiceSegment>()
        for (s in chunk) pending.add(s)
        while (pending.size >= frameSamples) {
            val frame = FloatArray(frameSamples) { pending[it] }
            val isVoice = detector.isVoice(frame)
            if (isVoice) {
                if (!hasVoice) {              // 语音起始：开启新段，回补前导避免丢字头
                    hasVoice = true
                    if (recent.isNotEmpty()) {
                        for (s in recent) segBuf.add(s)
                        segStartGlobalSample = (globalSample - recent.size).coerceAtLeast(0L)
                    } else {
                        segStartGlobalSample = globalSample
                    }
                    silenceRun = 0
                } else {
                    silenceRun = 0
                }
                accumulateFrame(frame)
                // ⚠️ 单段上限必须在**语音帧**分支也检查：原逻辑只在"非语音帧"分支判断
                // segBuf >= maxSegSamples，于是持续说话（无停顿、无静音帧）时段会一直累积——
                // 实测 3 分钟有声书切出 26s 的段，送 ASR 被 12s 上限截断，后半句全部丢失
                // （表现为"整段识别不出来 / 只出前半句"）。此处到点即强切，保证段长 ≤ 上限。
                if (segBuf.size >= maxSegSamples) {
                    recent.clear()
                    emit()?.let { out.add(it) }
                }
            } else if (hasVoice) {
                // 句中静音：纳入段内（不切碎句子），累计静音帧
                silenceRun++
                accumulateFrame(frame)
                if (silenceRun >= pauseFrames || segBuf.size >= maxSegSamples) {
                    // 单段达上限（非句尾停顿）是强切：recent 末尾是本段语音而非静音，
                    // 清空避免下一段把它当字头前导重复计入（句尾停顿切则保留，recent 末尾是静音）
                    if (segBuf.size >= maxSegSamples) recent.clear()
                    emit()?.let { out.add(it) }
                }
            }
            // !hasVoice && 静音：纯静音区，跳过不累积
            // 滚动维护前导缓冲（无论语音/静音），始终保留最近 LEAD_IN_FRAMES 帧
            for (s in frame) {
                recent.add(s)
                if (recent.size > leadInSamples) recent.removeFirst()
            }
            pending.subList(0, frameSamples).clear()   // 移除已消费的一帧
            globalSample += frameSamples
        }
        return out
    }

    /** 结束喂入，返回末尾未发出的语音段（长度不足 [minSegMs] 的尾巴会被丢弃）。 */
    fun flush(): List<TimedVadSegmenter.VoiceSegment> {
        val out = ArrayList<TimedVadSegmenter.VoiceSegment>()
        if (hasVoice && segBuf.isNotEmpty()) emit()?.let { out.add(it) }
        return out
    }

    private fun accumulateFrame(frame: FloatArray) {
        for (s in frame) segBuf.add(s)
    }

    /** 把当前段缓冲固化成一个 [VoiceSegment] 并清空；短于 [minSegSamples] 返回 null（不送 ASR）。 */
    private fun emit(): TimedVadSegmenter.VoiceSegment? {
        val size = segBuf.size
        val startMs = segStartGlobalSample * 1000L / SAMPLE_RATE
        val endMs = (segStartGlobalSample + size) * 1000L / SAMPLE_RATE
        val raw = if (size >= minSegSamples) segBuf.toFloatArray() else null
        segBuf.clear()
        hasVoice = false
        silenceRun = 0
        return raw?.let { TimedVadSegmenter.VoiceSegment(startMs, endMs, it) }
    }

    companion object {
        private const val SAMPLE_RATE = 16_000
        /** 默认句尾停顿（ms）。 */
        private const val PAUSE_MS = 1000
        /** 默认单段上限（ms）。 */
        private const val MAX_SEG_MS = 8000
        /** 默认最短有效段（ms）：250ms 以下基本是噪声明/碎片，ASR 也吐不出有意义文本。 */
        private const val MIN_SEG_MS = 250
        /** 语音起始前导帧数：段起点比首个语音帧再往前若干帧（256样本×10 ≈ 160ms），含住字头。 */
        private const val LEAD_IN_FRAMES = 10
    }
}
