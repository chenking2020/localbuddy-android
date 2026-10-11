package com.example.localbuddy

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.graphics.SurfaceTexture
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 视频字幕硬烧管线（纯 Android MediaCodec + GLES，无第三方依赖）。
 *
 * 流程：MediaExtractor 拆出视频轨与音频轨 →
 *   视频：decoder 输出到 SurfaceTexture(OES 纹理) → GL 绘制「视频帧 + 当前时间点字幕」 →
 *        encoder（H.264，输入 Surface）→ MediaMuxer；
 *   音频：直接把 extractor 的原始编码 sample 透传给 MediaMuxer（不重编码，保真且省事）。
 *
 * 字幕按时间轴 [SubtitleCue] 在对应帧上叠加（Canvas 渲染文字 → RGBA 纹理 → alpha 混合），
 * 源视频的 KEY_ROTATION 元数据会被烘焙进输出（旋转 90/270 时输出宽高互换）。
 *
 * 整个 EGL/GL/编解码循环在专用 [HandlerThread] 上跑，避免阻塞调用协程；进度通过
 * [onProgress]（0f..1f）回传。
 */
object SubtitleEncoder {

    private const val TAG = "SubtitleEncoder"
    private const val TIMEOUT_US = 10_000L
    /** 等待单帧就绪的最长时间（纳秒），超时则继续（避免极端卡死）。 */
    private const val FRAME_WAIT_NS = 5_000_000_000L
    /** 编码输出长边上限：超过则等比下采样，显著降低重编码耗时。 */
    private const val MAX_ENCODE_SIDE = 1280
    /** 字幕样式（相对输出高度）：单行、贴底、字体为原一半。 */
    private const val SUB_STRIP_H_RATIO = 0.07f
    private const val SUB_FONT_SIZE_RATIO = 0.034f
    private const val SUB_BOTTOM_MARGIN_RATIO = 0.02f
    private const val SUB_MAX_WIDTH_RATIO = 0.9f

    /** 一条字幕：在 [startMs, endMs] 区间内显示 [text]。 */
    data class SubtitleCue(val startMs: Long, val endMs: Long, val text: String)

    /**
     * 把 [cues] 硬烧进 [videoUri] 指向的视频，输出到 [outFile]。
     * @param onProgress 进度回调（0..1），调用方线程不保证。
     */
    fun burnSubtitles(
        context: Context,
        videoUri: Uri,
        cues: List<SubtitleCue>,
        outFile: File,
        onProgress: ((Float) -> Unit)? = null
    ) {
        val sortedCues = cues.filter { it.text.isNotBlank() }.sortedBy { it.startMs }
        // 用独立的 GL 线程跑整个循环（EGL 上下文线程亲和）
        val glThread = HandlerThread("subtitle-burn-gl").also { it.start() }
        val handler = Handler(glThread.looper)
        val error = arrayOf<Throwable?>(null)
        val done = java.util.concurrent.CountDownLatch(1)

        handler.post {
            try {
                runBurn(context, videoUri, sortedCues, outFile, onProgress)
            } catch (t: Throwable) {
                Log.e(TAG, "burn failed", t)
                error[0] = t
            } finally {
                done.countDown()
            }
        }
        try {
            done.await()
        } finally {
            glThread.quitSafely()
        }
        error[0]?.let { throw it }
    }

    // ──────────────────────────────────────────────────────────────
    // 主流程（运行在 GL 线程）
    // ──────────────────────────────────────────────────────────────
    private fun runBurn(
        context: Context,
        videoUri: Uri,
        cues: List<SubtitleCue>,
        outFile: File,
        onProgress: ((Float) -> Unit)?
    ) {
        // 两个 extractor：一个读视频轨，一个读音频轨（一个 extractor 同时只能 select 一个轨）
        val (vExt, videoTrack, videoFormat) = openExtractor(context, videoUri, "video/")
            ?: throw IllegalStateException("未找到视频轨道")
        val audio = openExtractor(context, videoUri, "audio/")

        try {
            val srcWidth = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
            val srcHeight = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
            // 不手动处理 rotation-degrees：decoder 配合 SurfaceTexture 时，其 getTransformMatrix
            // 已包含正确朝向，采样后画面即正立。手动再旋转 quad 反而会导致双重旋转/黑边。
            // 限制编码分辨率长边 ≤ MAX_ENCODE_SIDE：高分辨率(4K/1080p)重编码是大头，
            // 降到 1280 长边可成倍降低编码耗时，对字幕可读性几乎无损。仅下采样、不上采样。
            val (outWidth, outHeight) = capToMaxSide(srcWidth, srcHeight, MAX_ENCODE_SIDE)
            val durationUs = runCatching { videoFormat.getLong(MediaFormat.KEY_DURATION) }
                .getOrDefault(0L).coerceAtLeast(1L)

            // 预读全部音频 sample（便于：① 决定是否加音频轨 ② 写入时 PTS 单调）
            val audioSamples = readAudioSamples(audio?.first)

            // EGL：encoder 输入 Surface 需 EGL_RECORDABLE_ANDROID
            val egl = EglCore(withRecordable = true)
            // 1) encoder + 其输入 Surface
            val encoder = createEncoder(outWidth, outHeight, videoFormat)
            val encoderInputSurface = encoder.createInputSurface()
            val encoderSurface = WindowSurface(egl, encoderInputSurface)
            encoderSurface.makeCurrent()

            // 2) OES 纹理 + SurfaceTexture 接收 decoder 输出
            val splitCues = splitLongCues(cues, outWidth, outHeight)
            val renderer = VideoSubtitleRenderer(outWidth, outHeight, splitCues)
            val decoder = createDecoder(videoFormat, renderer.surface)
            encoder.start()
            decoder.start()

            val muxer = MediaMuxer(outFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val info = MediaCodec.BufferInfo()
            var muxerStarted = false
            var videoTrackIdx = -1
            var audioTrackIdx = -1
            var decoderInputDone = false
            var decoderOutputDone = false
            var encoderOutputDone = false
            var signaledEncoderEos = false
            var decodedFrames = 0
            var encodedFrames = 0
            Log.i(TAG, "burn start: ${outWidth}x${outHeight} cues=${cues.size} audio=${audioSamples.size}")

            fun maybeStartMuxer() {
                if (muxerStarted || videoTrackIdx < 0) return
                if (audioSamples.isNotEmpty()) {
                    audioTrackIdx = muxer.addTrack(audio!!.third)
                }
                muxer.start()
                muxerStarted = true
            }

            val drainEnc: () -> Unit = {
                drainEncoder(encoder, muxer, info,
                    videoTrackIdxRef = { videoTrackIdx },
                    onTrack = { idx -> videoTrackIdx = idx; maybeStartMuxer() },
                    onEos = { encoderOutputDone = true },
                    writeToMuxer = { buf, bi, track ->
                        if (muxerStarted) {
                            muxer.writeSampleData(track, buf, bi)
                            encodedFrames++
                        }
                    })
            }

            // 帧循环：喂解码 → 渲染+烧字幕 → 编码 → 写 muxer。decoder EOS 后继续 drain encoder 到其 EOS。
            while (!decoderOutputDone || !encoderOutputDone) {
                // 1) 喂 decoder 输入
                if (!decoderInputDone) {
                    val inIdx = decoder.dequeueInputBuffer(TIMEOUT_US)
                    if (inIdx >= 0) {
                        val buf = decoder.getInputBuffer(inIdx)!!
                        val sz = vExt.readSampleData(buf, 0)
                        if (sz < 0) {
                            decoder.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            decoderInputDone = true
                        } else {
                            decoder.queueInputBuffer(inIdx, 0, sz, vExt.sampleTime, 0)
                            vExt.advance()
                        }
                    }
                }
                // 2) 取 decoder 输出
                if (!decoderOutputDone) {
                    val outIdx = decoder.dequeueOutputBuffer(info, TIMEOUT_US)
                    when {
                        outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED ->
                            Log.d(TAG, "decoder format: ${decoder.outputFormat}")
                        outIdx >= 0 -> {
                            val eos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            if (eos) {
                                decoder.releaseOutputBuffer(outIdx, false)
                                decoderOutputDone = true
                                if (!signaledEncoderEos) {
                                    encoder.signalEndOfInputStream(); signaledEncoderEos = true
                                }
                            } else if (info.size > 0) {
                                // 渲染到 SurfaceTexture，再画「视频+字幕」到 encoder 输入 Surface
                                decoder.releaseOutputBuffer(outIdx, true)
                                renderer.awaitNewFrame()
                                renderer.drawFrame(info.presentationTimeUs)
                                encoderSurface.setPresentationTime(info.presentationTimeUs * 1000L)
                                encoderSurface.swapBuffers()
                                decodedFrames++
                                onProgress?.invoke(
                                    (info.presentationTimeUs.toFloat() / durationUs).coerceIn(0f, 1f))
                            } else {
                                decoder.releaseOutputBuffer(outIdx, false)
                            }
                        }
                    }
                }
                // 3) drain encoder 输出（含首次 OUTPUT_FORMAT_CHANGED → 建 muxer 轨道）
                drainEnc()
            }

            // 写音频（视频轨已写完，音频 PTS 单调即可）
            if (audioSamples.isNotEmpty() && muxerStarted && audioTrackIdx >= 0) {
                audioSamples.forEach { (data, ptsUs, flags) ->
                    val bi = MediaCodec.BufferInfo().apply {
                        offset = 0
                        size = data.limit()
                        presentationTimeUs = ptsUs
                        this.flags = flags
                    }
                    muxer.writeSampleData(audioTrackIdx, data, bi)
                }
            }

            runCatching { muxer.stop() }
            runCatching { muxer.release() }
            runCatching { decoder.stop() }; runCatching { decoder.release() }
            runCatching { encoder.stop() }; runCatching { encoder.release() }
            renderer.release()
            encoderSurface.release()
            egl.release()
            Log.i(TAG, "burn done: decoded=$decodedFrames encoded=$encodedFrames muxerStarted=$muxerStarted " +
                "outSize=${outFile.length()} bytes")
            if (encodedFrames == 0) {
                throw IllegalStateException("编码输出 0 帧，可能编码器配置失败（分辨率/格式不支持）")
            }
            if (outFile.length() < 10_000) {
                throw IllegalStateException("输出文件过小(${outFile.length()}B)，烧录可能失败")
            }
            onProgress?.invoke(1f)
        } finally {
            runCatching { vExt.release() }
            audio?.let { runCatching { it.first.release() } }
        }
    }

    /**
     * 把单行放不下的过长 cue 按 fontSize 宽度拆成多段连续 cue，每段单行可显示；
     * 时间按字数比例分配。短 cue 原样保留。供 [runBurn] 在烧录前预处理。
     */
    private fun splitLongCues(cues: List<SubtitleCue>, width: Int, height: Int): List<SubtitleCue> {
        val fontSize = (height * SUB_FONT_SIZE_RATIO).coerceAtLeast(14f)
        val maxW = width * SUB_MAX_WIDTH_RATIO
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = fontSize }
        val out = ArrayList<SubtitleCue>(cues.size)
        for (cue in cues) {
            val text = cue.text.trim()
            if (paint.measureText(text) <= maxW) {
                out.add(cue)
                continue
            }
            val segments = wrapToWidth(text, maxW, paint)
            val totalChars = segments.sumOf { it.length }.coerceAtLeast(1)
            var t = cue.startMs
            for (seg in segments) {
                val segDur = (cue.endMs - cue.startMs) * seg.length / totalChars
                out.add(SubtitleCue(t, t + segDur, seg))
                t += segDur
            }
        }
        return out
    }

    /** 贪心逐字累加，超 [maxW] 即断段（中文友好；英文可能在词内断，可接受）。 */
    private fun wrapToWidth(text: String, maxW: Float, paint: Paint): List<String> {
        val out = ArrayList<String>()
        val sb = StringBuilder()
        for (ch in text) {
            sb.append(ch)
            if (sb.length > 1 && paint.measureText(sb.toString()) > maxW) {
                sb.deleteCharAt(sb.length - 1)   // 回退末字符到下一段
                out.add(sb.toString().trim())
                sb.clear()
                sb.append(ch)
            }
        }
        if (sb.isNotEmpty()) out.add(sb.toString().trim())
        return out
    }

    // ──────────────────────────────────────────────────────────────
    // 编码器输出抽取：处理格式变更 / 写 muxer / EOS
    // ──────────────────────────────────────────────────────────────
    private fun drainEncoder(
        encoder: MediaCodec,
        muxer: MediaMuxer,
        info: MediaCodec.BufferInfo,
        videoTrackIdxRef: () -> Int,
        onTrack: (Int) -> Unit,
        onEos: () -> Unit,
        writeToMuxer: (ByteBuffer, MediaCodec.BufferInfo, Int) -> Unit
    ) {
        while (true) {
            val idx = encoder.dequeueOutputBuffer(info, TIMEOUT_US)
            when {
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val track = muxer.addTrack(encoder.outputFormat)
                    onTrack(track)
                }
                idx >= 0 -> {
                    val encoded = encoder.getOutputBuffer(idx)!!
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                        info.size = 0
                    }
                    if (info.size > 0) {
                        encoded.position(info.offset)
                        encoded.limit(info.offset + info.size)
                        val track = videoTrackIdxRef()
                        if (track >= 0) writeToMuxer(encoded, info, track)
                    }
                    encoder.releaseOutputBuffer(idx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        onEos()
                        return
                    }
                }
                idx == MediaCodec.INFO_TRY_AGAIN_LATER -> return
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // 编/解码器创建
    // ──────────────────────────────────────────────────────────────
    /** 把 (w,h) 等比缩放到长边 ≤ [maxSide]，且保证两维为偶数（H.264 要求）。仅下采样。 */
    private fun capToMaxSide(w: Int, h: Int, maxSide: Int): Pair<Int, Int> {
        val long = maxOf(w, h)
        if (long <= maxSide) return even(w) to even(h)
        val s = maxSide.toFloat() / long
        return even((w * s).toInt()) to even((h * s).toInt())
    }

    private fun even(v: Int): Int = if (v % 2 == 0) v else v + 1

    private fun createEncoder(width: Int, height: Int, srcFormat: MediaFormat): MediaCodec {
        val format = MediaFormat.createVideoFormat("video/avc", width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            val fps = runCatching { srcFormat.getInteger(MediaFormat.KEY_FRAME_RATE) }
                .getOrDefault(30).coerceIn(1, 60)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            // 码率：0.12 bits/pixel 对烧字幕的 720p 足够清晰且编码更快
            val bitrate = (width * height * fps * 0.12f).toInt().coerceAtLeast(800_000)
            setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
            // 提示编码器尽快产出（低延迟），减少 swapBuffers 反压等待
            if (android.os.Build.VERSION.SDK_INT >= 26) {
                runCatching { setInteger(MediaFormat.KEY_LATENCY, 0) }
            }
            runCatching { setInteger("operating-rate", Short.MAX_VALUE.toInt()) }
        }
        val enc = MediaCodec.createEncoderByType("video/avc")
        enc.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        return enc
    }

    private fun createDecoder(format: MediaFormat, surface: Surface): MediaCodec {
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val dec = MediaCodec.createDecoderByType(mime)
        dec.configure(format, surface, null, 0)
        return dec
    }

    // ──────────────────────────────────────────────────────────────
    // Extractor 工具
    // ──────────────────────────────────────────────────────────────
    /** 返回 (extractor, trackIndex, trackFormat)；找不到指定类型轨道则 null。 */
    private fun openExtractor(
        context: Context, uri: Uri, mimePrefix: String
    ): Triple<MediaExtractor, Int, MediaFormat>? {
        val ext = MediaExtractor()
        context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { afd ->
            ext.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
        } ?: ext.setDataSource(context, uri, null)
        val track = (0 until ext.trackCount).firstOrNull { i ->
            ext.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith(mimePrefix) == true
        } ?: run { ext.release(); return null }
        ext.selectTrack(track)
        return Triple(ext, track, ext.getTrackFormat(track))
    }

    /** 读出音频轨全部编码 sample（data, ptsUs, flags），用于后续整段透传。 */
    private fun readAudioSamples(audio: MediaExtractor?): List<Triple<ByteBuffer, Long, Int>> {
        if (audio == null) return emptyList()
        val out = ArrayList<Triple<ByteBuffer, Long, Int>>()
        val buf = ByteBuffer.allocate(2 * 1024 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        while (true) {
            buf.clear()
            val sz = audio.readSampleData(buf, 0)
            if (sz < 0) break
            val data = ByteArray(sz)
            buf.position(0)
            buf.limit(sz)
            buf.get(data)
            out.add(Triple(ByteBuffer.wrap(data), audio.sampleTime, audio.sampleFlags))
            if (!audio.advance()) break
        }
        return out
    }

    // ══════════════════════════════════════════════════════════════
    // GL 渲染：OES 视频纹理 + 字幕叠加
    // ══════════════════════════════════════════════════════════════
    private class VideoSubtitleRenderer(
        private val width: Int,
        private val height: Int,
        private val cues: List<SubtitleCue>
    ) : SurfaceTexture.OnFrameAvailableListener {

        private val texMatrix = FloatArray(16)

        // OES 视频纹理
        private val videoTexId = genTextures(1)[0]
        // 帧回调放在独立线程（frameThread）上派发，与本渲染循环所在线程（GL 线程）分离。
        // 这样 awaitNewFrame 在 GL 线程上用 wait/notify 阻塞等待时，frameThread 仍能派发
        // onFrameAvailable 并 notify，不会死锁；且比 Thread.sleep 轮询省 CPU、响应更快。
        private val frameThread = android.os.HandlerThread("st-frame").also { it.start() }
        private val frameHandler = android.os.Handler(frameThread.looper)
        val surfaceTexture = SurfaceTexture(videoTexId).apply {
            setOnFrameAvailableListener(this@VideoSubtitleRenderer, frameHandler)
        }
        val surface = Surface(surfaceTexture)

        private val frameLock = Object()
        @Volatile private var frameAvailable = false

        override fun onFrameAvailable(st: SurfaceTexture?) {
            synchronized(frameLock) {
                frameAvailable = true
                (frameLock as Object).notifyAll()
            }
        }

        /** 阻塞等待 SurfaceTexture 的新帧就绪（解码 releaseOutputBuffer(true) 后回调置位）。 */
        fun awaitNewFrame() {
            synchronized(frameLock) {
                val deadline = System.nanoTime() + FRAME_WAIT_NS
                while (!frameAvailable) {
                    val remain = deadline - System.nanoTime()
                    if (remain <= 0) {
                        Log.w(TAG, "awaitNewFrame timeout, continue anyway")
                        break
                    }
                    (frameLock as Object).wait(remain / 1_000_000L)
                }
                frameAvailable = false
            }
            surfaceTexture.updateTexImage()
            surfaceTexture.getTransformMatrix(texMatrix)
        }

        // 着色器
        private val videoProgram = createProgram(VIDEO_VS, VIDEO_FS)
        private val aPosV = GLES20.glGetAttribLocation(videoProgram, "aPosition")
        private val aTcV = GLES20.glGetAttribLocation(videoProgram, "aTexCoord")
        private val uTexMtxV = GLES20.glGetUniformLocation(videoProgram, "uTexMatrix")

        private val subProgram = createProgram(SUB_VS, SUB_FS)
        private val aPosS = GLES20.glGetAttribLocation(subProgram, "aPosition")
        private val aTcS = GLES20.glGetAttribLocation(subProgram, "aTexCoord")

        // 字幕纹理（按当前 cue 文本缓存，变化时重渲染）
        private var subTexId = genTextures(1)[0]
        private var subW = 0
        private var subH = 0
        private var lastText: String? = null

        // 顶点缓冲（视频用，OES 纹理由 texMatrix 处理朝向）
        private val quadPos = floatBufferOf(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
        private val quadTc = floatBufferOf(floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f, 1f, 1f))
        // 字幕用：翻转 V。GLUtils.texImage2D 上传 Bitmap 时 Bitmap 顶行(y=0,文字所在)→ texel v=0，
        // 故让「顶部顶点」取 v=0 才正立。顶点顺序仍是 (左下,右下,左上,右上)。
        private val quadTcFlipV = floatBufferOf(floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f))

        fun drawFrame(ptsUs: Long) {
            GLES20.glViewport(0, 0, width, height)
            GLES20.glClearColor(0f, 0f, 0f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            // 1) 视频：全屏 quad，用 SurfaceTexture 的 texMatrix 采样 OES（已含正确朝向）
            GLES20.glUseProgram(videoProgram)
            GLES20.glUniformMatrix4fv(uTexMtxV, 1, false, texMatrix, 0)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTexId)
            GLES20.glUniform1i(GLES20.glGetUniformLocation(videoProgram, "uTexture"), 0)
            bindQuad(aPosV, aTcV)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

            // 2) 字幕：按当前时间选 cue，文本变化才重绘 Bitmap 上传
            val ms = ptsUs / 1000L
            val cue = cues.firstOrNull { ms in it.startMs..it.endMs }
            val text = cue?.text
            if (!text.isNullOrBlank()) {
                if (text != lastText) {
                    uploadSubtitle(text)
                    lastText = text
                }
                drawSubtitle()
            }
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
        }

        private fun drawSubtitle() {
            if (subW == 0 || subH == 0) return
            GLES20.glUseProgram(subProgram)
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, subTexId)
            GLES20.glUniform1i(GLES20.glGetUniformLocation(subProgram, "uTexture"), 0)
            // 字幕条贴在画面底部，居中。NDC 坐标由像素尺寸换算。
            // bottomMargin 为距画面底边的留白（仅避开系统手势区/圆角）。
            val stripPixH = (height * SUB_STRIP_H_RATIO).toInt().coerceAtLeast(1)
            val bottomMargin = height * SUB_BOTTOM_MARGIN_RATIO
            val scaleY = stripPixH.toFloat() / height
            val marginBottomNdc = bottomMargin / height * 2f
            val bottom = -1f + marginBottomNdc   // 底部 y（上移 = 留底边）
            val top = bottom + 2f * scaleY       // 顶部 y
            val posX = floatBufferOf(floatArrayOf(-1f, bottom, 1f, bottom, -1f, top, 1f, top))
            GLES20.glEnableVertexAttribArray(aPosS)
            GLES20.glVertexAttribPointer(aPosS, 2, GLES20.GL_FLOAT, false, 0, posX)
            GLES20.glEnableVertexAttribArray(aTcS)
            GLES20.glVertexAttribPointer(aTcS, 2, GLES20.GL_FLOAT, false, 0, quadTcFlipV)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glDisable(GLES20.GL_BLEND)
        }

        /** 把 [text] 渲染成带半透明底条+白字描边的单行 Bitmap，上传到 subTexId。 */
        private fun uploadSubtitle(text: String) {
            val w = width.coerceAtMost(1920)
            val h = (height * SUB_STRIP_H_RATIO).toInt().coerceAtLeast(1)
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bmp)
            // 半透明黑色底条，提升字幕在浅色画面上的可读性
            val bg = Paint().apply { color = Color.argb(150, 0, 0, 0) }
            canvas.drawRect(0f, 0f, w.toFloat(), h.toFloat(), bg)
            val fontSize = (height * SUB_FONT_SIZE_RATIO).coerceAtLeast(14f)
            val fill = android.text.TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.WHITE
                textSize = fontSize
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
            val stroke = android.text.TextPaint(fill).apply {
                color = Color.BLACK
                style = Paint.Style.STROKE
                strokeWidth = fontSize * 0.16f
            }
            // 单行居中（长文本已由 splitLongCues 按宽度拆成多段 cue，此处不换行）
            val textW = fill.measureText(text)
            val x = (w - textW) / 2f
            val fm = fill.fontMetrics
            val baseline = (h - (fm.descent - fm.ascent)) / 2f - fm.ascent
            canvas.drawText(text, x, baseline, stroke)
            canvas.drawText(text, x, baseline, fill)

            subW = w; subH = h
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, subTexId)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            android.opengl.GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
            bmp.recycle()
        }

        private fun bindQuad(aPos: Int, aTc: Int) {
            GLES20.glEnableVertexAttribArray(aPos)
            GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, quadPos)
            GLES20.glEnableVertexAttribArray(aTc)
            GLES20.glVertexAttribPointer(aTc, 2, GLES20.GL_FLOAT, false, 0, quadTc)
        }

        fun release() {
            runCatching { surface.release() }
            runCatching { surfaceTexture.release() }
            GLES20.glDeleteTextures(1, intArrayOf(videoTexId), 0)
            GLES20.glDeleteTextures(1, intArrayOf(subTexId), 0)
            GLES20.glDeleteProgram(videoProgram)
            GLES20.glDeleteProgram(subProgram)
            frameThread.quitSafely()
        }

        private fun genTextures(n: Int): IntArray {
            val out = IntArray(n)
            GLES20.glGenTextures(n, out, 0)
            return out
        }
    }

    // ══════════════════════════════════════════════════════════════
    // EGL 极简封装
    // ══════════════════════════════════════════════════════════════
    private class EglCore(withRecordable: Boolean) {
        val display: android.opengl.EGLDisplay
        val config: android.opengl.EGLConfig
        private val context: android.opengl.EGLContext

        init {
            display = android.opengl.EGL14.eglGetDisplay(android.opengl.EGL14.EGL_DEFAULT_DISPLAY)
            check(display !== android.opengl.EGL14.EGL_NO_DISPLAY)
            val version = IntArray(2)
            android.opengl.EGL14.eglInitialize(display, version, 0, version, 1)
            val attr = arrayListOf(
                android.opengl.EGL14.EGL_RED_SIZE, 8,
                android.opengl.EGL14.EGL_GREEN_SIZE, 8,
                android.opengl.EGL14.EGL_BLUE_SIZE, 8,
                android.opengl.EGL14.EGL_ALPHA_SIZE, 8,
                android.opengl.EGL14.EGL_DEPTH_SIZE, 0,
                android.opengl.EGL14.EGL_STENCIL_SIZE, 0,
                android.opengl.EGL14.EGL_RENDERABLE_TYPE, android.opengl.EGL14.EGL_OPENGL_ES2_BIT
            )
            if (withRecordable) attr.addAll(listOf(EGL_RECORDABLE_ANDROID, 1))
            attr.add(android.opengl.EGL14.EGL_NONE)
            val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
            val num = IntArray(1)
            check(android.opengl.EGL14.eglChooseConfig(display, attr.toIntArray(), 0, configs, 0, 1, num, 0))
            config = configs[0]!!
            val ctxAttr = intArrayOf(
                android.opengl.EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                android.opengl.EGL14.EGL_NONE
            )
            context = android.opengl.EGL14.eglCreateContext(display, config, android.opengl.EGL14.EGL_NO_CONTEXT, ctxAttr, 0)
        }

        fun makeCurrent(eglSurface: android.opengl.EGLSurface): Boolean =
            android.opengl.EGL14.eglMakeCurrent(display, eglSurface, eglSurface, context)

        fun presentationTime(surface: android.opengl.EGLSurface, nsecs: Long) {
            android.opengl.EGLExt.eglPresentationTimeANDROID(display, surface, nsecs)
        }

        fun swap(surface: android.opengl.EGLSurface): Boolean =
            android.opengl.EGL14.eglSwapBuffers(display, surface)

        fun release() {
            android.opengl.EGL14.eglMakeCurrent(display,
                android.opengl.EGL14.EGL_NO_SURFACE, android.opengl.EGL14.EGL_NO_SURFACE,
                android.opengl.EGL14.EGL_NO_CONTEXT)
            android.opengl.EGL14.eglDestroyContext(display, context)
            android.opengl.EGL14.eglTerminate(display)
        }

        companion object {
            private const val EGL_RECORDABLE_ANDROID = 0x3142
        }
    }

    private class WindowSurface(private val egl: EglCore, surface: Surface) {
        private val eglSurface: android.opengl.EGLSurface

        init {
            val attr = intArrayOf(android.opengl.EGL14.EGL_NONE)
            eglSurface = android.opengl.EGL14.eglCreateWindowSurface(
                egl.display, egl.config, surface, attr, 0
            )
            check(eglSurface !== android.opengl.EGL14.EGL_NO_SURFACE)
        }

        fun makeCurrent() = egl.makeCurrent(eglSurface)
        fun setPresentationTime(nsecs: Long) = egl.presentationTime(eglSurface, nsecs)
        fun swapBuffers() = egl.swap(eglSurface)
        fun release() {
            android.opengl.EGL14.eglDestroySurface(egl.display, eglSurface)
        }
    }

    // ──────────────────────────────────────────────────────────────
    // GLSL / 小工具
    // ──────────────────────────────────────────────────────────────
    private const val VIDEO_VS = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat4 uTexMatrix;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;
        }
    """

    private const val VIDEO_FS = """
        #extension GL_OES_EGL_image_external : require
        precision mediump float;
        varying vec2 vTexCoord;
        uniform samplerExternalOES uTexture;
        void main() { gl_FragColor = texture2D(uTexture, vTexCoord); }
    """

    private const val SUB_VS = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        varying vec2 vTexCoord;
        void main() { gl_Position = aPosition; vTexCoord = aTexCoord; }
    """

    private const val SUB_FS = """
        precision mediump float;
        varying vec2 vTexCoord;
        uniform sampler2D uTexture;
        void main() { gl_FragColor = texture2D(uTexture, vTexCoord); }
    """

    private fun createProgram(vsSrc: String, fsSrc: String): Int {
        fun shader(type: Int, src: String): Int {
            val s = GLES20.glCreateShader(type)
            GLES20.glShaderSource(s, src); GLES20.glCompileShader(s)
            val status = IntArray(1)
            GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, status, 0)
            check(status[0] != 0) { "shader compile: ${GLES20.glGetShaderInfoLog(s)}" }
            return s
        }
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, shader(GLES20.GL_VERTEX_SHADER, vsSrc))
        GLES20.glAttachShader(p, shader(GLES20.GL_FRAGMENT_SHADER, fsSrc))
        GLES20.glLinkProgram(p)
        val status = IntArray(1)
        GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, status, 0)
        check(status[0] != 0) { "program link: ${GLES20.glGetProgramInfoLog(p)}" }
        return p
    }

    private fun floatBufferOf(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(a).also { it.position(0) }
}
