package com.example.localbuddy

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * NNAPI（Android NPU/DSP 硬件加速）开关。
 *
 * 开启后 [SpeechMTEngine] 的 OrtSession 优先注册 NNAPI EP，支持的算子走 NPU 可能大幅加速；
 * 但 GroupQueryAttention 融合算子 / dynamic shape / KV cache 的 NNAPI 支持参差，可能无效、
 * 更慢或精度异常，故默认关、可在设置页切换。切换后需重建 session（[SpeechMTEngine.reload]）。
 *
 * 持久化方式同 [ThemePref]：SharedPreferences + StateFlow，[MainActivity.onCreate] 调 [init]。
 */
object NnapiPref {

    private const val FILE = "ui_prefs"
    private const val KEY = "use_nnapi"

    @Volatile private var prefs: SharedPreferences? = null
    private val _enabled = MutableStateFlow(false)
    val enabled: StateFlow<Boolean> = _enabled.asStateFlow()

    /** 加载已保存的偏好。多次调用幂等。应在 MainActivity.onCreate 早期调用一次。 */
    fun init(context: Context) {
        if (prefs != null) return
        synchronized(this) {
            if (prefs != null) return
            val p = context.applicationContext.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            prefs = p
            _enabled.value = p.getBoolean(KEY, false)   // 默认关：NNAPI 兼容性风险
        }
    }

    /** 切换并持久化。切换后调用方应触发 [SpeechMTEngine.reload] 重建 session。 */
    fun set(context: Context, on: Boolean) {
        init(context)
        prefs!!.edit().putBoolean(KEY, on).apply()
        _enabled.value = on
    }

    fun isEnabled(): Boolean = _enabled.value
}
