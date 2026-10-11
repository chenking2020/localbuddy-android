package com.example.localbuddy

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * 主题模式偏好：浅色 / 深色 / 跟随系统。
 *
 * 用 SharedPreferences 持久化（单值，无需引入 DataStore 依赖），暴露 [StateFlow] 供 Compose
 * 顶层收集。在 [MainActivity.onCreate] 调用 [init] 加载；[SettingsScreen] 调用 [set] 切换，
 * StateFlow 即刻推送 -> 整棵 Compose 树重组成新主题。
 *
 * 注意：启动闪屏（XML 主题 Theme.Localbuddy）只能跟随系统夜间模式（values-night），
 * 无法读取本偏好；Compose 内容加载后立即按此偏好生效，差异仅短暂闪屏。
 */
enum class ThemeMode(val label: String, val desc: String) {
    SYSTEM("跟随系统", "随系统深浅色自动切换"),
    LIGHT("浅色", "始终浅色"),
    DARK("深色", "始终深色")
}

object ThemePref {

    private const val FILE = "ui_prefs"
    private const val KEY = "theme_mode"

    @Volatile private var prefs: SharedPreferences? = null
    private val _mode = MutableStateFlow(ThemeMode.SYSTEM)
    val mode: StateFlow<ThemeMode> = _mode.asStateFlow()

    /** 加载已保存的偏好。多次调用幂等。应在 MainActivity.onCreate 早期调用一次。 */
    fun init(context: Context) {
        if (prefs != null) return
        synchronized(this) {
            if (prefs != null) return
            val p = context.applicationContext.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            prefs = p
            _mode.value = ThemeMode.entries.getOrElse(p.getInt(KEY, 0)) { ThemeMode.SYSTEM }
        }
    }

    /** 切换主题模式并持久化。 */
    fun set(context: Context, mode: ThemeMode) {
        init(context)
        prefs!!.edit().putInt(KEY, mode.ordinal).apply()
        _mode.value = mode
    }
}
