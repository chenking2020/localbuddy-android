package com.example.localbuddy

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.DocumentScanner
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Subtitles
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.localbuddy.ui.theme.LocalbuddyTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // 保持屏幕常亮：APP 在前台时不熄屏，避免长视频字幕生成 / 录音翻译中途息屏中断
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        ThemePref.init(this)          // 加载主题偏好（深色/浅色/跟随系统）
        NnapiPref.init(this)          // 加载 NNAPI 开关偏好（NPU 加速）
        setContent {
            val mode by ThemePref.mode.collectAsState()
            val darkTheme = when (mode) {
                ThemeMode.SYSTEM -> isSystemInDarkTheme()
                ThemeMode.LIGHT -> false
                ThemeMode.DARK -> true
            }
            // edge-to-edge 下系统栏透明，需按实际深浅设置图标外观（含强制模式），避免图标与背景反差不足
            val view = LocalView.current
            if (!view.isInEditMode) {
                SideEffect {
                    val window = (view.context as ComponentActivity).window
                    WindowCompat.getInsetsController(window, view).apply {
                        isAppearanceLightStatusBars = !darkTheme
                        isAppearanceLightNavigationBars = !darkTheme
                    }
                }
            }
            LocalbuddyTheme(darkTheme = darkTheme) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainApp()
                }
            }
        }
    }
}

// ── 导航路由 ──
// 文本对话 / 语音通话 已融合进首页对话框，不再单独成页：
// 它们各自只会有首页这一个 ViewModel 宿主，避免出现第二份模型实例。
sealed class Screen(val route: String) {
    object Main : Screen("main")
    object ScanAgent : Screen("scanAgent")
    object DocTrans : Screen("docTrans")
    object SpeechMT : Screen("speechMT")
    object CrossTalk : Screen("crossTalk")
    object VideoSubtitle : Screen("videoSubtitle")
    object ListenSubtitle : Screen("listenSubtitle")
    object Settings : Screen("settings")
}

private val Orange = Color(0xFFEA580C)       // 录音翻译 · 橙
private val Cyan = Color(0xFF0284C7)         // 视频字幕 · 天蓝
private val Green = Color(0xFF059669)        // 实时听音 · 翠绿

// ── 功能列表 ──
// 首页对话框顶部展示的六个智能体（顺序即宫格顺序，三列两行）。
// isComingSoon = true 时卡片禁用点击（路由与页面代码保留，后续下线某个智能体时加该标记即可）。
private val functionItems = listOf(
    FunctionItem(
        id = 2,
        title = "录音翻译智能体",
        description = "会议、对话场景下实时录音并转写翻译，全程离线，隐私无忧。",
        icon = Icons.Default.Mic,
        accent = Orange,
        route = Screen.SpeechMT.route,
        tag = "实时 · 语音 · 翻译"
    ),
    FunctionItem(
        id = 4,
        title = "本地视频字幕智能体",
        description = "为本地视频自动生成字幕和总结，沉浸式字幕体验，快速视频内容总结查看。",
        icon = Icons.Default.Subtitles,
        accent = Cyan,
        route = Screen.VideoSubtitle.route,
        tag = "视频 · 字幕 · 离线"
    ),
    FunctionItem(
        id = 3,
        title = "跨语沟通智能体",
        description = "多人多语实时沟通：按说话人转写并译成对方语言，双色分区展示，全程离线。",
        icon = Icons.Default.RecordVoiceOver,
        accent = Color(0xFF0D9488),
        route = Screen.CrossTalk.route,
        tag = "多人 · 多语 · 实时"
    ),
    FunctionItem(
        id = 5,
        title = "实时视频听音智能体",
        description = "实时听音悬浮字幕，支持 50 种语言。Beta 版，当前效率较低，正在升级中。",
        icon = Icons.Default.Subtitles,
        accent = Green,
        route = Screen.ListenSubtitle.route,
        tag = "视频 · 实时 · 悬浮",
        isBeta = true
    ),
    FunctionItem(
        id = 0,
        title = "全能扫描智能体",
        description = "拍照扫描各类格式的扫描件",
        icon = Icons.Default.DocumentScanner,
        accent = Color(0xFF4F46E5),
        route = Screen.ScanAgent.route,
        tag = "全能 · 扫描 · 高清"
    ),
    FunctionItem(
        id = 1,
        title = "文档识别翻译智能体",
        description = "离线识别各类文档（txt/word/pdf/图片）并翻译，支持格式还原导出。",
        icon = Icons.Default.Description,
        accent = Color(0xFF7C3AED),
        route = Screen.DocTrans.route,
        tag = "文档 · OCR · 翻译"
    )
)

// ── 主应用组件 ──
@Composable
fun MainApp() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = Screen.Main.route
    ) {
        composable(Screen.Main.route) {
            HomeChatScreen(
                agents = functionItems,
                onAgentClick = { route -> navController.navigate(route) },
                onSettingsClick = { navController.navigate(Screen.Settings.route) }
            )
        }
        composable(Screen.ScanAgent.route) {
            ScanAgentScreen(onBack = { navController.popBackStack() })
        }
        composable(Screen.DocTrans.route) {
            DocTransScreen(onBack = { navController.popBackStack() })
        }
        composable(Screen.SpeechMT.route) {
            SpeechMTScreen(onBack = { navController.popBackStack() })
        }
        composable(Screen.CrossTalk.route) {
            CrossLangScreen(onBack = { navController.popBackStack() })
        }
        composable(Screen.VideoSubtitle.route) {
            VideoSubtitleScreen(onBack = { navController.popBackStack() })
        }
        composable(Screen.ListenSubtitle.route) {
            ListenSubtitleScreen(onBack = { navController.popBackStack() })
        }
        composable(Screen.Settings.route) {
            SettingsScreen(onBack = { navController.popBackStack() })
        }
    }
}
