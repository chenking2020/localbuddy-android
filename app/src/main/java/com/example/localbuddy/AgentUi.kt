package com.example.localbuddy

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Language
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonColors
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// 智能体主题色（与首页 FunctionCard accent 一致），二级页面据此区分。
// 统一用 Tailwind 600 级：各色相分离清晰、白字/彩底对比足够。
internal val AgentPurple = Color(0xFF4F46E5)   // 扫描智能体 · 靛蓝
internal val AgentOrange = Color(0xFFEA580C)    // 录音翻译智能体 · 橙
internal val AgentCyan = Color(0xFF0284C7)      // 视频字幕智能体 · 天蓝
internal val AgentGreen = Color(0xFF059669)     // 实时听音智能体 · 翠绿
internal val AgentBlue = Color(0xFF2563EB)      // 文本对话智能体 · 宝蓝
internal val AgentRose = Color(0xFFE11D48)      // 语音通话智能体 · 玫红

/**
 * 二级页面统一的渐变 Hero 头部：主题色渐变背景 + 返回按钮 + 图标 + 标题 + 副标题。
 * 渐变在 windowInsetsPadding 之前应用，故会延伸到状态栏下方；内容内缩避开状态栏。
 */
@Composable
fun AgentHeader(
    accent: Color,
    icon: ImageVector,
    title: String,
    subtitle: String,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    val gradient = Brush.linearGradient(
        listOf(accent, lerp(accent, Color.White, 0.22f))
    )
    Box(
        modifier = modifier
            .fillMaxWidth()
            .background(gradient)
            .windowInsetsPadding(WindowInsets.statusBars)
            .padding(start = 12.dp, end = 20.dp, top = 10.dp, bottom = 18.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // 返回按钮：半透明白底 + 白色箭头
                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color.White.copy(alpha = 0.2f))
                        .clickable(onClick = onBack),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "返回",
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }
                Spacer(modifier = Modifier.weight(1f))
                // 右侧图标块
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(Color.White.copy(alpha = 0.22f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
            }
            Column(
                modifier = Modifier.padding(start = 4.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = 0.88f)
                )
            }
        }
    }
}

/**
 * 卡片内带图标点缀的标题行（"配置"/"实时结果"等），统一三页卡片标题样式。
 * trailing 放右侧附属内容（如"进行中"指示、操作按钮）。
 */
@Composable
fun SectionTitle(
    icon: ImageVector,
    text: String,
    accent: Color,
    modifier: Modifier = Modifier,
    trailing: @Composable (() -> Unit)? = null
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Box(
            modifier = Modifier
                .size(26.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(accent.copy(alpha = 0.15f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = accent,
                modifier = Modifier.size(16.dp)
            )
        }
        Text(
            text = text,
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.weight(1f)
        )
        trailing?.invoke()
    }
}

/** 主操作按钮的主题色配色（含 disabled 态），让按钮与头部 accent 呼应。 */
@Composable
fun agentButtonColors(accent: Color): ButtonColors = ButtonDefaults.buttonColors(
    containerColor = accent,
    contentColor = Color.White,
    disabledContainerColor = accent.copy(alpha = 0.38f),
    disabledContentColor = Color.White.copy(alpha = 0.7f)
)

/**
 * 语音源语种选择弹窗：第一项「自动识别」（ASR 自动检测语种），其后为语音识别支持的
 * 30 种语种 + 22 种中国方言（选定后强制该语种转写，短语音不再依赖语种识别）。
 * 多个 ASR 智能体共用（录音翻译 / 本地视频字幕 / 实时听音字幕）。
 */
@Composable
fun SrcLangPickerDialog(
    accent: Color,
    selected: String,
    onPick: (String) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("选择语音语种") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(
                            if (selected == LANG_AUTO) accent.copy(alpha = 0.12f) else Color.Transparent
                        )
                        .clickable { onPick(LANG_AUTO) }
                        .padding(horizontal = 8.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Filled.AutoAwesome, contentDescription = null,
                        modifier = Modifier.size(18.dp),
                        tint = if (selected == LANG_AUTO) accent else MaterialTheme.colorScheme.outline
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        "自动识别", Modifier.weight(1f),
                        fontWeight = if (selected == LANG_AUTO) FontWeight.SemiBold else FontWeight.Normal
                    )
                    if (selected == LANG_AUTO) {
                        Icon(Icons.Filled.Check, contentDescription = null,
                            tint = accent, modifier = Modifier.size(16.dp))
                    }
                }
                PickerSectionTitle("语种", accent)
                asrLangs.forEach { lang ->
                    LangRow(accent, lang, lang.code == selected, onPick)
                }
                PickerSectionTitle("中国方言", accent)
                asrDialects.forEach { lang ->
                    LangRow(accent, lang, lang.code == selected, onPick)
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("关闭") } }
    )
}

/** 语种选择弹窗里的分组小标题。 */
@Composable
private fun PickerSectionTitle(text: String, accent: Color) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelLarge,
        fontWeight = FontWeight.SemiBold,
        color = accent,
        modifier = Modifier.padding(top = 12.dp, bottom = 2.dp, start = 4.dp)
    )
}

/** 语种/方言单选行。 */
@Composable
private fun LangRow(
    accent: Color,
    lang: Language,
    selected: Boolean,
    onPick: (String) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (selected) accent.copy(alpha = 0.12f) else Color.Transparent)
            .clickable { onPick(lang.code) }
            .padding(horizontal = 8.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(lang.flag, fontSize = 18.sp)
        Spacer(Modifier.width(10.dp))
        Text(
            lang.name, Modifier.weight(1f),
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
        if (selected) {
            Icon(Icons.Filled.Check, contentDescription = null,
                tint = accent, modifier = Modifier.size(16.dp))
        }
    }
}

