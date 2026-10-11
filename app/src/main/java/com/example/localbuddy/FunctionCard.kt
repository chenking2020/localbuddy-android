package com.example.localbuddy

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

// Beta 标记色（琥珀，醒目区别于功能 accent 色）
private val BetaColor = Color(0xFFF59E0B)

// 数据类，定义功能卡片的信息
data class FunctionItem(
    val id: Int,
    val title: String,
    val description: String,
    val icon: ImageVector,
    val accent: Color,
    val route: String,
    val tag: String = "智能体",
    val isComingSoon: Boolean = false,
    val isBeta: Boolean = false
)

/**
 * 功能卡片 — 横向布局，左侧大尺寸图标、中间标题+分类标签+描述、右侧操作按钮。
 * 不可用项以低对比度展示并禁用点击。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FunctionCard(
    functionItem: FunctionItem,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val available = !functionItem.isComingSoon
    val containerAlpha = if (available) 1f else 0.6f

    ElevatedCard(
        modifier = modifier.fillMaxWidth(),
        onClick = onClick,
        enabled = available,
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.surface,
            disabledContainerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.elevatedCardElevation(
            defaultElevation = 3.dp,
            disabledElevation = 1.dp
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // 彩色图标卡片
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(functionItem.accent.copy(alpha = 0.15f * containerAlpha)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = functionItem.icon,
                    contentDescription = null,
                    tint = functionItem.accent.copy(alpha = containerAlpha),
                    modifier = Modifier.size(28.dp)
                )
            }

            // 标题 + 标签 + 描述
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    CategoryTag(text = functionItem.tag, color = functionItem.accent)
                    if (functionItem.isBeta) {
                        BetaChip()
                    }
                    if (functionItem.isComingSoon) {
                        ComingSoonChip()
                    }
                }

                Text(
                    text = functionItem.title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = containerAlpha),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Text(
                    text = functionItem.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = androidx.compose.ui.unit.TextUnit.Unspecified
                )
            }

            // 右侧前进按钮
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(
                        if (available) functionItem.accent.copy(alpha = 0.12f)
                        else MaterialTheme.colorScheme.surfaceVariant
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                    contentDescription = null,
                    tint = if (available) functionItem.accent
                    else MaterialTheme.colorScheme.outline,
                    modifier = Modifier.size(18.dp)
                )
            }
        }
    }
}

@Composable
private fun CategoryTag(text: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(color.copy(alpha = 0.12f))
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun ComingSoonChip() {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Text(
            text = "待上线",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun BetaChip() {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(BetaColor.copy(alpha = 0.15f))
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Text(
            text = "Beta",
            style = MaterialTheme.typography.labelSmall,
            color = BetaColor,
            fontWeight = FontWeight.SemiBold
        )
    }
}
