package com.example.localbuddy

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.automirrored.filled.ReceiptLong
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.input.pointer.pointerInput
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.localbuddy.scan.DocScanner
import com.example.localbuddy.scan.FileExport
import com.example.localbuddy.scan.ImageComposer
import com.example.localbuddy.scan.PdfExporter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

/**
 * 全能扫描智能体主页面。
 * 流水线：采集（拍照/相册）-> 扫描处理 + 导出（长图/PDF）。（OCR/翻译已移至「文档识别翻译智能体」。）
 * UI 与其他智能体二级页对齐：AgentHeader 渐变头 + ElevatedCard 分组卡片 + SectionTitle 图标标题。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScanAgentScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val vm: ScanAgentViewModel = viewModel(
        factory = ViewModelProvider.AndroidViewModelFactory.getInstance(
            (context.applicationContext as Application)
        )
    )
    val state by vm.state.collectAsState()
    val scope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasCamPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val camPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasCamPermission = it
    }

    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(10)) { uris ->
        // 相册多选：并发解码，逐张追加为扫描页（顺序随多选选中次序）
        scope.launch {
            uris.forEach { uri ->
                val bmp = withContext(Dispatchers.IO) {
                    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                }
                if (bmp != null) vm.addPage(bmp)
            }
        }
    }
    val longImageLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("image/jpeg")) { uri ->
        if (uri != null) scope.launch {
            val images = vm.exportImages()
            if (images.isNotEmpty()) {
                val composed = withContext(Dispatchers.IO) { ImageComposer.composeVertical(images) }
                withContext(Dispatchers.IO) { FileExport.writeBitmap(context, uri, composed) }
            }
        }
    }
    val pdfLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { uri ->
        if (uri != null) scope.launch {
            val images = vm.exportImages()
            if (images.isNotEmpty()) {
                withContext(Dispatchers.IO) {
                    context.contentResolver.openOutputStream(uri)?.use { PdfExporter.export(images, it) }
                }
            }
        }
    }

    val imageCapture = remember { ImageCapture.Builder().build() }
    fun takePhoto() {
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bmp = image.toBitmap()
                    val rot = image.imageInfo.rotationDegrees
                    val out = if (rot != 0) {
                        val m = Matrix().apply { postRotate(rot.toFloat()) }
                        Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                    } else bmp
                    image.close()
                    vm.addPage(out)
                }
                override fun onError(exc: ImageCaptureException) { /* 忽略，用户可重试 */ }
            }
        )
    }

    // 目标语种选择对话框（与录音翻译页一致的国旗 + 中文名选择器）
    // 全屏预览：当前预览页的下标（null = 未在预览），点击缩略图设置、翻页/关闭时更新
    var previewIndex by remember { mutableStateOf<Int?>(null) }
    // 相机（绑定后的 Camera，供手电筒控制）与手电筒状态
    var camera by remember { mutableStateOf<androidx.camera.core.Camera?>(null) }
    var torchOn by remember { mutableStateOf(false) }
    // 导出格式选择
    var showExportDialog by remember { mutableStateOf(false) }
    fun toggleTorch() {
        val c = camera ?: return
        val next = !torchOn
        torchOn = next
        scope.launch { runCatching { c.cameraControl.enableTorch(next) } }
    }

    Scaffold(
        topBar = {
            AgentHeader(
                accent = AgentPurple,
                icon = Icons.Default.DocumentScanner,
                title = "全能扫描智能体",
                subtitle = "扫描 · OCR · 翻译 · 全程离线",
                onBack = onBack
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 12.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // ── 卡片1：采集（扫描模式 + 相机/相册） ──
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    SectionTitle(
                        icon = Icons.Default.CameraAlt,
                        text = "采集",
                        accent = AgentPurple
                    )

                    ConfigRow(label = "扫描模式") {
                        ScanModePicker(
                            current = state.mode,
                            onSelect = { vm.setMode(it) }
                        )
                    }

                    // 色彩：黑白 / 彩色（所有格式通用）
                    ConfigRow(label = "色彩") {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            ScanModePill(
                                text = "黑白",
                                selected = !state.colorMode,
                                onClick = { vm.setColorMode(false) }
                            )
                            ScanModePill(
                                text = "彩色",
                                selected = state.colorMode,
                                onClick = { vm.setColorMode(true) }
                            )
                        }
                    }

                    if (hasCamPermission) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(340.dp)
                                .clip(RoundedCornerShape(16.dp))
                                .background(MaterialTheme.colorScheme.surfaceVariant)
                        ) {
                            CameraPreview(imageCapture, lifecycleOwner) { cam -> camera = cam }
                            // 右上手电筒
                            Box(
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .padding(10.dp)
                                    .size(40.dp)
                                    .clip(CircleShape)
                                    .background(Color.Black.copy(alpha = 0.4f))
                                    .clickable { toggleTorch() },
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(if (torchOn) Icons.Default.FlashOn else Icons.Default.FlashOff,
                                    contentDescription = if (torchOn) "关闭手电" else "打开手电",
                                    tint = Color.White, modifier = Modifier.size(22.dp))
                            }
                            // 右下相册
                            Box(
                                modifier = Modifier
                                    .align(Alignment.BottomEnd)
                                    .padding(14.dp)
                                    .size(46.dp)
                                    .clip(CircleShape)
                                    .background(Color.Black.copy(alpha = 0.4f))
                                    .clickable {
                                        galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(Icons.Default.PhotoLibrary, contentDescription = "从相册选择（可多选）",
                                    tint = Color.White, modifier = Modifier.size(22.dp))
                            }
                            // 中下圆形快门
                            Box(
                                modifier = Modifier
                                    .align(Alignment.BottomCenter)
                                    .padding(bottom = 18.dp)
                                    .size(64.dp)
                                    .clip(CircleShape)
                                    .border(3.dp, Color.White, CircleShape)
                                    .clickable { takePhoto() },
                                contentAlignment = Alignment.Center
                            ) {
                                Box(Modifier.size(48.dp).clip(CircleShape).background(Color.White))
                            }
                        }
                    } else {
                        // 未授权占位：图标 + 说明 + 授权按钮 + 相册（无相机也能从相册扫）
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp)
                                .clip(RoundedCornerShape(16.dp))
                                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Icon(
                                    Icons.Default.CameraAlt,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.outline,
                                    modifier = Modifier.size(36.dp)
                                )
                                Text(
                                    "需要相机权限才能拍照",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    OutlinedButton(
                                        onClick = { galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
                                    ) {
                                        Icon(Icons.Default.AddPhotoAlternate, null, Modifier.size(18.dp))
                                        Spacer(Modifier.width(6.dp)); Text("从相册")
                                    }
                                    Button(
                                        onClick = { camPermLauncher.launch(Manifest.permission.CAMERA) },
                                        colors = agentButtonColors(AgentPurple)
                                    ) { Text("授予相机权限") }
                                }
                            }
                        }
                    }
                }
            }

            // ── 卡片2：页面与导出（缩略图列表 + 扫描全部 + 长图/PDF） ──
            ElevatedCard(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(18.dp),
                elevation = CardDefaults.elevatedCardElevation(defaultElevation = 2.dp)
            ) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    SectionTitle(
                        icon = Icons.Default.PhotoLibrary,
                        text = "扫描件",
                        accent = AgentPurple,
                        trailing = {
                            if (state.pages.isNotEmpty()) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(
                                        "${state.pages.size} 页",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Spacer(Modifier.width(4.dp))
                                    TextButton(onClick = { vm.clear() }) { Text("清空") }
                                }
                            }
                        }
                    )

                    if (state.pages.isEmpty()) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(96.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Icon(
                                    Icons.Default.Image,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.outline,
                                    modifier = Modifier.size(26.dp)
                                )
                                Text(
                                    "拍照或选图，自动生成扫描件；长按可拖拽排序",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    } else {
                        val rowState = rememberLazyListState()
                        var dragState by remember { mutableStateOf<DragState?>(null) }
                        // 定宽步进法：缩略图 108dp + 间距 10dp = 一槽，用于把横向拖拽量换算成页序步进
                        val slotPx = with(LocalDensity.current) { (118.dp).toPx() }
                        LazyRow(
                            state = rowState,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            items(state.pages, key = { it.id }) { page ->
                                val idx = state.pages.indexOfFirst { it.id == page.id }
                                val disp = page.scanned ?: page.original
                                Box(
                                    modifier = Modifier
                                        .size(108.dp, 144.dp)
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(MaterialTheme.colorScheme.surfaceVariant)
                                        .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(10.dp))
                                        .clickable { previewIndex = idx }
                                        // 长按后横向拖拽排序：把累计位移按定宽槽换算成页序步进
                                        .pointerInput(page.id) {
                                            detectDragGesturesAfterLongPress(
                                                onDragStart = { dragState = DragState(idx, page.id, 0f) },
                                                onDrag = { change, amount ->
                                                    change.consume()
                                                    val d = dragState ?: return@detectDragGesturesAfterLongPress
                                                    if (d.id != page.id) return@detectDragGesturesAfterLongPress
                                                    var acc = d.accum + amount.x
                                                    val step = Math.round(acc / slotPx)
                                                    if (step != 0) {
                                                        val target = (d.startIndex + step).coerceIn(0, state.pages.lastIndex)
                                                        vm.movePage(d.startIndex, target)
                                                        // 消化已跨过的步进，避免手指追不上来回抖
                                                        dragState = DragState(target, d.id, acc - step * slotPx)
                                                    } else {
                                                        dragState = d.copy(accum = acc)
                                                    }
                                                },
                                                onDragEnd = { dragState = null },
                                                onDragCancel = { dragState = null }
                                            )
                                        }
                                ) {
                                    Image(
                                        bitmap = disp.asImageBitmap(),
                                        contentDescription = null,
                                        modifier = Modifier.fillMaxSize()
                                    )
                                    // 扫描中 loading / 已扫描标记
                                    if (page.scanning) {
                                        Box(modifier = Modifier.align(Alignment.Center), contentAlignment = Alignment.Center) {
                                            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp, color = AgentPurple)
                                        }
                                        Box(modifier = Modifier.align(Alignment.TopStart).padding(4.dp)) {
                                            TagBadge(text = "扫描中", color = AgentPurple)
                                        }
                                    } else if (page.scanned != null) {
                                        Box(modifier = Modifier.align(Alignment.TopStart).padding(4.dp)) {
                                            TagBadge(text = "已扫描", color = AgentPurple)
                                        }
                                    }
                                    // 拖拽把手提示（左下角半透明圆）
                                    Box(
                                        modifier = Modifier
                                            .align(Alignment.BottomStart)
                                            .padding(4.dp)
                                            .size(20.dp)
                                            .clip(CircleShape)
                                            .background(Color.Black.copy(alpha = 0.4f)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Icon(
                                            Icons.Default.DragHandle,
                                            contentDescription = "长按拖拽排序",
                                            tint = Color.White,
                                            modifier = Modifier.size(13.dp)
                                        )
                                    }
                                    // 删除按钮（右上角半透明圆）
                                    Box(
                                        modifier = Modifier
                                            .align(Alignment.TopEnd)
                                            .padding(4.dp)
                                            .size(22.dp)
                                            .clip(CircleShape)
                                            .background(Color.Black.copy(alpha = 0.45f))
                                            .clickable { vm.removePage(page.id) },
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Icon(
                                            Icons.Default.Close,
                                            contentDescription = "删除",
                                            tint = Color.White,
                                            modifier = Modifier.size(14.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Button(
                            onClick = { showExportDialog = true },
                            enabled = state.pages.any { it.scanned != null },
                            modifier = Modifier.weight(1f),
                            colors = agentButtonColors(AgentPurple)
                        ) {
                            Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("导出")
                        }
                        OutlinedButton(
                            onClick = { vm.scanAll() },
                            enabled = !state.processing && state.pages.isNotEmpty(),
                            modifier = Modifier.weight(1f)
                        ) {
                            if (state.processing) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                Spacer(Modifier.width(8.dp))
                                Text("扫描中…")
                            } else {
                                Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("重扫全部")
                            }
                        }
                    }
                }
            }

            Spacer(Modifier.height(8.dp))
        }
    }

    if (showExportDialog) {
        AlertDialog(
            onDismissRequest = { showExportDialog = false },
            title = { Text("导出格式") },
            text = {
                Column {
                    Row(Modifier.fillMaxWidth().clickable {
                        showExportDialog = false
                        longImageLauncher.launch("scan_${System.currentTimeMillis()}.jpg")
                    }.padding(vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Image, null, Modifier.size(20.dp), tint = AgentPurple)
                        Spacer(Modifier.width(10.dp))
                        Text("长图 (JPG)", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                    }
                    Row(Modifier.fillMaxWidth().clickable {
                        showExportDialog = false
                        pdfLauncher.launch("scan_${System.currentTimeMillis()}.pdf")
                    }.padding(vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.PictureAsPdf, null, Modifier.size(20.dp), tint = AgentPurple)
                        Spacer(Modifier.width(10.dp))
                        Text("PDF", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                    }
                }
            },
            confirmButton = { TextButton(onClick = { showExportDialog = false }) { Text("取消") } }
        )
    }

    val pi = previewIndex
    if (pi != null && pi in state.pages.indices) {
        ImagePreviewDialog(
            pages = state.pages,
            index = pi,
            onIndexChange = { previewIndex = it },
            onDismiss = { previewIndex = null }
        )
    }
}

/** 拖拽排序过程中的临时状态：被拖页的起始/当前下标、id、已累计的横向位移（px）。 */
private data class DragState(val startIndex: Int, val id: String, val accum: Float)

@Composable
private fun CameraPreview(
    imageCapture: ImageCapture,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    onCameraReady: (androidx.camera.core.Camera) -> Unit
) {
    val context = LocalContext.current
    val previewView = remember { PreviewView(context) }
    AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
    LaunchedEffect(Unit) {
        val provider = getCameraProvider(context)
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        provider.unbindAll()
        runCatching {
            val cam = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            onCameraReady(cam)
        }
    }
}

private suspend fun getCameraProvider(context: Context): ProcessCameraProvider =
    suspendCancellableCoroutine { cont ->
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try { cont.resume(future.get()) } catch (e: Exception) { cont.cancel(e) }
        }, ContextCompat.getMainExecutor(context))
    }

// ── 局部 UI 原语：与录音翻译页保持视觉一致 ──

/** 卡片内 label + 内容行（label 固定宽度，内容自适应）。 */
@Composable
private fun ConfigRow(label: String, content: @Composable () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.width(76.dp)
        )
        Box(modifier = Modifier.weight(1f)) { content() }
    }
}

/**
 * 扫描样式选择器（对标竞品「常用 + 更多」）：
 * 顶部平铺常用 3 个 pill（文档/名片/白板），末尾「更多」展开分类网格弹窗。
 * 选中「更多」内样式时，「更多」pill 高亮并显示「更多·{样式名}」。
 */
@Composable
private fun ScanModePicker(
    current: DocScanner.Mode,
    onSelect: (DocScanner.Mode) -> Unit
) {
    var showMore by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        DocScanner.Mode.COMMON.forEach { mode ->
            ScanModePill(
                text = mode.label,
                selected = mode == current,
                onClick = { onSelect(mode) }
            )
        }
        // 「更多」：选中非常用样式时高亮并显示当前样式名
        val extraLabel = current.takeIf { it.isExtra }?.label
        ScanModePill(
            text = if (extraLabel != null) "更多·$extraLabel" else "更多",
            selected = extraLabel != null,
            trailing = Icons.Default.KeyboardArrowDown,
            onClick = { showMore = true }
        )
    }
    if (showMore) {
        ModePickerSheet(
            current = current,
            onSelect = { mode ->
                showMore = false
                onSelect(mode)
            },
            onDismiss = { showMore = false }
        )
    }
}

/** 单个样式 pill：选中高亮 accent。 */
@Composable
private fun ScanModePill(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    trailing: ImageVector? = null
) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(10.dp))
            .background(if (selected) AgentPurple.copy(alpha = 0.14f) else MaterialTheme.colorScheme.surfaceVariant)
            .border(1.dp, if (selected) AgentPurple.copy(alpha = 0.5f) else Color.Transparent, RoundedCornerShape(10.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            color = if (selected) AgentPurple else MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (trailing != null) {
            Icon(trailing, contentDescription = null, tint = if (selected) AgentPurple else MaterialTheme.colorScheme.outline, modifier = Modifier.size(14.dp))
        }
    }
}

/** 「更多」样式选择弹窗：按类分组（文档/票据/证件/效果增强）的图标宫格。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModePickerSheet(
    current: DocScanner.Mode,
    onSelect: (DocScanner.Mode) -> Unit,
    onDismiss: () -> Unit
) {
    val moreModes = DocScanner.Mode.entries.filter { it.isExtra }
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = MaterialTheme.colorScheme.surface
    ) {
        Text(
            text = "选择扫描样式",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp)
        )
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(start = 20.dp, end = 20.dp, bottom = 20.dp)
        ) {
            DocScanner.Mode.Category.entries.forEach { category ->
                Text(
                    text = category.label,
                    style = MaterialTheme.typography.titleSmall,
                    color = AgentPurple,
                    modifier = Modifier.padding(top = 14.dp, bottom = 8.dp)
                )
                // 每 3 个一行平铺（不再用嵌套 LazyVerticalGrid——其放在 verticalScroll 内会因无限高崩溃）
                moreModes.filter { it.category == category }
                    .chunked(3)
                    .forEach { rowModes ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            rowModes.forEach { mode ->
                                Box(modifier = Modifier.weight(1f)) {
                                    ScanModeTile(
                                        mode = mode,
                                        selected = mode == current,
                                        onClick = { onSelect(mode) }
                                    )
                                }
                            }
                            // 不足 3 个时用占位补权重，保持对齐
                            repeat(3 - rowModes.size) { Spacer(Modifier.weight(1f)) }
                        }
                        Spacer(Modifier.height(10.dp))
                    }
            }
        }
    }
}

/** 宫格里的单个样式卡片：图标 + 名称，选中态 accent 描边 + 右上角勾。 */
@Composable
private fun ScanModeTile(
    mode: DocScanner.Mode,
    selected: Boolean,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(if (selected) AgentPurple.copy(alpha = 0.12f) else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f))
            .border(1.dp, if (selected) AgentPurple else Color.Transparent, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(vertical = 12.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                mode.icon(),
                contentDescription = null,
                tint = if (selected) AgentPurple else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(26.dp)
            )
            Text(
                text = mode.label,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
                color = if (selected) AgentPurple else MaterialTheme.colorScheme.onSurface
            )
        }
        // 选中勾（右上角）
        if (selected) {
            Icon(
                Icons.Default.CheckCircle,
                contentDescription = "已选中",
                tint = AgentPurple,
                modifier = Modifier.align(Alignment.TopEnd).padding(4.dp).size(16.dp)
            )
        }
    }
}

/** 全屏图片预览：点背景/角标关闭，左右箭头在前一张/下一张间翻页。 */
@Composable
private fun ImagePreviewDialog(
    pages: List<ScanPage>,
    index: Int,
    onIndexChange: (Int) -> Unit,
    onDismiss: () -> Unit
) {
    val bmp = pages[index].scanned ?: pages[index].original
    Dialog(onDismissRequest = onDismiss) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .clickable(indication = null, interactionSource = remember { MutableInteractionSource() }) { onDismiss() }
        ) {
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "预览第 ${index + 1} 页",
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(8.dp)
            )
            // 右上角关闭
            IconButton(
                onClick = onDismiss,
                modifier = Modifier.align(Alignment.TopEnd).padding(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(Color.Black.copy(alpha = 0.5f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Close, contentDescription = "关闭预览", tint = Color.White, modifier = Modifier.size(18.dp))
                }
            }
            // 左右翻页（多于 1 页才显示）
            if (pages.size > 1) {
                IconButton(
                    onClick = { onIndexChange((index - 1).coerceAtLeast(0)) },
                    enabled = index > 0,
                    modifier = Modifier.align(Alignment.CenterStart).padding(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(Color.Black.copy(alpha = 0.5f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.AutoMirrored.Filled.KeyboardArrowLeft, contentDescription = "上一页", tint = Color.White, modifier = Modifier.size(24.dp))
                    }
                }
                IconButton(
                    onClick = { onIndexChange((index + 1).coerceAtMost(pages.size - 1)) },
                    enabled = index < pages.size - 1,
                    modifier = Modifier.align(Alignment.CenterEnd).padding(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(Color.Black.copy(alpha = 0.5f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, contentDescription = "下一页", tint = Color.White, modifier = Modifier.size(24.dp))
                    }
                }
            }
            // 页码指示
            Text(
                text = "${index + 1} / ${pages.size}",
                style = MaterialTheme.typography.labelMedium,
                color = Color.White,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp)
            )
        }
    }
}

/** 扫描样式 → 图标映射（material-icons-extended）。 */
private fun DocScanner.Mode.icon(): ImageVector = when (this) {
    DocScanner.Mode.DOCUMENT -> Icons.Default.Description
    DocScanner.Mode.BUSINESS_CARD -> Icons.Default.Contacts
    DocScanner.Mode.RECEIPT -> Icons.AutoMirrored.Filled.ReceiptLong
    DocScanner.Mode.ID_CARD -> Icons.Default.CreditCard
    DocScanner.Mode.TABLE -> Icons.Default.TableChart
    DocScanner.Mode.BOOK -> Icons.AutoMirrored.Filled.MenuBook
    DocScanner.Mode.WHITEBOARD -> Icons.Default.PresentToAll
    DocScanner.Mode.PHOTO -> Icons.Default.Photo
}

/** 缩略图上的小标签（如“已扫描”）。 */
@Composable
private fun TagBadge(text: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(color.copy(alpha = 0.15f))
            .padding(horizontal = 6.dp, vertical = 2.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelSmall,
            color = color,
            fontWeight = FontWeight.SemiBold
        )
    }
}

