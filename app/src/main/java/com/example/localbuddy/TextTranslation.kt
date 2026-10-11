package com.example.localbuddy

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

// 定义语言选项
data class Language(val code: String, val name: String, val flag: String)

/** 源语种"自动识别"哨兵码（各 ASR 智能体的源语种选择第一项）。 */
const val LANG_AUTO = "auto"

/**
 * 官网列出的支持语种（19 种）：跨语沟通智能体双方语种配置的可选集（TTS 跨语种音色克隆已验证覆盖）。
 * ⚠️ 语音识别（录音翻译 / 本地视频字幕 / 实时听音字幕）的源语种列表见 [asrLangs] + [asrDialects]；
 * 全量 50 语种见 [languages]，仅用于翻译目标。
 */
val supportedLangs = listOf(
    Language("zh", "中文", "🇨🇳"),
    Language("en", "英语", "🇺🇸"),
    Language("de", "德语", "🇩🇪"),
    Language("es", "西班牙语", "🇪🇸"),
    Language("fr", "法语", "🇫🇷"),
    Language("ja", "日语", "🇯🇵"),
    Language("it", "意大利语", "🇮🇹"),
    Language("hu", "匈牙利语", "🇭🇺"),
    Language("ko", "韩语", "🇰🇷"),
    Language("ru", "俄语", "🇷🇺"),
    Language("fa", "波斯语", "🇮🇷"),
    Language("ar", "阿拉伯语", "🇸🇦"),
    Language("pl", "波兰语", "🇵🇱"),
    Language("pt", "葡萄牙语", "🇵🇹"),
    Language("cs", "捷克语", "🇨🇿"),
    Language("da", "丹麦语", "🇩🇰"),
    Language("sv", "瑞典语", "🇸🇪"),
    Language("el", "希腊语", "🇬🇷"),
    Language("tr", "土耳其语", "🇹🇷")
)

/**
 * 语音识别（Qwen3-ASR 0.6B）官方支持的 30 种语种：ASR 源语种选择的可选集
 * （录音翻译 / 本地视频字幕 / 实时听音字幕共用）。
 * 语种码与官方 README 一致；[SpeechMTEngine] 的 ASR_LANG_EN 负责把码映射成
 * "language {Name}" 提示词里的英文名。
 */
val asrLangs = listOf(
    Language("zh", "中文", "🇨🇳"),
    Language("en", "英语", "🇺🇸"),
    Language("yue", "粤语", "🇨🇳"),
    Language("ar", "阿拉伯语", "🇸🇦"),
    Language("de", "德语", "🇩🇪"),
    Language("fr", "法语", "🇫🇷"),
    Language("es", "西班牙语", "🇪🇸"),
    Language("pt", "葡萄牙语", "🇵🇹"),
    Language("id", "印尼语", "🇮🇩"),
    Language("it", "意大利语", "🇮🇹"),
    Language("ko", "韩语", "🇰🇷"),
    Language("ru", "俄语", "🇷🇺"),
    Language("th", "泰语", "🇹🇭"),
    Language("vi", "越南语", "🇻🇳"),
    Language("ja", "日语", "🇯🇵"),
    Language("tr", "土耳其语", "🇹🇷"),
    Language("hi", "印地语", "🇮🇳"),
    Language("ms", "马来语", "🇲🇾"),
    Language("nl", "荷兰语", "🇳🇱"),
    Language("sv", "瑞典语", "🇸🇪"),
    Language("da", "丹麦语", "🇩🇰"),
    Language("fi", "芬兰语", "🇫🇮"),
    Language("pl", "波兰语", "🇵🇱"),
    Language("cs", "捷克语", "🇨🇿"),
    Language("fil", "菲律宾语", "🇵🇭"),
    Language("fa", "波斯语", "🇮🇷"),
    Language("el", "希腊语", "🇬🇷"),
    Language("hu", "匈牙利语", "🇭🇺"),
    Language("mk", "马其顿语", "🇲🇰"),
    Language("ro", "罗马尼亚语", "🇷🇴")
)

/**
 * 语音识别官方支持的 22 种中国方言：附在 30 语种之后作为源语种可选项。
 * 码为自定义标识（拼音 / 方言名），仅在本 App 内部流通；
 * [SpeechMTEngine] 的 ASR_LANG_EN 按其归口语种注入提示词英文名。
 */
val asrDialects = listOf(
    Language("anhui", "安徽话", "🇨🇳"),
    Language("dongbei", "东北话", "🇨🇳"),
    Language("fujian", "福建话", "🇨🇳"),
    Language("gansu", "甘肃话", "🇨🇳"),
    Language("guizhou", "贵州话", "🇨🇳"),
    Language("hebei", "河北话", "🇨🇳"),
    Language("henan", "河南话", "🇨🇳"),
    Language("hubei", "湖北话", "🇨🇳"),
    Language("hunan", "湖南话", "🇨🇳"),
    Language("jiangxi", "江西话", "🇨🇳"),
    Language("ningxia", "宁夏话", "🇨🇳"),
    Language("shandong", "山东话", "🇨🇳"),
    Language("shaanxi", "陕西话", "🇨🇳"),
    Language("shanxi", "山西话", "🇨🇳"),
    Language("sichuan", "四川话", "🇨🇳"),
    Language("tianjin", "天津话", "🇨🇳"),
    Language("yunnan", "云南话", "🇨🇳"),
    Language("zhejiang", "浙江话", "🇨🇳"),
    Language("yue-hk", "粤语（香港口音）", "🇨🇳"),
    Language("yue-gd", "粤语（广东口音）", "🇨🇳"),
    Language("wu", "吴语", "🇨🇳"),
    Language("minnan", "闽南语", "🇨🇳")
)

/** 按码查语种：ASR 30 语种 → 22 方言 → 19 官网语种 → 全量表。 */
fun langOf(code: String): Language? =
    (asrLangs + asrDialects).firstOrNull { it.code == code }
        ?: supportedLangs.firstOrNull { it.code == code }
        ?: languages.firstOrNull { it.code == code }

/** 源语种显示名（auto → "自动识别"）。 */
fun srcLangLabel(code: String): String =
    if (code == LANG_AUTO) "自动识别" else (langOf(code)?.let { "${it.flag} ${it.name}" } ?: code)

val languages = listOf(
    Language("zh", "中文", "🇨🇳"),
    Language("en", "英语", "🇬🇧"),
    Language("fr", "法语", "🇫🇷"),
    Language("pt", "葡萄牙语", "🇵🇹"),
    Language("es", "西班牙语", "🇪🇸"),
    Language("ja", "日语", "🇯🇵"),
    Language("tr", "土耳其语", "🇹🇷"),
    Language("ru", "俄语", "🇷🇺"),
    Language("ar", "阿拉伯语", "🇸🇦"),
    Language("ko", "韩语", "🇰🇷"),
    Language("th", "泰语", "🇹🇭"),
    Language("it", "意大利语", "🇮🇹"),
    Language("de", "德语", "🇩🇪"),
    Language("vi", "越南语", "🇻🇳"),
    Language("ms", "马来语", "🇲🇾"),
    Language("id", "印尼语", "🇮🇩"),
    Language("tl", "菲律宾语", "🇵🇭"),
    Language("hi", "印地语", "🇮🇳"),
    Language("zh-Hant", "繁体中文", "🇨🇳"),
    Language("pl", "波兰语", "🇵🇱"),
    Language("cs", "捷克语", "🇨🇸"),
    Language("nl", "荷兰语", "🇳🇱"),
    Language("km", "高棉语", "🇰🇭"),
    Language("my", "缅甸语", "🇲🇲"),
    Language("fa", "波斯语", "🇮🇷"),
    Language("gu", "古吉拉特语", "🇮🇳"),
    Language("ur", "乌尔都语", "🇵🇰"),
    Language("te", "泰卢固语", "🇮🇳"),
    Language("mr", "马拉地语", "🇮🇳"),
    Language("he", "希伯来语", "🇮🇱"),
    Language("bn", "孟加拉语", "🇧🇩"),
    Language("ta", "泰米尔语", "🇱🇰"),
    Language("uk", "乌克兰语", "🇺🇦"),
    Language("bo", "藏语", "🇨🇳"),
    Language("kk", "哈萨克语", "🇰🇿"),
    Language("mn", "蒙古语", "🇲🇳"),
    Language("ug", "维吾尔语", "🇨🇳"),
    Language("yue", "粤语", "🇨🇳")
)

/**
 * 翻译（MT）支持的语种码集合：与 [languages]（目标语种选择器的可选集）一致。
 * 语音识别能识别但翻译不支持的语种（如 sv / da / el / fi / hu / mk / ro）必须关闭翻译——
 * 否则 MT 会把原文原样吐出当译文。
 */
val mtSupportedCodes: Set<String> = languages.map { it.code }.toSet()

/**
 * ASR 语种码 → 翻译语种码：方言归到其所属语种（粤语口音 → yue，其余 → zh），
 * fil（菲律宾语）→ tl（[languages] 用的码），其余原样。
 */
fun asrToMtLang(code: String): String = when {
    code == "yue-hk" || code == "yue-gd" -> "yue"
    code == "fil" -> "tl"
    asrDialects.any { it.code == code } -> "zh"
    else -> code
}

/**
 * 配置阶段可判定的「不可翻译」原因；null 表示可翻译。
 * 源语种为 [LANG_AUTO] 时无法预知，只能在运行时按检测到的语种判断（[needsTranslation]）。
 */
fun translateBlockReason(srcCode: String, tgtCode: String): String? {
    if (srcCode == LANG_AUTO) return null
    val src = asrToMtLang(srcCode)
    if (src !in mtSupportedCodes) return "该语音语种暂不支持翻译"
    if (src == tgtCode) return "语音语种与目标语种相同，无需翻译"
    return null
}

/**
 * 运行时判定：源语种（ASR 检测或强制指定的码）是否需要翻译到 tgtCode。
 * 源语种未知（null / 空 / auto）时返回 true——交回 MT 自检，保持原有行为。
 */
fun needsTranslation(srcCode: String?, tgtCode: String): Boolean {
    val s = srcCode?.takeIf { it.isNotBlank() && it != LANG_AUTO } ?: return true
    val src = asrToMtLang(s)
    return src in mtSupportedCodes && src != tgtCode
}

// ViewModel 管理翻译状态和逻辑
class TranslationViewModel : ViewModel() {
    private val _sourceText = MutableStateFlow("")
    val sourceText: StateFlow<String> = _sourceText.asStateFlow()

    private val _translatedText = MutableStateFlow("")
    val translatedText: StateFlow<String> = _translatedText.asStateFlow()

    private val _sourceLanguage = MutableStateFlow(languages[0])
    val sourceLanguage: StateFlow<Language> = _sourceLanguage.asStateFlow()

    private val _targetLanguage = MutableStateFlow(languages[1])
    val targetLanguage: StateFlow<Language> = _targetLanguage.asStateFlow()

    private val _isTranslating = MutableStateFlow(false)
    val isTranslating: StateFlow<Boolean> = _isTranslating.asStateFlow()

    private var translationJob: Job? = null
    private lateinit var translationModel: TranslationModel

    fun initializeModel(context: Context) {
        if (!::translationModel.isInitialized) {
            translationModel = TranslationModel(context)
            viewModelScope.launch { SpeechMTEngine.get(context).warmUp() }
        }
    }

    fun updateSourceText(text: String) {
        _sourceText.value = text
        startTranslation(text)
    }

    fun setSourceLanguage(language: Language) {
        _sourceLanguage.value = language
        if (sourceText.value.isNotEmpty()) {
            startTranslation(sourceText.value)
        }
    }

    fun setTargetLanguage(language: Language) {
        _targetLanguage.value = language
        if (sourceText.value.isNotEmpty()) {
            startTranslation(sourceText.value)
        }
    }

    fun swapLanguages() {
        val currentSource = _sourceLanguage.value
        val currentTarget = _targetLanguage.value
        _sourceLanguage.value = currentTarget
        _targetLanguage.value = currentSource

        if (sourceText.value.isNotEmpty()) {
            val currentTranslated = _translatedText.value
            _sourceText.value = currentTranslated
            _translatedText.value = sourceText.value
        }
    }

    private fun startTranslation(text: String) {
        translationJob?.cancel()

        if (text.isEmpty()) {
            _translatedText.value = ""
            _isTranslating.value = false
            return
        }

        _isTranslating.value = true
        translationJob = viewModelScope.launch {
            try {
                val result = translationModel.translate(text, _sourceLanguage.value.code, _targetLanguage.value.code)
                _translatedText.value = result
            } catch (e: Exception) {
                _translatedText.value = "翻译出错: ${e.message}"
            } finally {
                _isTranslating.value = false
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        translationJob?.cancel()
    }
}

/**
 * 翻译模型：委托给 [SpeechMTEngine] 的 ONNX MT（Hunyuan-MT）推理，
 * 与「录音翻译」共用同一套模型/分词器/会话。
 */
class TranslationModel(context: Context) {
    private val engine = SpeechMTEngine.get(context)

    suspend fun translate(text: String, sourceLang: String, targetLang: String): String =
        engine.translate(text, sourceLang, targetLang)

    fun cleanup() {}
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextTranslationScreen(onBack: () -> Unit) {
    val viewModel: TranslationViewModel = viewModel()
    val context = LocalContext.current

    // 从ViewModel收集状态
    val sourceText by viewModel.sourceText.collectAsState()
    val translatedText by viewModel.translatedText.collectAsState()
    val sourceLanguage by viewModel.sourceLanguage.collectAsState()
    val targetLanguage by viewModel.targetLanguage.collectAsState()
    val isTranslating by viewModel.isTranslating.collectAsState()

    // 初始化模型
    LaunchedEffect(Unit) {
        viewModel.initializeModel(context)
    }

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = { Text("文本翻译") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "关闭"
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // 语言选择行
            LanguageSelectorRow(
                sourceLanguage = sourceLanguage,
                targetLanguage = targetLanguage,
                onSourceLanguageSelected = { viewModel.setSourceLanguage(it) },
                onTargetLanguageSelected = { viewModel.setTargetLanguage(it) },
                onSwapLanguages = { viewModel.swapLanguages() }
            )

            // 源语言输入框
            TranslationTextField(
                title = "输入文本",
                value = sourceText,
                onValueChange = { viewModel.updateSourceText(it) },
                placeholder = "请输入要翻译的文本...",
                isTranslating = false,
                modifier = Modifier.weight(1f)
            )

            // 统计信息
            Text(
                text = "字符数: ${sourceText.length}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant ,
                modifier = Modifier.align(Alignment.End)
            )

            // 翻译结果框
            TranslationTextField(
                title = "翻译结果",
                value = translatedText,
                onValueChange = {},
                placeholder = if (isTranslating) "正在翻译..." else "翻译结果将在这里显示...",
                isTranslating = isTranslating,
                readOnly = true,
                modifier = Modifier.weight(1f)
            )


        }
    }
}

@Composable
fun LanguageSelectorRow(
    sourceLanguage: Language,
    targetLanguage: Language,
    onSourceLanguageSelected: (Language) -> Unit,
    onTargetLanguageSelected: (Language) -> Unit,
    onSwapLanguages: () -> Unit
) {
    var showSourceDialog by remember { mutableStateOf(false) }
    var showTargetDialog by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // 源语言选择
        LanguageSelector(
            language = sourceLanguage,
            onClick = { showSourceDialog = true },
            modifier = Modifier.weight(1f)
        )

        // 交换按钮
        IconButton(
            onClick = onSwapLanguages,
            modifier = Modifier.size(48.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Menu,
                contentDescription = "交换语言",
                tint = MaterialTheme.colorScheme.primary
            )
        }

        // 目标语言选择
        LanguageSelector(
            language = targetLanguage,
            onClick = { showTargetDialog = true },
            modifier = Modifier.weight(1f)
        )
    }

    // 语言选择对话框
    if (showSourceDialog) {
        LanguageSelectionDialog(
            selectedLanguage = sourceLanguage,
            onLanguageSelected = {
                onSourceLanguageSelected(it)
                showSourceDialog = false
            },
            onDismiss = { showSourceDialog = false }
        )
    }

    if (showTargetDialog) {
        LanguageSelectionDialog(
            selectedLanguage = targetLanguage,
            onLanguageSelected = {
                onTargetLanguageSelected(it)
                showTargetDialog = false
            },
            onDismiss = { showTargetDialog = false }
        )
    }
}

@Composable
fun LanguageSelector(
    language: Language,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    OutlinedCard(
        onClick = onClick,
        modifier = modifier,
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = language.flag,
                fontSize = 20.sp,
                modifier = Modifier.width(30.dp),
                textAlign = TextAlign.Center
            )
            Text(
                text = language.name,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.weight(1f)
            )
            Icon(
                imageVector = Icons.Default.ArrowDropDown,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
fun LanguageSelectionDialog(
    selectedLanguage: Language,
    onLanguageSelected: (Language) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("选择语言") },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                languages.forEach { language ->
                    LanguageItem(
                        language = language,
                        isSelected = language.code == selectedLanguage.code,
                        onClick = { onLanguageSelected(language) }
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("取消")
            }
        }
    )
}

@Composable
fun LanguageItem(
    language: Language,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Card(
        onClick = onClick,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surface
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = language.flag,
                fontSize = 20.sp,
                modifier = Modifier.width(30.dp),
                textAlign = TextAlign.Center
            )
            Text(
                text = language.name,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                modifier = Modifier.weight(1f)
            )
            if (isSelected) {
                Icon(
                    imageVector = Icons.Default.Check,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
fun TranslationTextField(
    title: String,
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    isTranslating: Boolean,
    readOnly: Boolean = false,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Medium
        )

        Box(
            modifier = Modifier
                .fillMaxSize()
                .clip(RoundedCornerShape(12.dp))
                .border(
                    width = 1.dp,
                    color = MaterialTheme.colorScheme.outlineVariant,
                    shape = RoundedCornerShape(12.dp)
                )
                .background(MaterialTheme.colorScheme.surface)
        ) {
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                readOnly = readOnly,
                enabled = !readOnly,
                textStyle = TextStyle(
                    fontSize = 16.sp,
                    lineHeight = 24.sp,
                    color = if (readOnly) MaterialTheme.colorScheme.onSurface
                    else MaterialTheme.colorScheme.onBackground
                ),
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                decorationBox = { innerTextField ->
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.TopStart
                    ) {
                        if (value.isEmpty() && !isTranslating) {
                            Text(
                                text = placeholder,
                                fontSize = 16.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                lineHeight = 24.sp
                            )
                        }
                        innerTextField()
                    }
                }
            )

            // 翻译中的加载指示器
            if (isTranslating) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.05f)),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(36.dp),
                        strokeWidth = 3.dp
                    )
                }
            }
        }
    }
}
