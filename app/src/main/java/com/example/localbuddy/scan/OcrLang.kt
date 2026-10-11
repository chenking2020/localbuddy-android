package com.example.localbuddy.scan

/**
 * 识别语言 → PP-OCRv5 识别模型目录。检测(det)通用、识别(rec)按脚本/语言分模型。
 *
 * dir 为 assets/ocr 下的模型目录名；默认「自动」= CN 模型（简/繁/英/日）。
 * 注：`arabic_*` 若尚未放入 assets，则不出现在列表（保持健壮）。
 */
object OcrLang {

    /** 期望的 rec 模型及其中文名（顺序即 UI 展示顺序；第一项为默认=CN）。 */
    val MODELS: List<OcrModel> = listOf(
        OcrModel("PP-OCRv5_mobile_rec_onnx", "中文 / 英文 / 日文(自动)"),
        OcrModel("latin_PP-OCRv5_mobile_rec_onnx", "拉丁文（法/德/西/意等）"),
        OcrModel("korean_PP-OCRv5_mobile_rec_onnx", "韩文"),
        OcrModel("th_PP-OCRv5_mobile_rec_onnx", "泰文"),
        OcrModel("el_PP-OCRv5_mobile_rec_onnx", "希腊文"),
        OcrModel("cyrillic_PP-OCRv5_mobile_rec_onnx", "西里尔文（俄/乌克兰等）"),
        OcrModel("devanagari_PP-OCRv5_mobile_rec_onnx", "天城文（印地语等）"),
        OcrModel("ta_PP-OCRv5_mobile_rec_onnx", "泰米尔文"),
        OcrModel("te_PP-OCRv5_mobile_rec_onnx", "泰卢固文"),
        OcrModel("arabic_PP-OCRv5_mobile_rec_onnx", "阿拉伯文（阿拉伯/波斯等）"),
    )

    data class OcrModel(val dir: String, val label: String)

    const val DEFAULT_DIR = "PP-OCRv5_mobile_rec_onnx"

    fun label(dir: String): String = MODELS.firstOrNull { it.dir == dir }?.label ?: dir
}

/** 解析好的识别模型配置（dict / 尺寸），ONNX session 由引擎按 dir 懒加载持有。 */
class OcrRecModel(val dir: String, val dict: List<String>, val recH: Int, val maxW: Int) {
    override fun toString(): String = "Rec($dir, dict=${dict.size}, h=$recH, w<=$maxW)"
}