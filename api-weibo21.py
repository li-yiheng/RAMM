# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import mimetypes
import logging
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm



DATA_SPLITS = [
    {
        "name": "train",
        "input_xlsx": "/root/weibo21/train_datasets.xlsx",
        "output_json": "train_datas_narrative_category.json",
    },
    {
        "name": "test",
        "input_xlsx": "/root/weibo21/test_datasets.xlsx",
        "output_json": "test_datas_narrative_category.json",
    },
]


IMAGE_ROOT = "/root/weibo21"


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


MODEL_NAME = "qwen-vl-max"


MAX_WORKERS = 8
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 120


TEMPERATURE_MAIN = 0.7


TEMPERATURE_CATEGORY_ONLY = 0.1

MAX_TOKENS_MAIN = 256
MAX_TOKENS_CATEGORY_ONLY = 32


LIMIT = None


OUTPUT_NARRATIVE_FIELD = "core_narrative"
OUTPUT_CATEGORY_FIELD = "category"


OUTPUT_ORIGINAL_CATEGORY_FIELD = "original_category"


LOG_LEVEL = logging.INFO




ALLOWED_CATEGORIES = [
    "经济",
    "健康",
    "军事",
    "科学",
    "政治",
    "国际",
    "教育",
    "娱乐",
    "社会",
]

CATEGORY_SET = set(ALLOWED_CATEGORIES)

CATEGORY_DEFINITIONS = """
类别定义（只能选一个最主要类别）：
1. 经济：宏观经济、金融、股市、公司经营、产业、贸易、消费价格、就业、财经政策
2. 健康：医疗、公共卫生、疾病、疫情、药物、保健、医院、医患、营养
3. 军事：军队、武器、战争、军演、国防、战术、军事冲突、安全防务
4. 科学：科研发现、技术创新、航天、人工智能、实验、工程科技、自然科学
5. 政治：政府、政策、官员、选举、党派、治理、政治争议、公共权力
6. 国际：外交、国际关系、国际组织、跨国事务、国家间合作或摩擦（非纯军事中心）
7. 教育：学校、考试、招生、教师、学生、课程、校园、教育制度与教育政策
8. 娱乐：明星、影视、综艺、音乐、演出、文娱八卦、粉丝事件
9. 社会：民生、法治、社会事件、事故、灾害、治安、伦理、公共生活、社会现象

判别规则：
- 每条新闻只能输出一个类别。
- 如果内容可能跨多个类别，选择“最核心、最主导、最直接”的主题类别。
- 若是国际冲突且军事信息最核心，归为“军事”。
- 若是国际新闻但重点是外交、合作、制裁、国际组织、跨国事务，归为“国际”。
- 若是科技公司融资、产业、市场、商业模式，优先看“经济”；若重点是技术突破本身，归为“科学”。
- 若是灾害、事故、公共安全、民生舆情，归为“社会”。
""".strip()




logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)




SYSTEM_PROMPT = """
你是一名专注于媒体内容分析、多模态新闻理解与虚假信息研究的专家级 AI 助手。

你的任务有两个，必须同时完成：
任务一：分析新闻文本与配图，提炼其“核心叙事”或“中心主张”。
任务二：根据新闻最核心的主题，将其划分为且仅划分为一个类别。

全局要求：
1. 必须联合参考文本与图片，不得只依据其中单一模态。
2. 对核心叙事的提炼必须客观、中性、克制，不判断真伪，不带立场。
3. 核心叙事必须是单句中文，只保留底层中心主张，不复述过多表面细节。
4. 类别必须且只能从给定九类中选择一个。
5. 输出必须是严格 JSON，对象中只能包含两个键：
   - "core_narrative"
   - "category"
6. 不要输出 Markdown，不要输出解释，不要输出额外字段，不要输出任何 JSON 之外的内容。
""".strip()

USER_PROMPT_TEMPLATE = """
下面先给出类别体系与示例。注意：以下示例中的“图像说明”只是为了帮助你理解任务，不是要求你把正式样本中的图片转成文字后再判断；在正式样本里，你会直接接收原始图像。

{category_definitions}

示例1：
文本：
“爆炸性画面！XX县一名投票站工作人员被隐藏摄像头拍到正在撕毁选票，选举被操纵的证据已经无法否认！”
图像说明：
一个人站在纸篓旁，画面模糊，暗示与选票处理有关。
正确输出：
{{"core_narrative":"选举舞弊已被确证。","category":"政治"}}

示例2：
文本：
“最新研究显示某新型疫苗可显著降低重症风险，专家建议高风险人群尽快接种。”
图像说明：
医护人员在接种点为居民注射疫苗。
正确输出：
{{"core_narrative":"该疫苗被描述为能有效降低重症风险并值得推广接种。","category":"健康"}}

示例3：
文本：
“受国际原油价格上升和运输成本增加影响，多地生活必需品价格出现上涨。”
图像说明：
超市货架与价格标签，强调物价变化。
正确输出：
{{"core_narrative":"能源与运输成本上升被认为推动了生活必需品价格上涨。","category":"经济"}}

现在请处理正式样本。

正式样本：
文本：
{text_content}
图像：
[你将直接接收该新闻的原始配图]

请只输出严格 JSON，格式如下：
{{"core_narrative":"...","category":"..."}}
""".strip()

CATEGORY_ONLY_SYSTEM_PROMPT = """
你是一名新闻主题分类助手。
请根据新闻文本和图片，将新闻归入且仅归入以下九类之一：
经济、健康、军事、科学、政治、国际、教育、娱乐、社会。

要求：
1. 只能输出一个类别名称。
2. 不要输出解释，不要输出标点，不要输出额外内容。
3. 如果存在多类别交叉，选择最核心主题。
""".strip()

CATEGORY_ONLY_USER_PROMPT_TEMPLATE = """
{category_definitions}

正式样本：
文本：
{text_content}
图像：
[你将直接接收该新闻的原始配图]

只输出一个类别名称。
""".strip()


# ================= 5. 客户端初始化 =================

def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "未检测到环境变量 DASHSCOPE_API_KEY。"
            "请先执行：export DASHSCOPE_API_KEY='你的密钥'"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


client = build_client()


# ================= 6. Weibo21 数据读取工具 =================

def normalize_column_name(col: Any) -> str:
    return str(col).strip().lower()


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True
) -> Optional[str]:
    """
    在 DataFrame 中查找目标列。
    """
    normalized_map = {
        normalize_column_name(col): col
        for col in df.columns
    }

    normalized_candidates = [
        normalize_column_name(c)
        for c in candidates
    ]

    for cand in normalized_candidates:
        if cand in normalized_map:
            return normalized_map[cand]

    for norm_col, original_col in normalized_map.items():
        for cand in normalized_candidates:
            if cand in norm_col:
                return original_col

    if required:
        raise KeyError(
            f"未找到必要列。候选列名={candidates}；"
            f"当前 xlsx 实际列名={list(df.columns)}"
        )

    return None


def is_nan_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def safe_str(value: Any) -> str:
    if is_nan_like(value):
        return ""
    return str(value).strip()


def format_id_value(value: Any, fallback: int) -> str:
    if is_nan_like(value):
        return str(fallback)

    if isinstance(value, bool):
        return str(int(value))

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    value_str = str(value).strip()

    if value_str == "":
        return str(fallback)

    if re.fullmatch(r"\d+\.0", value_str):
        return value_str[:-2]

    return value_str


def parse_label(value: Any) -> Optional[int]:
    if is_nan_like(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return int(round(value))

    value_str = str(value).strip()
    if value_str == "":
        return None

    lower = value_str.lower()

    if lower in {"0", "0.0", "false", "real", "nonrumor", "non-rumor", "non_rumor"}:
        return 0

    if lower in {"1", "1.0", "true", "fake", "rumor"}:
        return 1

    try:
        return int(float(value_str))
    except Exception:
        return None


def clean_image_path(path_text: str) -> str:
    path_text = str(path_text).strip()
    path_text = path_text.strip("[](){}'\"")
    path_text = path_text.replace("\\", "/")
    return path_text


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output


IMAGE_PATH_PATTERN = re.compile(
    r"(?:(?:\.{1,2}[\\/])?)*(?:nonrumor_images|rumor_images)[\\/][^\s\]\[;,|'\"]+?\.(?:jpg|jpeg|png|bmp|webp)",
    re.IGNORECASE
)

FALLBACK_IMAGE_FILENAME_PATTERN = re.compile(
    r"[^\s\]\[;,|'\"]+?\.(?:jpg|jpeg|png|bmp|webp)",
    re.IGNORECASE
)


def extract_image_paths_from_cell(cell_value: Any) -> List[str]:
    text = safe_str(cell_value)

    if text == "":
        return []

    matched_paths = IMAGE_PATH_PATTERN.findall(text)

    if not matched_paths:
        matched_paths = FALLBACK_IMAGE_FILENAME_PATTERN.findall(text)

    cleaned = [
        clean_image_path(path)
        for path in matched_paths
        if clean_image_path(path) != ""
    ]

    return unique_keep_order(cleaned)


def resolve_image_path(rel_or_abs_path: str) -> str:
    raw_path = clean_image_path(rel_or_abs_path)

    if not raw_path:
        return ""

    path_obj = Path(raw_path)
    image_root = Path(IMAGE_ROOT)

    candidates: List[Path] = []

    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        candidates.append(image_root / raw_path)
        candidates.append(Path.cwd() / raw_path)

        filename = path_obj.name
        candidates.append(image_root / filename)
        candidates.append(image_root / "rumor_images" / filename)
        candidates.append(image_root / "nonrumor_images" / filename)

    dedup_candidates: List[Path] = []
    seen = set()

    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            dedup_candidates.append(candidate)
            seen.add(key)

    for candidate in dedup_candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    return str(dedup_candidates[0]) if dedup_candidates else ""


def select_first_valid_image_path(image_paths: List[str]) -> str:
    if not image_paths:
        return ""

    resolved_paths = [
        resolve_image_path(p)
        for p in image_paths
        if safe_str(p) != ""
    ]

    for path in resolved_paths:
        if path and os.path.exists(path):
            return path

    return resolved_paths[0] if resolved_paths else ""


def load_weibo21_xlsx(path: str) -> List[Dict[str, Any]]:
    logger.info("正在读取 Weibo21 xlsx: %s", path)

    df = pd.read_excel(path)
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"xlsx 文件为空: {path}")

    id_col = find_column(
        df,
        candidates=[
            "id",
            "mid",
            "weibo_id",
            "微博id",
            "微博ID",
            "idx",
            "index",
            "编号",
            "序号",
            "unnamed: 0"
        ],
        required=False
    )

    label_col = find_column(
        df,
        candidates=[
            "label",
            "labels",
            "target",
            "rumor_label",
            "真假标签",
            "标签"
        ],
        required=True
    )

    original_category_col = find_column(
        df,
        candidates=[
            "original_category",
            "原始类别",
            "原类别",
            "category",
            "类别",
            "topic",
            "domain",
            "source",
            "news_category"
        ],
        required=False
    )

    content_col = find_column(
        df,
        candidates=[
            "content",
            "text",
            "正文",
            "文本",
            "微博内容"
        ],
        required=True
    )

    image_col = find_column(
        df,
        candidates=[
            "image",
            "images",
            "img",
            "picture",
            "pic",
            "图片",
            "图像"
        ],
        required=True
    )

    logger.info(
        "列匹配结果: id_col=%s, label_col=%s, original_category_col=%s, "
        "content_col=%s, image_col=%s",
        id_col,
        label_col,
        original_category_col,
        content_col,
        image_col
    )

    data: List[Dict[str, Any]] = []

    for row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Loading {Path(path).name}"
    ):
        if id_col is not None:
            item_id = format_id_value(row.get(id_col, row_index), row_index)
        else:
            item_id = str(row_index)

        label = parse_label(row.get(label_col, None))
        if label is None:
            label = 0

        text = safe_str(row.get(content_col, ""))

        image_raw = safe_str(row.get(image_col, ""))
        image_paths = extract_image_paths_from_cell(image_raw)

        if image_paths:
            output_image = image_paths[0]
        else:
            output_image = image_raw

        original_category = (
            safe_str(row.get(original_category_col, ""))
            if original_category_col is not None
            else ""
        )

        item: Dict[str, Any] = {
            "id": item_id,
            "image": output_image,
            "text": text,
            "label": int(label),
            OUTPUT_ORIGINAL_CATEGORY_FIELD: original_category,

            # 内部字段：只用于 process_item 找图片，最终保存时会过滤掉
            "image_paths": image_paths,
        }

        data.append(item)

    logger.info("读取完成: %s，共 %d 条", path, len(data))
    return data




def detect_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "image/jpeg"


def encode_image_to_data_url(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        mime_type = detect_mime_type(image_path)
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"
    except Exception as e:
        logger.error("图片读取失败: %s | %s", image_path, str(e))
        return None


def clean_text_for_prompt(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_content_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")).strip())
                elif "text" in part:
                    texts.append(str(part.get("text", "")).strip())
            else:
                texts.append(str(part).strip())
        return "\n".join([t for t in texts if t]).strip()

    return str(content).strip()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    text = strip_code_fence(text)

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


def normalize_narrative(text: Any) -> str:
    if text is None:
        return "未成功生成核心叙事。"

    text = str(text).strip()

    text = re.sub(r"^\s*(核心叙事|核心摘要|摘要|叙事)\s*[:：]\s*", "", text)
    text = text.replace("**", "").replace("`", "").replace("#", "")
    text = text.strip(" \t\n\r\"'“”‘’")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "未成功生成核心叙事。"

    parts = re.split(r"(?<=[。！？!?；;])\s+", text)
    if parts and parts[0].strip():
        text = parts[0].strip()

    m = re.search(r"[。！？!?；;]", text)
    if m:
        text = text[:m.end()].strip()

    if text and text[-1] not in "。！？!?；;":
        text += "。"

    return text


def normalize_category(raw_category: Any) -> Optional[str]:
    if raw_category is None:
        return None

    c = str(raw_category).strip()
    if not c:
        return None

    c = c.replace("类目", "").replace("类别", "").replace("分类", "").strip()
    c = c.strip(" \t\n\r\"'“”‘’：:。；;,.，、")

    if c in CATEGORY_SET:
        return c

    low = c.lower()

    mapping = {
        "finance": "经济",
        "financial": "经济",
        "economy": "经济",
        "business": "经济",
        "财经": "经济",
        "金融": "经济",
        "商业": "经济",
        "产业": "经济",
        "贸易": "经济",

        "health": "健康",
        "medical": "健康",
        "medicine": "健康",
        "public health": "健康",
        "医疗": "健康",
        "卫生": "健康",
        "医药": "健康",
        "疫情": "健康",
        "疾病": "健康",

        "military": "军事",
        "defense": "军事",
        "war": "军事",
        "army": "军事",
        "国防": "军事",
        "战争": "军事",
        "军队": "军事",
        "军演": "军事",
        "安全防务": "军事",

        "science": "科学",
        "technology": "科学",
        "tech": "科学",
        "科研": "科学",
        "科技": "科学",
        "航天": "科学",
        "人工智能": "科学",
        "ai": "科学",

        "politics": "政治",
        "political": "政治",
        "government": "政治",
        "election": "政治",
        "policy": "政治",
        "政府": "政治",
        "政策": "政治",
        "选举": "政治",
        "官员": "政治",

        "international": "国际",
        "world": "国际",
        "foreign": "国际",
        "diplomacy": "国际",
        "外交": "国际",
        "国际关系": "国际",
        "国际事务": "国际",
        "跨国": "国际",

        "education": "教育",
        "school": "教育",
        "campus": "教育",
        "exam": "教育",
        "教师": "教育",
        "学生": "教育",
        "学校": "教育",
        "考试": "教育",
        "招生": "教育",

        "entertainment": "娱乐",
        "celebrity": "娱乐",
        "movie": "娱乐",
        "film": "娱乐",
        "music": "娱乐",
        "variety": "娱乐",
        "明星": "娱乐",
        "影视": "娱乐",
        "综艺": "娱乐",
        "文娱": "娱乐",

        "society": "社会",
        "social": "社会",
        "public event": "社会",
        "民生": "社会",
        "法治": "社会",
        "事故": "社会",
        "灾害": "社会",
        "灾难": "社会",
        "治安": "社会",
        "社会事件": "社会",
    }

    if low in mapping:
        return mapping[low]

    if c in mapping:
        return mapping[c]

    substring_rules: List[Tuple[List[str], str]] = [
        (["财经", "金融", "股市", "物价", "通胀", "产业", "公司", "贸易", "宏观经济"], "经济"),
        (["健康", "医疗", "卫生", "疫苗", "疫情", "疾病", "医院", "药物"], "健康"),
        (["军事", "国防", "军队", "战争", "军演", "武器", "导弹", "防务"], "军事"),
        (["科学", "科技", "科研", "技术", "航天", "实验", "人工智能", "ai"], "科学"),
        (["政治", "政府", "政策", "选举", "党派", "官员", "立法", "治理"], "政治"),
        (["国际", "外交", "联合国", "跨国", "国际关系", "制裁"], "国际"),
        (["教育", "学校", "学生", "教师", "考试", "高考", "招生", "校园"], "教育"),
        (["娱乐", "明星", "影视", "综艺", "音乐", "演出", "粉丝"], "娱乐"),
        (["社会", "民生", "法治", "事故", "灾害", "灾难", "治安", "舆情"], "社会"),
    ]

    for keywords, target in substring_rules:
        for kw in keywords:
            if kw.lower() in low or kw in c:
                return target

    return None


def keyword_fallback_category(text: str) -> str:
    t = (text or "").lower()

    rules: List[Tuple[List[str], str]] = [
        (["股市", "金融", "财经", "贸易", "通胀", "经济", "公司", "企业", "产业", "市场", "投资", "就业"], "经济"),
        (["疫情", "疫苗", "医院", "医生", "健康", "卫生", "疾病", "药物", "感染", "病例"], "健康"),
        (["军队", "战争", "导弹", "军演", "武器", "国防", "战机", "海军", "陆军", "空军"], "军事"),
        (["科学", "科技", "科研", "实验", "芯片", "人工智能", "ai", "航天", "卫星", "火箭"], "科学"),
        (["政府", "选举", "政策", "官员", "议会", "总统", "总理", "立法", "政治", "党派"], "政治"),
        (["外交", "联合国", "国际", "多国", "跨国", "制裁", "国际组织", "峰会"], "国际"),
        (["学校", "教育", "考试", "高考", "教师", "学生", "招生", "校园", "课堂"], "教育"),
        (["明星", "娱乐", "综艺", "影视", "电影", "电视剧", "音乐", "演唱会", "粉丝"], "娱乐"),
        (["事故", "灾害", "灾难", "民生", "社会", "治安", "舆情", "交通", "火灾", "地震"], "社会"),
    ]

    for keywords, category in rules:
        if any(kw in t for kw in keywords):
            return category

    return "社会"


# ================= 8. Prompt 构造 =================

def build_main_messages(weibo_text: str, image_data_url: str) -> List[Dict[str, Any]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        text_content=weibo_text,
        category_definitions=CATEGORY_DEFINITIONS
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ],
        },
    ]


def build_category_only_messages(weibo_text: str, image_data_url: str) -> List[Dict[str, Any]]:
    user_prompt = CATEGORY_ONLY_USER_PROMPT_TEMPLATE.format(
        text_content=weibo_text,
        category_definitions=CATEGORY_DEFINITIONS
    )

    return [
        {
            "role": "system",
            "content": CATEGORY_ONLY_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ],
        },
    ]




def safe_chat_completion(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    item_id: Optional[Any] = None,
) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            content = completion.choices[0].message.content
            return extract_content_text(content)

        except Exception as e:
            last_error = e
            logger.warning(
                "API 调用失败 | item_id=%s | 第 %d/%d 次重试 | error=%s",
                str(item_id),
                attempt,
                MAX_RETRIES,
                str(e)
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    return f"API调用失败：{str(last_error)}"


def parse_main_result(raw_text: str) -> Tuple[str, Optional[str]]:
    raw_text = raw_text.strip()
    json_text = extract_json_object(raw_text)

    if json_text:
        try:
            obj = json.loads(json_text)
            narrative = normalize_narrative(obj.get("core_narrative", ""))
            category = normalize_category(obj.get("category", ""))
            return narrative, category
        except Exception:
            pass

    narrative = ""
    category = None

    m1 = re.search(
        r'(?:core_narrative|核心叙事)\s*["：: ]+\s*["“]?(.+?)["”]?(?:,|\n|$)',
        raw_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m1:
        narrative = m1.group(1).strip()

    m2 = re.search(
        r'(?:category|类别)\s*["：: ]+\s*["“]?(.+?)["”]?(?:,|\n|$)',
        raw_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if m2:
        category = normalize_category(m2.group(1).strip())

    if not narrative:
        narrative = raw_text

    narrative = normalize_narrative(narrative)
    return narrative, category


def call_main_task(
    weibo_text: str,
    image_data_url: str,
    item_id: Optional[Any] = None
) -> Tuple[str, Optional[str], str]:
    messages = build_main_messages(weibo_text, image_data_url)
    raw_response = safe_chat_completion(
        messages=messages,
        model=MODEL_NAME,
        temperature=TEMPERATURE_MAIN,
        max_tokens=MAX_TOKENS_MAIN,
        item_id=item_id,
    )

    if raw_response.startswith("API调用失败："):
        return "未成功生成核心叙事。", None, raw_response

    narrative, category = parse_main_result(raw_response)
    return narrative, category, raw_response


def call_category_only(
    weibo_text: str,
    image_data_url: str,
    item_id: Optional[Any] = None
) -> Optional[str]:
    messages = build_category_only_messages(weibo_text, image_data_url)
    raw_response = safe_chat_completion(
        messages=messages,
        model=MODEL_NAME,
        temperature=TEMPERATURE_CATEGORY_ONLY,
        max_tokens=MAX_TOKENS_CATEGORY_ONLY,
        item_id=item_id,
    )

    if raw_response.startswith("API调用失败："):
        return None

    return normalize_category(raw_response)




def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(item)

    image_paths = item.get("image_paths", [])
    if not isinstance(image_paths, list):
        image_paths = extract_image_paths_from_cell(image_paths)

    image_path = select_first_valid_image_path(image_paths)

    if not image_path:
        image_field = item.get("image", "")
        image_path = resolve_image_path(image_field)

    weibo_text = clean_text_for_prompt(item.get("text", ""))
    item_id = item.get("id", None)

    if not item.get("image", "") and not image_paths:
        result[OUTPUT_NARRATIVE_FIELD] = "图片字段缺失，无法生成核心叙事。"
        result[OUTPUT_CATEGORY_FIELD] = "社会"
        return result

    image_data_url = encode_image_to_data_url(image_path)
    if not image_data_url:
        result[OUTPUT_NARRATIVE_FIELD] = f"图片文件未找到或无法读取：{image_path}"
        result[OUTPUT_CATEGORY_FIELD] = keyword_fallback_category(weibo_text)
        return result

    if not weibo_text:
        result[OUTPUT_NARRATIVE_FIELD] = "文本为空，无法生成核心叙事。"
        result[OUTPUT_CATEGORY_FIELD] = "社会"
        return result

    narrative, category, raw_response = call_main_task(
        weibo_text=weibo_text,
        image_data_url=image_data_url,
        item_id=item_id
    )

    if category not in CATEGORY_SET:
        category = call_category_only(
            weibo_text=weibo_text,
            image_data_url=image_data_url,
            item_id=item_id
        )

    if category not in CATEGORY_SET:
        category = keyword_fallback_category(weibo_text)

    result[OUTPUT_NARRATIVE_FIELD] = narrative
    result[OUTPUT_CATEGORY_FIELD] = category

    return result




def ordered_output_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "image": item.get("image", ""),
        "text": item.get("text", ""),
        "label": item.get("label", 0),
        OUTPUT_ORIGINAL_CATEGORY_FIELD: item.get(OUTPUT_ORIGINAL_CATEGORY_FIELD, ""),
        OUTPUT_NARRATIVE_FIELD: item.get(OUTPUT_NARRATIVE_FIELD, ""),
        OUTPUT_CATEGORY_FIELD: item.get(OUTPUT_CATEGORY_FIELD, ""),
    }


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)

    ordered_data = [
        ordered_output_item(item)
        for item in data
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered_data, f, ensure_ascii=False, indent=2)


# ================= 12. 单个 split 主流程 =================

def process_split(split_config: Dict[str, str]) -> None:
    split_name = split_config["name"]
    input_xlsx = split_config["input_xlsx"]
    output_json = split_config["output_json"]

    logger.info("========== 开始处理 split: %s ==========", split_name)
    logger.info("输入 xlsx: %s", input_xlsx)
    logger.info("输出 JSON: %s", output_json)
    logger.info("模型名称: %s", MODEL_NAME)
    logger.info("类别集合: %s", " / ".join(ALLOWED_CATEGORIES))

    try:
        data = load_weibo21_xlsx(input_xlsx)
    except Exception as e:
        logger.error("读取输入 xlsx 失败: %s", str(e))
        return

    if LIMIT is not None:
        batch = data[:LIMIT]
        logger.info("当前仅处理前 %d 条样本", len(batch))
    else:
        batch = data
        logger.info("当前处理全部 %d 条样本", len(batch))

    results: List[Tuple[int, Dict[str, Any]]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_item, item): idx
            for idx, item in enumerate(batch)
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"Processing {split_name}"
        ):
            idx = future_to_idx[future]

            try:
                processed_item = future.result()
            except Exception as e:
                logger.exception("第 %d 条样本处理失败: %s", idx, str(e))
                raw_item = dict(batch[idx])
                raw_item[OUTPUT_NARRATIVE_FIELD] = f"处理失败：{str(e)}"
                raw_item[OUTPUT_CATEGORY_FIELD] = keyword_fallback_category(
                    clean_text_for_prompt(raw_item.get("text", ""))
                )
                processed_item = raw_item

            results.append((idx, processed_item))

    results.sort(key=lambda x: x[0])
    ordered_results = [x[1] for x in results]

    if ordered_results:
        preview = ordered_output_item(ordered_results[0])
        logger.info("----- 效果预览 -----")
        logger.info("ID: %s", str(preview.get("id", "N/A")))
        logger.info("Image: %s", str(preview.get("image", "")))
        logger.info("Text: %s", clean_text_for_prompt(preview.get("text", ""))[:150])
        logger.info("Label: %s", str(preview.get("label", "")))
        logger.info("Original Category: %s", str(preview.get(OUTPUT_ORIGINAL_CATEGORY_FIELD, "")))
        logger.info("%s: %s", OUTPUT_NARRATIVE_FIELD, preview.get(OUTPUT_NARRATIVE_FIELD, ""))
        logger.info("%s: %s", OUTPUT_CATEGORY_FIELD, preview.get(OUTPUT_CATEGORY_FIELD, ""))
        logger.info("--------------------")

    try:
        save_json(output_json, ordered_results)
        logger.info("处理完成，结果已保存到: %s", output_json)
    except Exception as e:
        logger.error("保存输出 JSON 失败: %s", str(e))

    logger.info("========== split: %s 处理结束 ==========", split_name)


# ================= 13. 主流程 =================

def main() -> None:
    logger.info("开始执行 Weibo21 多模态核心叙事 + 单标签九分类任务")
    logger.info("图片根目录: %s", IMAGE_ROOT)

    for split_config in DATA_SPLITS:
        process_split(split_config)

    logger.info("全部任务处理完成。")


if __name__ == "__main__":
    main()
