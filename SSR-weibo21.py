import os
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline as modelscope_pipeline



TRAIN_EXCEL_PATH = "/mnt/workspace/weibo21/train_datasets.xlsx"
TEST_EXCEL_PATH = "/mnt/workspace/weibo21/test_datasets.xlsx"


IMAGE_ROOT_DIR = "/mnt/workspace/weibo21"


OUTPUT_BASE_DIR = "SSR_Weibo21_PaperMethod_XLSX_BlankImageFallback"


K_VALUE = 6


WEIGHT_IMAGE = 0.5
WEIGHT_TEXT = 0.5

MODEL_ID = "iic/multi-modal_clip-vit-large-patch14_336_zh"
MODEL_REVISION = "v1.0.1"


NORMALIZE_EMBEDDINGS = False


BLANK_IMAGE_SIZE = 336
BLANK_IMAGE_COLOR = (255, 255, 255)


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)



logger.info("正在使用 ModelScope 初始化中文 CLIP pipeline...")

MM_PIPELINE = modelscope_pipeline(
    task=Tasks.multi_modal_embedding,
    model=MODEL_ID,
    model_revision=MODEL_REVISION
)

logger.info("ModelScope pipeline 初始化完成。")


def normalize_column_name(col: Any) -> str:
    return str(col).strip().lower()


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True
) -> Optional[str]:
    normalized_map = {
        normalize_column_name(col): col
        for col in df.columns
    }

    candidate_set = [normalize_column_name(c) for c in candidates]

    for candidate in candidate_set:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for normalized_col, original_col in normalized_map.items():
        for candidate in candidate_set:
            if candidate in normalized_col:
                return original_col

    if required:
        raise KeyError(
            f"未找到必要列。候选列名为: {candidates}; "
            f"当前 xlsx 实际列名为: {list(df.columns)}"
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


def parse_label(value: Any) -> int:
    if is_nan_like(value):
        raise ValueError("label 为空，无法解析。")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return int(value)

    value_str = str(value).strip()

    if value_str == "":
        raise ValueError("label 为空字符串，无法解析。")

    lower_value = value_str.lower()

    if lower_value in {"0", "0.0", "false", "real", "nonrumor", "non-rumor"}:
        return 0

    if lower_value in {"1", "1.0", "true", "fake", "rumor"}:
        return 1

    try:
        return int(float(value_str))
    except Exception as exc:
        raise ValueError(f"无法将 label={value!r} 解析为 int。") from exc


def clean_image_path(path_text: str) -> str:
    path_text = str(path_text).strip()
    path_text = path_text.strip("[](){}'\"")
    path_text = path_text.replace("\\", "/")
    return path_text


IMAGE_PATH_PATTERN = re.compile(
    r"(?:(?:\.{1,2}[\\/])?)*(?:nonrumor_images|rumor_images)[\\/][^\s\]\[;,|'\"]+?\.(?:jpg|jpeg|png|bmp|webp)",
    re.IGNORECASE
)

FALLBACK_IMAGE_FILENAME_PATTERN = re.compile(
    r"[^\s\]\[;,|'\"]+?\.(?:jpg|jpeg|png|bmp|webp)",
    re.IGNORECASE
)


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []

    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)

    return output


def extract_image_paths_from_cell(cell_value: Any) -> List[str]:
    text = safe_str(cell_value)

    if text == "":
        return []

    matched_paths = IMAGE_PATH_PATTERN.findall(text)

    if not matched_paths:
        matched_paths = FALLBACK_IMAGE_FILENAME_PATTERN.findall(text)

    cleaned_paths = [
        clean_image_path(path)
        for path in matched_paths
        if clean_image_path(path) != ""
    ]

    return unique_keep_order(cleaned_paths)


def extract_image_paths_from_row(
    row: pd.Series,
    image_col: Optional[str]
) -> List[str]:
    image_paths: List[str] = []

    if image_col is not None:
        image_paths.extend(extract_image_paths_from_cell(row.get(image_col, "")))

    if len(image_paths) == 0:
        for value in row.values:
            image_paths.extend(extract_image_paths_from_cell(value))

    return unique_keep_order(image_paths)


def resolve_image_path(raw_path: str, image_root_dir: str) -> Path:
    raw_path = clean_image_path(raw_path)
    image_root = Path(image_root_dir)

    path_obj = Path(raw_path)

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

    unique_candidates: List[Path] = []
    seen = set()

    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate_key)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return unique_candidates[0]


def assert_basic_paths_exist(
    train_excel_path: str,
    test_excel_path: str,
    image_root_dir: str
) -> None:
    train_excel = Path(train_excel_path)
    test_excel = Path(test_excel_path)
    image_root = Path(image_root_dir)

    if not train_excel.exists():
        raise FileNotFoundError(f"训练集 xlsx 不存在: {train_excel}")

    if not test_excel.exists():
        raise FileNotFoundError(f"测试集 xlsx 不存在: {test_excel}")

    if not image_root.exists():
        raise FileNotFoundError(f"图片根目录不存在: {image_root}")

    rumor_dir = image_root / "rumor_images"
    nonrumor_dir = image_root / "nonrumor_images"

    if not rumor_dir.exists():
        logger.warning(f"未发现 rumor_images 文件夹: {rumor_dir}")

    if not nonrumor_dir.exists():
        logger.warning(f"未发现 nonrumor_images 文件夹: {nonrumor_dir}")


def create_blank_image() -> Image.Image:
    return Image.new(
        mode="RGB",
        size=(BLANK_IMAGE_SIZE, BLANK_IMAGE_SIZE),
        color=BLANK_IMAGE_COLOR
    )


def load_image_as_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        image = img.convert("RGB").copy()

    return image


def to_numpy_vector(embedding: Any) -> np.ndarray:
    if torch.is_tensor(embedding):
        vector = embedding.detach().cpu().numpy()
    else:
        vector = np.asarray(embedding)

    vector = np.squeeze(vector).astype(np.float32)

    if vector.ndim != 1:
        raise ValueError(f"embedding 维度异常，期望一维向量，实际 shape={vector.shape}")

    return vector


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm < eps:
        return vector

    return vector / norm


# =========================
# 5. 数据加载函数
# =========================

def load_weibo21_xlsx_data(
    excel_path: str,
    image_root_dir: str
) -> List[Dict[str, Any]]:
    logger.info(f"正在读取 xlsx: {excel_path}")

    df = pd.read_excel(excel_path)
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"xlsx 文件为空: {excel_path}")

    label_col = find_column(
        df,
        candidates=["label", "labels", "target", "class", "类别", "标签"],
        required=True
    )

    content_col = find_column(
        df,
        candidates=["content", "text", "正文", "文本", "微博内容"],
        required=True
    )

    image_col = find_column(
        df,
        candidates=["image", "images", "img", "picture", "pic", "图片", "图像"],
        required=False
    )

    title_col = find_column(
        df,
        candidates=["title", "标题"],
        required=False
    )

    source_col = find_column(
        df,
        candidates=["source", "来源"],
        required=False
    )

    id_col = find_column(
        df,
        candidates=["id", "idx", "index", "编号", "序号", "unnamed: 0"],
        required=False
    )

    logger.info(
        "列匹配结果: "
        f"id_col={id_col}, "
        f"title_col={title_col}, "
        f"label_col={label_col}, "
        f"source_col={source_col}, "
        f"content_col={content_col}, "
        f"image_col={image_col}"
    )

    data_list: List[Dict[str, Any]] = []

    empty_text_count = 0
    empty_image_path_count = 0
    missing_image_file_count = 0
    blank_image_sample_count = 0

    for row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"加载数据 {Path(excel_path).name}"
    ):
        if id_col is not None:
            raw_id = row.get(id_col, row_index)
            try:
                sample_id = int(float(raw_id))
            except Exception:
                sample_id = int(row_index)
        else:
            sample_id = int(row_index)

        text = safe_str(row.get(content_col, ""))

        if text == "" and title_col is not None:
            text = safe_str(row.get(title_col, ""))

        if text == "":
            empty_text_count += 1

        label = parse_label(row.get(label_col))

        title = safe_str(row.get(title_col, "")) if title_col is not None else ""
        source = safe_str(row.get(source_col, "")) if source_col is not None else ""

        raw_image_paths = extract_image_paths_from_row(row, image_col)

        if len(raw_image_paths) == 0:
            empty_image_path_count += 1

        resolved_existing_paths: List[str] = []
        missing_paths: List[str] = []

        for raw_image_path in raw_image_paths:
            resolved_path = resolve_image_path(raw_image_path, image_root_dir)

            if resolved_path.exists() and resolved_path.is_file():
                suffix = resolved_path.suffix.lower()

                if suffix in SUPPORTED_IMAGE_EXTENSIONS:
                    resolved_existing_paths.append(str(resolved_path))
                else:
                    logger.warning(
                        f"样本 id={sample_id} 的图片后缀不受支持: {resolved_path}"
                    )
                    missing_paths.append(str(resolved_path))
                    missing_image_file_count += 1
            else:
                missing_paths.append(str(resolved_path))
                missing_image_file_count += 1

        resolved_existing_paths = unique_keep_order(resolved_existing_paths)
        missing_paths = unique_keep_order(missing_paths)

        use_blank_image = len(resolved_existing_paths) == 0

        if use_blank_image:
            blank_image_sample_count += 1

        sample = {
            "id": int(sample_id),
            "text": text,
            "label": int(label),
            "title": title,
            "source": source,
            "raw_image_paths": raw_image_paths,
            "image_paths": resolved_existing_paths,
            "missing_image_paths": missing_paths,
            "use_blank_image": bool(use_blank_image)
        }

        data_list.append(sample)

    logger.info(f"数据读取完成: {excel_path}")
    logger.info(f"样本总数: {len(data_list)}")
    logger.info(f"空文本样本数: {empty_text_count}")
    logger.info(f"未解析到图片路径的样本数: {empty_image_path_count}")
    logger.info(f"缺失或无效图片文件计数: {missing_image_file_count}")
    logger.info(f"需要使用空白图片代替的样本数: {blank_image_sample_count}")

    valid_text_count = sum(
        1
        for item in data_list
        if item["text"] != ""
    )

    logger.info(f"具有有效文本的样本数: {valid_text_count}")

    if valid_text_count == 0:
        raise ValueError(
            f"{excel_path} 中没有任何有效文本样本，无法进行文本特征提取。"
        )

    return data_list


def extract_text_embedding(text: str) -> np.ndarray:
    with torch.no_grad():
        output = MM_PIPELINE.forward({"text": [text]})

    if "text_embedding" not in output:
        raise KeyError(
            f"ModelScope 文本输出中不存在 text_embedding，实际 keys={list(output.keys())}"
        )

    text_embedding = to_numpy_vector(output["text_embedding"])

    if NORMALIZE_EMBEDDINGS:
        text_embedding = l2_normalize(text_embedding)

    return text_embedding.astype(np.float32)


def extract_image_embedding_from_pil(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    with torch.no_grad():
        output = MM_PIPELINE.forward({"img": image})

    if "img_embedding" not in output:
        raise KeyError(
            f"ModelScope 图像输出中不存在 img_embedding，实际 keys={list(output.keys())}"
        )

    image_embedding = to_numpy_vector(output["img_embedding"])

    if NORMALIZE_EMBEDDINGS:
        image_embedding = l2_normalize(image_embedding)

    return image_embedding.astype(np.float32)


def extract_single_image_embedding(image_path: str) -> np.ndarray:
    image = load_image_as_rgb(image_path)
    return extract_image_embedding_from_pil(image)


def extract_blank_image_embedding() -> np.ndarray:
    blank_image = create_blank_image()
    return extract_image_embedding_from_pil(blank_image)


def extract_multi_image_embedding_with_blank_fallback(
    image_paths: List[str],
    sample_id: int
) -> Tuple[np.ndarray, bool]:
    image_embeddings: List[np.ndarray] = []

    if not isinstance(image_paths, list) or len(image_paths) == 0:
        logger.warning(f"样本 id={sample_id} 没有有效图片路径，使用空白图片代替。")
        blank_embedding = extract_blank_image_embedding()
        return blank_embedding, True

    for image_path in image_paths:
        try:
            image_embedding = extract_single_image_embedding(image_path)
            image_embeddings.append(image_embedding)
        except Exception as exc:
            logger.warning(
                f"样本 id={sample_id} 的图片特征提取失败: {image_path}; "
                f"错误: {exc}"
            )

    if len(image_embeddings) == 0:
        logger.warning(
            f"样本 id={sample_id} 所有图片均提取失败，使用空白图片代替。"
        )
        blank_embedding = extract_blank_image_embedding()
        return blank_embedding, True

    stacked = np.stack(image_embeddings, axis=0)
    mean_embedding = np.mean(stacked, axis=0).astype(np.float32)

    if NORMALIZE_EMBEDDINGS:
        mean_embedding = l2_normalize(mean_embedding).astype(np.float32)

    return mean_embedding, False


def get_embeddings(
    data_list: List[Dict[str, Any]],
    desc: str = ""
) -> List[Optional[Tuple[np.ndarray, np.ndarray, bool]]]:
    embeddings: List[Optional[Tuple[np.ndarray, np.ndarray, bool]]] = []

    blank_image_used_count = 0
    skipped_text_count = 0

    for item in tqdm(data_list, desc=desc):
        sample_id = int(item["id"])
        text = safe_str(item.get("text", ""))
        image_paths = item.get("image_paths", [])

        if text == "":
            logger.warning(f"样本 id={sample_id} 文本为空，跳过该样本。")
            embeddings.append(None)
            skipped_text_count += 1
            continue

        try:
            image_embedding, used_blank_image = extract_multi_image_embedding_with_blank_fallback(
                image_paths=image_paths,
                sample_id=sample_id
            )

            if used_blank_image:
                blank_image_used_count += 1

            text_embedding = extract_text_embedding(text)

            if image_embedding.shape[0] != text_embedding.shape[0]:
                raise ValueError(
                    f"样本 id={sample_id} 图文 embedding 维度不一致: "
                    f"image_dim={image_embedding.shape[0]}, "
                    f"text_dim={text_embedding.shape[0]}"
                )

            embeddings.append((image_embedding, text_embedding, used_blank_image))

        except Exception as exc:
            logger.warning(f"样本 id={sample_id} embedding 提取失败，跳过。错误: {exc}")
            embeddings.append(None)

    logger.info(f"{desc} 中使用空白图片的样本数: {blank_image_used_count}")
    logger.info(f"{desc} 中文本为空而被跳过的样本数: {skipped_text_count}")

    return embeddings




def build_embedding_matrices(
    embeddings_list: List[Optional[Tuple[np.ndarray, np.ndarray, bool]]]
) -> Tuple[List[int], np.ndarray, np.ndarray, List[bool]]:
    valid_indices = [
        i
        for i, emb in enumerate(embeddings_list)
        if emb is not None
    ]

    if len(valid_indices) == 0:
        raise ValueError("没有任何有效 embedding，无法构造矩阵。")

    image_embeddings = np.vstack([
        embeddings_list[i][0]
        for i in valid_indices
    ]).astype(np.float32)

    text_embeddings = np.vstack([
        embeddings_list[i][1]
        for i in valid_indices
    ]).astype(np.float32)

    used_blank_image_flags = [
        bool(embeddings_list[i][2])
        for i in valid_indices
    ]

    return valid_indices, image_embeddings, text_embeddings, used_blank_image_flags


def calculate_and_save_similarity(
    query_data: List[Dict[str, Any]],
    ref_data: List[Dict[str, Any]],
    output_filename: str,
    k: int = 10
) -> None:
    logger.info(f"开始处理输出文件: {output_filename}")

    if k <= 0:
        raise ValueError(f"k 必须为正整数，当前 k={k}")

    logger.info("正在为查询集生成特征嵌入...")
    query_embeddings_list = get_embeddings(query_data, desc="查询集嵌入")

    logger.info("正在为参考集生成特征嵌入...")
    ref_embeddings_list = get_embeddings(ref_data, desc="参考集嵌入")

    (
        valid_query_indices,
        query_image_embeddings,
        query_text_embeddings,
        query_blank_flags
    ) = build_embedding_matrices(query_embeddings_list)

    (
        valid_ref_indices,
        ref_image_embeddings,
        ref_text_embeddings,
        ref_blank_flags
    ) = build_embedding_matrices(ref_embeddings_list)

    logger.info(f"有效查询样本数: {len(valid_query_indices)}")
    logger.info(f"有效参考样本数: {len(valid_ref_indices)}")
    logger.info(f"查询集中使用空白图片的有效样本数: {sum(query_blank_flags)}")
    logger.info(f"参考集中使用空白图片的有效样本数: {sum(ref_blank_flags)}")

    logger.info("正在独立计算各模态相似度分数...")
    image_similarity_scores = np.dot(query_image_embeddings, ref_image_embeddings.T)
    text_similarity_scores = np.dot(query_text_embeddings, ref_text_embeddings.T)

    logger.info("正在融合图文相似度分数...")
    final_similarity_scores = (
        WEIGHT_IMAGE * image_similarity_scores
        + WEIGHT_TEXT * text_similarity_scores
    )

    is_self_comparison = query_data is ref_data

    ref_original_index_to_valid_position = {
        original_index: valid_position
        for valid_position, original_index in enumerate(valid_ref_indices)
    }

    results: List[Dict[str, Any]] = []

    logger.info(f"正在提取 Top-{k} 相似样本...")

    for valid_query_position in tqdm(
        range(len(valid_query_indices)),
        desc="Top-K 提取"
    ):
        original_query_index = valid_query_indices[valid_query_position]
        query_item = query_data[original_query_index]

        current_scores = final_similarity_scores[valid_query_position].copy()

        if is_self_comparison:
            self_position_in_ref = ref_original_index_to_valid_position.get(
                original_query_index,
                None
            )

            if self_position_in_ref is not None:
                current_scores[self_position_in_ref] = -np.inf

        finite_mask = np.isfinite(current_scores)
        finite_count = int(np.sum(finite_mask))

        if finite_count == 0:
            top_ref_infos: List[Dict[str, Any]] = []
        else:
            k_to_fetch = min(k, finite_count)

            candidate_indices = np.where(finite_mask)[0]
            candidate_scores = current_scores[candidate_indices]

            if k_to_fetch == len(candidate_indices):
                top_positions_in_candidate = np.argsort(candidate_scores)[::-1]
            else:
                unordered_top_positions = np.argpartition(
                    candidate_scores,
                    -k_to_fetch
                )[-k_to_fetch:]

                top_positions_in_candidate = unordered_top_positions[
                    np.argsort(candidate_scores[unordered_top_positions])[::-1]
                ]

            top_ref_valid_positions = candidate_indices[top_positions_in_candidate]
            top_scores = current_scores[top_ref_valid_positions]

            top_ref_infos = []

            for ref_valid_position, score_value in zip(top_ref_valid_positions, top_scores):
                original_ref_index = valid_ref_indices[int(ref_valid_position)]
                ref_item = ref_data[original_ref_index]

                ref_used_blank_image = bool(ref_blank_flags[int(ref_valid_position)])

                top_ref_infos.append({
                    "index": int(original_ref_index),
                    "id": int(ref_item["id"]),
                    "label": int(ref_item["label"]),
                    "score": float(score_value),
                    "used_blank_image": ref_used_blank_image
                })

        query_used_blank_image = bool(query_blank_flags[int(valid_query_position)])

        result_item = {
            "query_sample": {
                "index": int(original_query_index),
                "id": int(query_item["id"]),
                "used_blank_image": query_used_blank_image
            },
            "similar_ref_samples": top_ref_infos
        }

        results.append(result_item)

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result_item in results:
            json.dump(result_item, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"结果已成功保存至: {output_path}")




def main() -> None:
    assert_basic_paths_exist(
        train_excel_path=TRAIN_EXCEL_PATH,
        test_excel_path=TEST_EXCEL_PATH,
        image_root_dir=IMAGE_ROOT_DIR
    )

    logger.info("正在加载训练集...")
    train_data = load_weibo21_xlsx_data(
        excel_path=TRAIN_EXCEL_PATH,
        image_root_dir=IMAGE_ROOT_DIR
    )

    logger.info("正在加载测试集...")
    test_data = load_weibo21_xlsx_data(
        excel_path=TEST_EXCEL_PATH,
        image_root_dir=IMAGE_ROOT_DIR
    )

    output_base_dir = Path(OUTPUT_BASE_DIR)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    output_file_train_vs_train = output_base_dir / "weibo21_CLIP_train_vs_train.jsonl"
    output_file_test_vs_train = output_base_dir / "weibo21_CLIP_test_vs_train.jsonl"

    calculate_and_save_similarity(
        query_data=train_data,
        ref_data=train_data,
        output_filename=str(output_file_train_vs_train),
        k=K_VALUE
    )

    calculate_and_save_similarity(
        query_data=test_data,
        ref_data=train_data,
        output_filename=str(output_file_test_vs_train),
        k=K_VALUE
    )

    logger.info("所有检索任务已完成。")


if __name__ == "__main__":
    main()
