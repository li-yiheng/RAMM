import os
import json
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from modelscope import AutoModel, AutoTokenizer





TRAIN_JSON = "train_datas_narrative_category.json"


TEST_JSON = "train_datas_narrative_category.json"


OUTPUT_JSON = "train_to_train_retrieval_bge.json"


MODEL_NAME_OR_PATH = "./BAAI/bge-base-zh-v1.5"


K_IN = 6
K_OUT = 6


BATCH_SIZE = 64
MAX_LENGTH = 256


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


ID_FIELD = "id"
NARRATIVE_FIELD = "core_narrative"
CATEGORY_FIELD = "category"


EXCLUDE_SAME_ID = False


USE_QUERY_INSTRUCTION = False
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："



def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 list[dict] 格式。")
    return data


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = " ".join(text.split())
    return text


def validate_dataset(data: List[Dict[str, Any]], dataset_name: str) -> None:
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_name} 第 {idx} 条不是 dict。")
        if ID_FIELD not in item:
            raise KeyError(f"{dataset_name} 第 {idx} 条缺少字段：{ID_FIELD}")
        if NARRATIVE_FIELD not in item:
            raise KeyError(f"{dataset_name} 第 {idx} 条缺少字段：{NARRATIVE_FIELD}")
        if CATEGORY_FIELD not in item:
            raise KeyError(f"{dataset_name} 第 {idx} 条缺少字段：{CATEGORY_FIELD}")




def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class BGEEmbedder:
    def __init__(self, model_name_or_path: str, device: str = "cpu", max_length: int = 256):
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        add_instruction: bool = False
    ) -> torch.Tensor:
        processed_texts = []
        for text in texts:
            text = clean_text(text)
            if add_instruction:
                text = f"{QUERY_INSTRUCTION}{text}"
            processed_texts.append(text)

        all_embeddings = []

        for start in tqdm(range(0, len(processed_texts), batch_size), desc="Encoding with BGE"):
            batch_texts = processed_texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, return_dict=True)

            if not hasattr(outputs, "last_hidden_state"):
                raise RuntimeError("模型输出中未找到 last_hidden_state，无法做 mean pooling。")

            embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)




def build_cross_similarity_matrix(
    query_embeddings: torch.Tensor,
    ref_embeddings: torch.Tensor
) -> torch.Tensor:
    return query_embeddings @ ref_embeddings.T


def build_test_to_train_retrieval_results(
    test_data: List[Dict[str, Any]],
    train_data: List[Dict[str, Any]],
    test_embeddings: torch.Tensor,
    train_embeddings: torch.Tensor,
    k_in: int = 3,
    k_out: int = 2,
    exclude_same_id: bool = True
) -> List[Dict[str, Any]]:
    test_ids = [str(item.get(ID_FIELD, "")) for item in test_data]
    test_categories = [clean_text(item.get(CATEGORY_FIELD, "")) for item in test_data]

    train_ids = [str(item.get(ID_FIELD, "")) for item in train_data]
    train_categories = [clean_text(item.get(CATEGORY_FIELD, "")) for item in train_data]

    sim_matrix = build_cross_similarity_matrix(test_embeddings, train_embeddings)

    results = []

    for i in tqdm(range(len(test_data)), desc="Retrieving test -> train top-k"):
        query_id = test_ids[i]
        query_category = test_categories[i]
        sims = sim_matrix[i]

        in_domain_candidates = []
        out_domain_candidates = []

        for j in range(len(train_data)):
            ref_id = train_ids[j]
            ref_category = train_categories[j]

            if exclude_same_id and query_id == ref_id:
                continue

            score = float(sims[j].item())

            ref_item = {
                "index": j,
                "id": ref_id,
                "category": ref_category,
                "score": round(score, 10)
            }

            if ref_category == query_category:
                in_domain_candidates.append(ref_item)
            else:
                out_domain_candidates.append(ref_item)

        in_domain_candidates.sort(key=lambda x: x["score"], reverse=True)
        out_domain_candidates.sort(key=lambda x: x["score"], reverse=True)

        in_topk = in_domain_candidates[:k_in]
        out_topk = out_domain_candidates[:k_out]

        result = {
            "query_sample": {
                "index": i,
                "id": query_id,
                "category": query_category,
                "split": "test"
            },
            "in_domain_ref_samples": in_topk,
            "out_domain_ref_samples": out_topk,
            "similar_ref_samples": in_topk + out_topk
        }

        results.append(result)

    return results




def main():
    print(f"Loading training data from: {TRAIN_JSON}")
    train_data = load_json(TRAIN_JSON)
    validate_dataset(train_data, "训练集")

    print(f"Loading test data from: {TEST_JSON}")
    test_data = load_json(TEST_JSON)
    validate_dataset(test_data, "测试集")

    train_narratives = [clean_text(item.get(NARRATIVE_FIELD, "")) for item in train_data]
    test_narratives = [clean_text(item.get(NARRATIVE_FIELD, "")) for item in test_data]

    print(f"Loading BGE model from: {MODEL_NAME_OR_PATH}")
    print(f"Device: {DEVICE}")
    print(f"K_IN = {K_IN}, K_OUT = {K_OUT}")
    print(f"EXCLUDE_SAME_ID = {EXCLUDE_SAME_ID}")

    embedder = BGEEmbedder(
        model_name_or_path=MODEL_NAME_OR_PATH,
        device=DEVICE,
        max_length=MAX_LENGTH
    )

    print("Encoding training narratives...")
    train_embeddings = embedder.encode(
        texts=train_narratives,
        batch_size=BATCH_SIZE,
        add_instruction=USE_QUERY_INSTRUCTION
    )

    print("Encoding test narratives...")
    test_embeddings = embedder.encode(
        texts=test_narratives,
        batch_size=BATCH_SIZE,
        add_instruction=USE_QUERY_INSTRUCTION
    )

    print("Building retrieval results for test -> train ...")
    results = build_test_to_train_retrieval_results(
        test_data=test_data,
        train_data=train_data,
        test_embeddings=test_embeddings,
        train_embeddings=train_embeddings,
        k_in=K_IN,
        k_out=K_OUT,
        exclude_same_id=EXCLUDE_SAME_ID
    )

    save_json(OUTPUT_JSON, results)
    print(f"Done. Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
