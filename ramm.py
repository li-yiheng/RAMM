#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAMM: Retrieval-Augmented Multimodal Model for Fake News Detection
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2Model,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data



def load_json_or_jsonl(path: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)

    # fallback: try json first, then jsonl
    try:
        return load_json(path)
    except Exception:
        return load_jsonl(path)



def resolve_text(record: Dict[str, Any]) -> str:
    for key in ["text", "content", "news", "article", "claim", "sentence"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return ""



def resolve_image(record: Dict[str, Any]) -> str:
    for key in ["image", "image_path", "img", "img_path", "picture", "picture_path"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return ""



def resolve_label(record: Dict[str, Any]) -> int:
    for key in ["label", "y", "target", "veracity"]:
        if key in record and record[key] is not None:
            value = record[key]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, str):
                value_strip = value.strip().lower()
                if value_strip in {"0", "real", "true", "genuine"}:
                    return 0
                if value_strip in {"1", "fake", "false"}:
                    return 1
                return int(value_strip)
    raise KeyError(f"Cannot resolve label from record keys: {list(record.keys())}")



def resolve_id(record: Dict[str, Any], default_index: int) -> str:
    for key in ["id", "news_id", "item_id", "mid", "weibo_id"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return str(default_index)



def resolve_domain(record: Dict[str, Any]) -> str:
    for key in ["domain", "category", "topic", "class"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return "unknown"



def resolve_narrative_text(record: Dict[str, Any]) -> str:
    for key in ["abstract_narrative", "narrative", "summary", "ana_text"]:
        if key in record and record[key] is not None:
            return str(record[key])
    # fallback to raw text when no explicit narrative text is stored
    return resolve_text(record)



def safe_image_open(image_base_dir: str, relative_or_abs_path: str, image_size: int = 224) -> Image.Image:
    if relative_or_abs_path is None:
        relative_or_abs_path = ""
    if os.path.isabs(relative_or_abs_path):
        full_path = relative_or_abs_path
    else:
        full_path = os.path.join(image_base_dir, relative_or_abs_path)
    try:
        return Image.open(full_path).convert("RGB")
    except Exception:
        # strict but robust fallback: blank image
        return Image.new("RGB", (image_size, image_size), color=(255, 255, 255))



def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = last_hidden_state * mask
    denom = torch.clamp(mask.sum(dim=1), min=1e-6)
    return masked.sum(dim=1) / denom



def compute_classification_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probs).astype(float)
    preds_np = (probs_np >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
    }
    if len(np.unique(labels_np)) > 1:
        metrics["auc"] = float(roc_auc_score(labels_np, probs_np))
    else:
        metrics["auc"] = float("nan")
    return metrics




@dataclass
class NewsItem:
    index: int
    item_id: str
    text: str
    image: str
    label: int
    domain: str
    narrative_text: str


@dataclass
class RAMMExample:
    query_index: int
    label: int
    query_text: str
    query_image: str
    demo_index: int
    demo_text: str
    demo_image: str
    demo_label: int
    ana_candidate_indices: List[int]
    query_narrative_embedding: torch.Tensor
    candidate_narrative_embeddings: torch.Tensor


@dataclass
class ModelArguments:
    text_model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    vision_model_name: str = "Salesforce/blip2-opt-2.7b"
    narrative_encoder_name: str = "BAAI/bge-large-zh-v1.5"
    use_4bit: bool = False
    torch_dtype: str = "bfloat16"
    freeze_vision_encoder: bool = True
    freeze_qformer: bool = False
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.10
    image_token: str = "<image>"
    max_prompt_length: int = 768
    max_single_text_length: int = 256
    max_narrative_length: int = 256
    hidden_dropout: float = 0.10
    latent_dim: int = 768
    temperature: float = 0.07
    lambda_align: float = 0.2
    lambda_recon: float = 0.1
    lambda_compress: float = 0.2
    language: str = "zh"


@dataclass
class DataArguments:
    train_data_path: str = ""
    test_data_path: str = ""
    sra_train_retrieval_path: str = ""
    sra_test_retrieval_path: str = ""
    ana_train_retrieval_path: str = ""
    ana_test_retrieval_path: str = ""
    image_base_dir: str = "."
    kin: int = 3
    kout: int = 2


@dataclass
class TrainArguments:
    output_dir: str = "./outputs/ramm"
    seed: int = 3407
    epochs: int = 3
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    num_workers: int = 2
    log_steps: int = 20
    eval_steps: int = 200
    save_best_only: bool = True
    monitor_metric: str = "f1"
    fp32_narrative_encoder: bool = True





class RetrievalParser:
    @staticmethod
    def parse_sra_records(path: str) -> Dict[int, List[int]]:
        records = load_json_or_jsonl(path)
        if not isinstance(records, list):
            raise ValueError(f"SRA retrieval file must decode to a list, got {type(records)}")

        result: Dict[int, List[int]] = {}
        for item in records:
            query_sample = item.get("query_sample", {})
            q_idx = int(query_sample.get("index", -1))
            if q_idx < 0:
                continue
            ref_list = item.get("similar_ref_samples", [])
            cand = []
            for ref in ref_list:
                idx = ref.get("index", None)
                if idx is None:
                    continue
                cand.append(int(idx))
            if cand:
                result[q_idx] = cand
        return result

    @staticmethod
    def parse_ana_records(path: str, kin: int, kout: int) -> Dict[int, List[int]]:
        records = load_json_or_jsonl(path)
        if isinstance(records, dict):
            # robustness: some exporters may save under a root key
            for key in ["data", "records", "results", "items"]:
                if key in records and isinstance(records[key], list):
                    records = records[key]
                    break
        if not isinstance(records, list):
            raise ValueError(f"ANA retrieval file must decode to a list, got {type(records)}")

        result: Dict[int, List[int]] = {}
        for item in records:
            query_sample = item.get("query_sample", {})
            q_idx = int(query_sample.get("index", -1))
            if q_idx < 0:
                continue

            candidate_indices: List[int] = []
            in_domain = item.get("in_domain_ref_samples", [])
            out_domain = item.get("out_domain_ref_samples", [])

            if in_domain or out_domain:
                for ref in in_domain[:kin]:
                    if "index" in ref:
                        candidate_indices.append(int(ref["index"]))
                for ref in out_domain[:kout]:
                    if "index" in ref:
                        candidate_indices.append(int(ref["index"]))
            else:
                # fallback to pre-unified field
                merged = item.get("similar_ref_samples", [])
                for ref in merged[: kin + kout]:
                    if "index" in ref:
                        candidate_indices.append(int(ref["index"]))

            # remove duplicates while keeping order
            seen = set()
            deduped = []
            for idx in candidate_indices:
                if idx not in seen:
                    deduped.append(idx)
                    seen.add(idx)
            if deduped:
                result[q_idx] = deduped
        return result





class NarrativeTextEncoder:
    """
    Fixed sentence embedding encoder used only to produce narrative vectors for
    the ANA attention module. If the dataset already contains explicit narrative
    texts, those are used; otherwise raw article text is used as a faithful
    fallback because the uploaded files do not include saved narrative summaries.
    """

    def __init__(self, model_name: str, max_length: int = 256, device: str = "cuda", fp32: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(self.device)
        if not fp32:
            self.model.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> torch.Tensor:
        all_embs: List[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=math.ceil(len(texts) / batch_size), desc="Encoding narrative texts")
        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = masked_mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            emb = F.normalize(emb.float(), dim=-1)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)





class RAMMDataset(Dataset):
    def __init__(
        self,
        query_items: List[NewsItem],
        train_reference_items: List[NewsItem],
        sra_map: Dict[int, List[int]],
        ana_map: Dict[int, List[int]],
        query_narrative_cache: torch.Tensor,
        train_narrative_cache: torch.Tensor,
    ) -> None:
        self.query_items = query_items
        self.train_reference_items = train_reference_items
        self.sra_map = sra_map
        self.ana_map = ana_map
        self.query_narrative_cache = query_narrative_cache
        self.train_narrative_cache = train_narrative_cache

        self.valid_query_indices = list(range(len(query_items)))

    def __len__(self) -> int:
        return len(self.valid_query_indices)

    def __getitem__(self, idx: int) -> RAMMExample:
        q_idx = self.valid_query_indices[idx]
        item = self.query_items[q_idx]

        sra_candidates = self.sra_map.get(q_idx, [])
        if len(sra_candidates) == 0:
            # robust fallback: use the first training item
            demo_idx = 0
        else:
            demo_idx = int(sra_candidates[0])
        demo_item = self.train_reference_items[demo_idx]

        ana_candidates = self.ana_map.get(q_idx, [])
        ana_candidates = [c for c in ana_candidates if 0 <= c < len(self.train_reference_items)]
        if len(ana_candidates) == 0:
            # robust fallback to the SRA demo so that the ANA branch always has at least one candidate
            ana_candidates = [demo_idx]

        candidate_narrative_embeddings = self.train_narrative_cache[ana_candidates].clone().float()
        query_narrative_embedding = self.query_narrative_cache[q_idx].clone().float()

        return RAMMExample(
            query_index=q_idx,
            label=item.label,
            query_text=item.text,
            query_image=item.image,
            demo_index=demo_idx,
            demo_text=demo_item.text,
            demo_image=demo_item.image,
            demo_label=demo_item.label,
            ana_candidate_indices=ana_candidates,
            query_narrative_embedding=query_narrative_embedding,
            candidate_narrative_embeddings=candidate_narrative_embeddings,
        )


class RAMMCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        image_processor: AutoImageProcessor,
        train_reference_items: List[NewsItem],
        image_base_dir: str,
        image_token: str,
        language: str,
        max_prompt_length: int,
        max_single_text_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.train_reference_items = train_reference_items
        self.image_base_dir = image_base_dir
        self.image_token = image_token
        self.language = language
        self.max_prompt_length = max_prompt_length
        self.max_single_text_length = max_single_text_length

    def _make_prompt(self, demo_text: str, demo_label: int, query_text: str) -> str:
        if self.language.lower().startswith("zh"):
            return (
                "这是一条新闻示例：\n"
                f"新闻内容：\"{demo_text}\"，对应图片为 {self.image_token}，标签为 {demo_label}。\n"
                "现在给你一条新的新闻，请参考上面的示例进行类比推断。\n"
                f"待判断新闻内容：\"{query_text}\"，对应图片为 {self.image_token}。\n"
                "请仅输出 0 或 1，其中 0 表示真实，1 表示虚假。"
            )
        return (
            "Here is a demonstration news item.\n"
            f"News text: \"{demo_text}\". Its image is {self.image_token}. The label is {demo_label}.\n"
            "Now determine the label of the following news by analogy.\n"
            f"Target news text: \"{query_text}\". Its image is {self.image_token}.\n"
            "Output only 0 or 1, where 0 means real and 1 means fake."
        )

    def _process_image(self, image_path: str) -> torch.Tensor:
        image = safe_image_open(self.image_base_dir, image_path)
        processed = self.image_processor(images=image, return_tensors="pt")
        return processed["pixel_values"].squeeze(0)

    def __call__(self, features: List[RAMMExample]) -> Dict[str, Any]:
        prompt_texts: List[str] = []
        demo_images: List[torch.Tensor] = []
        query_images: List[torch.Tensor] = []
        single_query_texts: List[str] = []
        single_query_images: List[torch.Tensor] = []
        labels: List[int] = []
        query_narr_embeddings: List[torch.Tensor] = []
        cand_narr_list: List[torch.Tensor] = []
        cand_counts: List[int] = []
        candidate_texts_flat: List[str] = []
        candidate_images_flat: List[torch.Tensor] = []

        for ex in features:
            prompt_texts.append(self._make_prompt(ex.demo_text, ex.demo_label, ex.query_text))
            demo_images.append(self._process_image(ex.demo_image))
            query_images.append(self._process_image(ex.query_image))
            single_query_texts.append(ex.query_text)
            single_query_images.append(self._process_image(ex.query_image))
            labels.append(int(ex.label))
            query_narr_embeddings.append(ex.query_narrative_embedding.float())
            cand_narr_list.append(ex.candidate_narrative_embeddings.float())
            cand_counts.append(len(ex.ana_candidate_indices))

            for cand_idx in ex.ana_candidate_indices:
                cand_item = self.train_reference_items[cand_idx]
                candidate_texts_flat.append(cand_item.text)
                candidate_images_flat.append(self._process_image(cand_item.image))

        prompt_batch = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        single_query_batch = self.tokenizer(
            single_query_texts,
            padding=True,
            truncation=True,
            max_length=self.max_single_text_length,
            return_tensors="pt",
        )
        candidate_batch = self.tokenizer(
            candidate_texts_flat,
            padding=True,
            truncation=True,
            max_length=self.max_single_text_length,
            return_tensors="pt",
        ) if len(candidate_texts_flat) > 0 else None

        max_cands = max(cand_counts) if cand_counts else 1
        narrative_dim = query_narr_embeddings[0].numel() if query_narr_embeddings else 1024
        padded_cand_narr = torch.zeros(len(features), max_cands, narrative_dim, dtype=torch.float32)
        candidate_mask = torch.zeros(len(features), max_cands, dtype=torch.bool)
        for i, cand_tensor in enumerate(cand_narr_list):
            cur_n = cand_tensor.size(0)
            padded_cand_narr[i, :cur_n] = cand_tensor
            candidate_mask[i, :cur_n] = True

        batch = {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "prompt_demo_pixel_values": torch.stack(demo_images, dim=0),
            "prompt_query_pixel_values": torch.stack(query_images, dim=0),
            "query_input_ids": single_query_batch["input_ids"],
            "query_attention_mask": single_query_batch["attention_mask"],
            "query_pixel_values": torch.stack(single_query_images, dim=0),
            "candidate_input_ids": candidate_batch["input_ids"] if candidate_batch is not None else None,
            "candidate_attention_mask": candidate_batch["attention_mask"] if candidate_batch is not None else None,
            "candidate_pixel_values": torch.stack(candidate_images_flat, dim=0) if len(candidate_images_flat) > 0 else None,
            "candidate_counts": cand_counts,
            "query_narrative_embeddings": torch.stack(query_narr_embeddings, dim=0),
            "candidate_narrative_embeddings": padded_cand_narr,
            "candidate_mask": candidate_mask,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
        return batch




class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariationalBottleneck(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mu_net = MLP(hidden_size * 2, hidden_size, latent_dim, dropout)
        self.logvar_net = MLP(hidden_size * 2, hidden_size, latent_dim, dropout)
        self.decoder = MLP(latent_dim, hidden_size, hidden_size, dropout)

    def forward(self, h_u: torch.Tensor, h_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        joint = torch.cat([h_u, h_pos], dim=-1)
        mu = self.mu_net(joint)
        logvar = self.logvar_net(joint).clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = self.decoder(z)
        return z, mu, logvar, recon


class RAMMModel(nn.Module):
    def __init__(
        self,
        text_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        vision_model: nn.Module,
        qformer: nn.Module,
        query_tokens: torch.Tensor,
        image_token_id: int,
        narrative_embedding_dim: int,
        latent_dim: int,
        hidden_dropout: float,
        temperature: float,
        lambda_align: float,
        lambda_recon: float,
        lambda_compress: float,
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.vision_model = vision_model
        self.qformer = qformer
        self.register_buffer("query_tokens", query_tokens, persistent=False)
        self.image_token_id = image_token_id
        self.hidden_size = int(text_model.config.hidden_size)
        self.num_query_tokens = int(query_tokens.shape[1])

        qformer_dim = int(qformer.config.hidden_size)
        self.vision_projection = nn.Linear(qformer_dim, self.hidden_size)

        self.narrative_adapter = nn.Linear(narrative_embedding_dim, self.hidden_size)
        self.narrative_score = nn.Linear(self.hidden_size * 2, 1)
        self.variational_bottleneck = VariationalBottleneck(
            hidden_size=self.hidden_size,
            latent_dim=latent_dim,
            dropout=hidden_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.temperature = temperature
        self.lambda_align = lambda_align
        self.lambda_recon = lambda_recon
        self.lambda_compress = lambda_compress

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def _encode_images_to_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self._device())
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        query_embeds = self.query_tokens.expand(image_embeds.size(0), -1, -1).to(self._device())
        qformer_outputs = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=image_embeds,
            return_dict=True,
        )
        projected = self.vision_projection(qformer_outputs.last_hidden_state)
        return projected

    def _pool_last_valid_token(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lens = attention_mask.long().sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lens]

    def encode_single_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self._device())
        attention_mask = attention_mask.to(self._device())
        visual_tokens = self._encode_images_to_visual_tokens(pixel_values)

        text_embeds = self._get_input_embeddings()(input_ids)
        fused_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        fused_mask = torch.cat(
            [torch.ones(input_ids.size(0), self.num_query_tokens, dtype=attention_mask.dtype, device=self._device()), attention_mask],
            dim=1,
        )
        outputs = self.text_model(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        return self._pool_last_valid_token(last_hidden, fused_mask)

    def encode_sra_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        demo_pixel_values: torch.Tensor,
        query_pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self._device())
        attention_mask = attention_mask.to(self._device())
        demo_visual = self._encode_images_to_visual_tokens(demo_pixel_values)
        query_visual = self._encode_images_to_visual_tokens(query_pixel_values)

        text_embeds = self._get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)

        new_embeds: List[torch.Tensor] = []
        new_masks: List[torch.Tensor] = []
        for i in range(batch_size):
            image_token_positions = torch.where(input_ids[i] == self.image_token_id)[0]
            replacement_features = [demo_visual[i], query_visual[i]]

            parts: List[torch.Tensor] = []
            mask_parts: List[torch.Tensor] = []
            cur = 0
            for j, pos in enumerate(image_token_positions.tolist()):
                parts.append(text_embeds[i, cur:pos])
                mask_parts.append(attention_mask[i, cur:pos])
                if j < len(replacement_features):
                    parts.append(replacement_features[j])
                    mask_parts.append(torch.ones(self.num_query_tokens, dtype=attention_mask.dtype, device=self._device()))
                cur = pos + 1
            parts.append(text_embeds[i, cur:])
            mask_parts.append(attention_mask[i, cur:])

            sample_embeds = torch.cat(parts, dim=0)
            sample_mask = torch.cat(mask_parts, dim=0)
            new_embeds.append(sample_embeds)
            new_masks.append(sample_mask)

        max_len = max(x.size(0) for x in new_embeds)
        padded_embeds = torch.zeros(batch_size, max_len, self.hidden_size, dtype=text_embeds.dtype, device=self._device())
        padded_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=self._device())

        for i, (emb, mask) in enumerate(zip(new_embeds, new_masks)):
            padded_embeds[i, : emb.size(0)] = emb
            padded_mask[i, : mask.size(0)] = mask

        outputs = self.text_model(
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        return self._pool_last_valid_token(last_hidden, padded_mask)

    def build_positive_sample(
        self,
        query_narrative_embeddings: torch.Tensor,
        candidate_narrative_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_proj = self.narrative_adapter(query_narrative_embeddings)
        cand_proj = self.narrative_adapter(candidate_narrative_embeddings)

        query_expand = query_proj.unsqueeze(1).expand_as(cand_proj)
        pair_feat = torch.cat([query_expand, cand_proj], dim=-1)
        scores = F.leaky_relu(self.narrative_score(pair_feat).squeeze(-1), negative_slope=0.2)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        h_pos = torch.sum(attn.unsqueeze(-1) * candidate_hidden, dim=1)
        return h_pos, attn

    def info_nce(self, z: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        positives = F.normalize(positives, dim=-1)
        logits = torch.matmul(z, positives.transpose(0, 1)) / self.temperature
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        labels = batch["labels"].to(self._device())

        # SRA branch: prompt-based analogical reasoning
        h_pre = self.encode_sra_prompt(
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            batch["prompt_demo_pixel_values"],
            batch["prompt_query_pixel_values"],
        )
        logits = self.classifier(h_pre).squeeze(-1)
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # ANA branch: single sample representation + candidate representations + CIBL
        h_u = self.encode_single_multimodal(
            batch["query_input_ids"],
            batch["query_attention_mask"],
            batch["query_pixel_values"],
        )

        cand_counts: List[int] = list(batch["candidate_counts"])
        max_cands = max(cand_counts)
        candidate_hidden = torch.zeros(
            len(cand_counts),
            max_cands,
            self.hidden_size,
            device=self._device(),
            dtype=h_u.dtype,
        )

        if batch["candidate_input_ids"] is not None:
            cand_hidden_flat = self.encode_single_multimodal(
                batch["candidate_input_ids"],
                batch["candidate_attention_mask"],
                batch["candidate_pixel_values"],
            )
            cursor = 0
            for i, n in enumerate(cand_counts):
                candidate_hidden[i, :n] = cand_hidden_flat[cursor : cursor + n]
                cursor += n

        candidate_mask = batch["candidate_mask"].to(self._device())
        query_narr = batch["query_narrative_embeddings"].to(self._device())
        candidate_narr = batch["candidate_narrative_embeddings"].to(self._device())

        h_pos, attn = self.build_positive_sample(
            query_narrative_embeddings=query_narr,
            candidate_narrative_embeddings=candidate_narr,
            candidate_mask=candidate_mask,
            candidate_hidden=candidate_hidden,
        )

        z, mu, logvar, recon = self.variational_bottleneck(h_u, h_pos)
        align_loss = self.info_nce(z, h_pos)
        recon_loss = F.mse_loss(recon, h_u)
        compress_loss = self.kl_divergence(mu, logvar)

        total_loss = (
            cls_loss
            + self.lambda_align * align_loss
            + self.lambda_recon * recon_loss
            + self.lambda_compress * compress_loss
        )

        probs = torch.sigmoid(logits)
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "align_loss": align_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "compress_loss": compress_loss.detach(),
            "logits": logits,
            "probs": probs,
            "labels": labels,
            "attention_weights": attn.detach(),
        }





class ModelFactory:
    @staticmethod
    def _dtype_from_string(dtype_name: str) -> torch.dtype:
        name = dtype_name.lower().strip()
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float16":
            return torch.float16
        return torch.float32

    @staticmethod
    def build(args: ModelArguments, narrative_embedding_dim: int, device: str) -> Tuple[AutoTokenizer, AutoImageProcessor, RAMMModel]:
        torch_dtype = ModelFactory._dtype_from_string(args.torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        text_model = AutoModelForCausalLM.from_pretrained(args.text_model_name, **load_kwargs)
        special_tokens_dict = {"additional_special_tokens": [args.image_token]}
        tokenizer.add_special_tokens(special_tokens_dict)
        text_model.resize_token_embeddings(len(tokenizer))
        text_model.config.pad_token_id = tokenizer.pad_token_id

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        text_model = get_peft_model(text_model, lora_cfg)

        blip_model = Blip2Model.from_pretrained(
            args.vision_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        vision_model = blip_model.vision_model
        qformer = blip_model.qformer
        query_tokens = blip_model.query_tokens
        image_processor = AutoImageProcessor.from_pretrained(args.vision_model_name, trust_remote_code=True)

        if args.freeze_vision_encoder:
            for p in vision_model.parameters():
                p.requires_grad = False
            vision_model.eval()

        if args.freeze_qformer:
            for p in qformer.parameters():
                p.requires_grad = False
            qformer.eval()

        if not args.use_4bit:
            text_model = text_model.to(device)
            vision_model = vision_model.to(device)
            qformer = qformer.to(device)
            query_tokens = query_tokens.to(device)

        image_token_id = tokenizer.convert_tokens_to_ids(args.image_token)
        model = RAMMModel(
            text_model=text_model,
            tokenizer=tokenizer,
            vision_model=vision_model,
            qformer=qformer,
            query_tokens=query_tokens,
            image_token_id=image_token_id,
            narrative_embedding_dim=narrative_embedding_dim,
            latent_dim=args.latent_dim,
            hidden_dropout=args.hidden_dropout,
            temperature=args.temperature,
            lambda_align=args.lambda_align,
            lambda_recon=args.lambda_recon,
            lambda_compress=args.lambda_compress,
        )
        if not args.use_4bit:
            model = model.to(device)
        return tokenizer, image_processor, model



class TrainerEngine:
    def __init__(self, model: RAMMModel, train_args: TrainArguments, device: str):
        self.model = model
        self.train_args = train_args
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_autocast = self.device.type == "cuda"
        self.autocast_dtype = torch.bfloat16

    def build_optimizer(self) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith("bias") or "norm" in n.lower():
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.train_args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.train_args.learning_rate)

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        output_dir: str = "./outputs/ramm",
    ) -> Dict[str, float]:
        ensure_dir(output_dir)
        optimizer = self.build_optimizer()
        total_update_steps = math.ceil(len(train_loader) / self.train_args.gradient_accumulation_steps) * self.train_args.epochs
        warmup_steps = int(total_update_steps * self.train_args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

        best_metric = -float("inf")
        best_metrics: Dict[str, float] = {}
        global_step = 0
        running_loss = 0.0

        self.model.train()
        for epoch in range(1, self.train_args.epochs + 1):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.train_args.epochs}")
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(pbar, start=1):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_autocast):
                    outputs = self.model(batch)
                    loss = outputs["loss"] / self.train_args.gradient_accumulation_steps

                loss.backward()
                running_loss += float(loss.item())

                if step % self.train_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.train_args.log_steps == 0:
                        pbar.set_postfix(
                            {
                                "loss": f"{running_loss / self.train_args.log_steps:.4f}",
                                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            }
                        )
                        running_loss = 0.0

                    if eval_loader is not None and global_step % self.train_args.eval_steps == 0:
                        metrics = self.evaluate(eval_loader)
                        monitored = metrics.get(self.train_args.monitor_metric, float("-inf"))
                        print(f"\n[Eval @ step {global_step}] {metrics}\n")
                        self.model.train()

                        save_ckpt = monitored > best_metric
                        #if save_ckpt:
                          #  best_metric = monitored
                          #  best_metrics = metrics
                            #self.save_checkpoint(os.path.join(output_dir, "best_model.pt"), optimizer, scheduler, epoch, global_step, metrics)
                        #elif not self.train_args.save_best_only:
                            #self.save_checkpoint(
                             #   os.path.join(output_dir, f"checkpoint_step_{global_step}.pt"),
                             #   optimizer,
                             #   scheduler,
                             #   epoch,
                             #   global_step,
                             #   metrics,
                            #)

            # epoch end eval
            if eval_loader is not None:
                metrics = self.evaluate(eval_loader)
                monitored = metrics.get(self.train_args.monitor_metric, float("-inf"))
                print(f"\n[Epoch {epoch} Eval] {metrics}\n")
                self.model.train()
                if monitored > best_metric:
                    best_metric = monitored
                    best_metrics = metrics
                    #self.save_checkpoint(os.path.join(output_dir, "best_model.pt"), optimizer, scheduler, epoch, global_step, metrics)

        if len(best_metrics) == 0 and eval_loader is None:
            best_metrics = {"status": 1.0}
        return best_metrics

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_probs: List[float] = []
        all_labels: List[int] = []
        losses: List[float] = []
        cls_losses: List[float] = []
        align_losses: List[float] = []
        recon_losses: List[float] = []
        compress_losses: List[float] = []

        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_autocast):
                outputs = self.model(batch)
            losses.append(float(outputs["loss"].item()))
            cls_losses.append(float(outputs["cls_loss"].item()))
            align_losses.append(float(outputs["align_loss"].item()))
            recon_losses.append(float(outputs["recon_loss"].item()))
            compress_losses.append(float(outputs["compress_loss"].item()))
            all_probs.extend(outputs["probs"].detach().cpu().tolist())
            all_labels.extend(outputs["labels"].detach().cpu().long().tolist())

        metrics = compute_classification_metrics(all_labels, all_probs)
        metrics.update(
            {
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "cls_loss": float(np.mean(cls_losses)) if cls_losses else float("nan"),
                "align_loss": float(np.mean(align_losses)) if align_losses else float("nan"),
                "recon_loss": float(np.mean(recon_losses)) if recon_losses else float("nan"),
                "compress_loss": float(np.mean(compress_losses)) if compress_losses else float("nan"),
            }
        )
        return metrics

    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
    ) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        }
        torch.save(state, path)
        print(f"Checkpoint saved to: {path}")





class PipelineBuilder:
    @staticmethod
    def load_news_items(path: str) -> List[NewsItem]:
        raw = load_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Main dataset must be a list. Got: {type(raw)}")
        items: List[NewsItem] = []
        for idx, rec in enumerate(raw):
            items.append(
                NewsItem(
                    index=idx,
                    item_id=resolve_id(rec, idx),
                    text=resolve_text(rec),
                    image=resolve_image(rec),
                    label=resolve_label(rec),
                    domain=resolve_domain(rec),
                    narrative_text=resolve_narrative_text(rec),
                )
            )
        return items

    @staticmethod
    def build_narrative_cache(
        items: List[NewsItem],
        encoder: NarrativeTextEncoder,
        cache_path: str,
        force_rebuild: bool = False,
    ) -> torch.Tensor:
        if os.path.exists(cache_path) and not force_rebuild:
            print(f"Loading narrative cache from {cache_path}")
            return torch.load(cache_path, map_location="cpu")
        texts = [x.narrative_text for x in items]
        cache = encoder.encode(texts, batch_size=32, show_progress=True)
        torch.save(cache, cache_path)
        print(f"Saved narrative cache to {cache_path}")
        return cache

    @staticmethod
    def build_datasets(
        data_args: DataArguments,
        model_args: ModelArguments,
        output_dir: str,
    ) -> Tuple[List[NewsItem], List[NewsItem], RAMMDataset, RAMMDataset, int]:
        ensure_dir(output_dir)
        train_items = PipelineBuilder.load_news_items(data_args.train_data_path)
        test_items = PipelineBuilder.load_news_items(data_args.test_data_path)

        sra_train_map = RetrievalParser.parse_sra_records(data_args.sra_train_retrieval_path)
        sra_test_map = RetrievalParser.parse_sra_records(data_args.sra_test_retrieval_path)
        ana_train_map = RetrievalParser.parse_ana_records(data_args.ana_train_retrieval_path, data_args.kin, data_args.kout)
        ana_test_map = RetrievalParser.parse_ana_records(data_args.ana_test_retrieval_path, data_args.kin, data_args.kout)

        encoder = NarrativeTextEncoder(
            model_name=model_args.narrative_encoder_name,
            max_length=model_args.max_narrative_length,
            device="cuda",
            fp32=model_args.language.lower().startswith("zh"),
        )

        train_cache = PipelineBuilder.build_narrative_cache(
            train_items,
            encoder,
            cache_path=os.path.join(output_dir, "train_narrative_cache.pt"),
        )
        test_cache = PipelineBuilder.build_narrative_cache(
            test_items,
            encoder,
            cache_path=os.path.join(output_dir, "test_narrative_cache.pt"),
        )
        narrative_dim = int(train_cache.size(1))
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_dataset = RAMMDataset(
            query_items=train_items,
            train_reference_items=train_items,
            sra_map=sra_train_map,
            ana_map=ana_train_map,
            query_narrative_cache=train_cache,
            train_narrative_cache=train_cache,
        )
        eval_dataset = RAMMDataset(
            query_items=test_items,
            train_reference_items=train_items,
            sra_map=sra_test_map,
            ana_map=ana_test_map,
            query_narrative_cache=test_cache,
            train_narrative_cache=train_cache,
        )
        return train_items, test_items, train_dataset, eval_dataset, narrative_dim




def parse_args() -> Tuple[ModelArguments, DataArguments, TrainArguments]:
    parser = argparse.ArgumentParser(description="Full RAMM training script")

    # data
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--sra_train_retrieval_path", type=str, required=True)
    parser.add_argument("--sra_test_retrieval_path", type=str, required=True)
    parser.add_argument("--ana_train_retrieval_path", type=str, required=True)
    parser.add_argument("--ana_test_retrieval_path", type=str, required=True)
    parser.add_argument("--image_base_dir", type=str, required=True)
    parser.add_argument("--kin", type=int, default=3)
    parser.add_argument("--kout", type=int, default=2)

    # model
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--vision_model_name", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--narrative_encoder_name", type=str, default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--freeze_qformer", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--image_token", type=str, default="<image>")
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_single_text_length", type=int, default=256)
    parser.add_argument("--max_narrative_length", type=int, default=256)
    parser.add_argument("--hidden_dropout", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_align", type=float, default=0.2)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--lambda_compress", type=float, default=0.002)
    parser.add_argument("--language", type=str, default="zh")

    # train
    parser.add_argument("--output_dir", type=str, default="./outputs/ramm")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_best_only", action="store_true")
    parser.add_argument("--monitor_metric", type=str, default="f1")

    ns = parser.parse_args()

    model_args = ModelArguments(
        text_model_name=ns.text_model_name,
        vision_model_name=ns.vision_model_name,
        narrative_encoder_name=ns.narrative_encoder_name,
        use_4bit=ns.use_4bit,
        torch_dtype=ns.torch_dtype,
        freeze_vision_encoder=ns.freeze_vision_encoder,
        freeze_qformer=ns.freeze_qformer,
        lora_r=ns.lora_r,
        lora_alpha=ns.lora_alpha,
        lora_dropout=ns.lora_dropout,
        image_token=ns.image_token,
        max_prompt_length=ns.max_prompt_length,
        max_single_text_length=ns.max_single_text_length,
        max_narrative_length=ns.max_narrative_length,
        hidden_dropout=ns.hidden_dropout,
        latent_dim=ns.latent_dim,
        temperature=ns.temperature,
        lambda_align=ns.lambda_align,
        lambda_recon=ns.lambda_recon,
        lambda_compress=ns.lambda_compress,
        language=ns.language,
    )
    data_args = DataArguments(
        train_data_path=ns.train_data_path,
        test_data_path=ns.test_data_path,
        sra_train_retrieval_path=ns.sra_train_retrieval_path,
        sra_test_retrieval_path=ns.sra_test_retrieval_path,
        ana_train_retrieval_path=ns.ana_train_retrieval_path,
        ana_test_retrieval_path=ns.ana_test_retrieval_path,
        image_base_dir=ns.image_base_dir,
        kin=ns.kin,
        kout=ns.kout,
    )
    train_args = TrainArguments(
        output_dir=ns.output_dir,
        seed=ns.seed,
        epochs=ns.epochs,
        train_batch_size=ns.train_batch_size,
        eval_batch_size=ns.eval_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        warmup_ratio=ns.warmup_ratio,
        max_grad_norm=ns.max_grad_norm,
        num_workers=ns.num_workers,
        log_steps=ns.log_steps,
        eval_steps=ns.eval_steps,
        save_best_only=ns.save_best_only,
        monitor_metric=ns.monitor_metric,
    )
    return model_args, data_args, train_args





def main() -> None:
    model_args, data_args, train_args = parse_args()
    ensure_dir(train_args.output_dir)
    set_seed(train_args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("RAMM full reproduction script")
    print(f"Device: {device}")
    print("Model args:", json.dumps(asdict(model_args), ensure_ascii=False, indent=2))
    print("Data args:", json.dumps(asdict(data_args), ensure_ascii=False, indent=2))
    print("Train args:", json.dumps(asdict(train_args), ensure_ascii=False, indent=2))
    print("=" * 80)

    train_items, test_items, train_dataset, eval_dataset, narrative_dim = PipelineBuilder.build_datasets(
        data_args=data_args,
        model_args=model_args,
        output_dir=train_args.output_dir,
    )

    tokenizer, image_processor, model = ModelFactory.build(
        args=model_args,
        narrative_embedding_dim=narrative_dim,
        device=device,
    )

    collator = RAMMCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        train_reference_items=train_items,
        image_base_dir=data_args.image_base_dir,
        image_token=model_args.image_token,
        language=model_args.language,
        max_prompt_length=model_args.max_prompt_length,
        max_single_text_length=model_args.max_single_text_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_args.train_batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_args.eval_batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    engine = TrainerEngine(model=model, train_args=train_args, device=device)
    best_metrics = engine.train(train_loader, eval_loader, train_args.output_dir)
    print("Best validation/test metrics:")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))

    final_metrics = engine.evaluate(eval_loader)
    print("Final evaluation metrics:")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))

    with open(os.path.join(train_args.output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(train_args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_args": asdict(model_args),
                "data_args": asdict(data_args),
                "train_args": asdict(train_args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
