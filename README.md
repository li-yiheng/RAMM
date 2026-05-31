# RAMM: Retrieval-Augmented Multimodal Model for Fake News Detection

<p align="center">
  <b>Retrieval-Augmented Multimodal Model for Cross-Domain Fake News Detection</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#method">Method</a> •
</p>

---

## News

- **[2026-05]** Initial engineering-oriented implementation of RAMM is released.

---

## Overview

**RAMM** is a retrieval-augmented multimodal framework for fake news detection. It is designed for **multi-domain** and **cross-domain** scenarios where a target news item should not be judged in isolation. Instead, RAMM retrieves related multimodal instances and high-level narrative neighbors, then uses them to improve veracity prediction.

The core motivation is that fake news often propagates in **clusters**: different image-text posts may express distinct surface forms while promoting the same underlying claim or misleading narrative. RAMM therefore combines:

1. **MLLM Backbone**: extracts multimodal representations from text-image news samples.
2. **SRA: Semantic Representation Alignment**: retrieves demonstration samples through multimodal image/text similarity and reformulates prediction as analogy-based reasoning.
3. **ANA: Abstract Narrative Alignment**: extracts core narratives, retrieves in-domain and out-of-domain homogeneous narrative neighbors, and constructs stronger positive references.
4. **CIBL: Common Information Bottleneck Loss**: jointly performs alignment, reconstruction, and compression to preserve useful common narrative signals while reducing noisy domain-specific shortcuts.

This repository provides a reproduction-oriented implementation of the RAMM pipeline, including narrative extraction, multimodal retrieval, abstract-narrative retrieval, model training, evaluation, and checkpoint saving.

---

## Method

### Pipeline

```text
Raw multimodal news
      │
      ├── api-weibo.py / api-weibo21.py
      │       └── extract core_narrative with a vision-language API by qwen-vl-max
      │
      ├── SSR-weibo.py / SSR-weibo21.py
      │       └── retrieve semantically similar demonstration samples by CLIP image/text similarity
      │
      ├── nara.py
      │       └── retrieve in-domain and out-of-domain narrative neighbors by text embeddings
      │
      └── ramm.py
              └── train / evaluate RAMM with MLLM + SRA + ANA + CIBL
