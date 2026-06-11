---
title: DPDP Act Compliance Filter — Machine Unlearning
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: mit
short_description: DPDP Act Machine Unlearning via TinyLlama
tags:
  - machine-unlearning
  - privacy
  - DPDP
  - LLM
  - LoRA
  - gradio
  - NLP
  - responsible-ai
---

# 🧠 DPDP Act Compliance Filter — Machine Unlearning Demo

**Parameter-level machine unlearning on TinyLlama-1.1B enforcing India's Right to Erasure.**

## What This Demo Shows

This Space demonstrates a fully-trained machine unlearning pipeline that surgically erases targeted private data from an LLM's weights — without retraining the model from scratch.

Try the **⚔️ Attack prompts** to attempt PII extraction — the model will fail to recall the erased authors, proving the DPDP erasure worked.

Try the **🧠 Utility prompts** to confirm the model's general intelligence is fully preserved.

## Tech Stack
- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 + trained LoRA adapters
- **Unlearning:** Gradient Ascent + KL Divergence Regularization
- **Evaluation:** Kolmogorov-Smirnov Test + ROUGE-L scoring
- **Framework:** PyTorch + HuggingFace PEFT + Gradio

## Source Code
Full training pipeline, evaluation scripts, and Colab notebook available on [GitHub](https://github.com/YOUR_USERNAME/dpdp-machine-unlearning).
