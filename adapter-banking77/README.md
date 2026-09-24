---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
license: cc-by-4.0
language: [en]
pipeline_tag: text-classification
tags:
- base_model:adapter:HuggingFaceTB/SmolLM2-135M-Instruct
- banking77
- lora
- transformers
---

# SmolLM2 Banking77 LoRA adapter

This is a 77-class English banking-support intent classifier. It is a research
demonstration, not a banking product, fraud detector, or authorization system.

## Model details

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac` (Apache-2.0)
- Dataset: PolyAI Banking77, repository revision
  `57ec275d8078af65b7731c2a98be812d844a6d6b` (CC BY 4.0)
- Adapter: LoRA rank 8, alpha 16, dropout 0.05 on `q_proj` and `v_proj`,
  with the classification head saved
- Training: all 10,003 official training examples, one epoch, seed 42,
  batch size 8, learning rate 5e-4, maximum length 96, float32
- Hardware: 16 GB Apple MacBook Air using MPS
- Frameworks: PyTorch 2.13.0, Transformers 5.16.1, PEFT 0.20.0,
  Accelerate 1.14.0, Datasets 4.1.1

## Evaluation

The official 3,080-example test split produced 83.25% exact-intent accuracy and
92.01% four-way operational-action accuracy. At the reported 0.70 confidence
threshold, coverage was 67.05%; accepted intent accuracy was 95.88%, and
accepted action accuracy was 98.45%. See `../evidence/` for machine-readable
results and the threshold curve.

The threshold curve is descriptive test-set analysis. Do not treat 0.70 as a
prospectively validated production threshold; select and lock a threshold on a
separate validation set before any deployment.

## Intended use and limitations

Use this adapter for learning, research, and reproducibility experiments. Do
not use it to make autonomous decisions about accounts, payments, fraud,
identity, eligibility, or customer access. Banking77 contains support intents,
not verified financial outcomes.

It was evaluated only on the English Banking77 distribution. It has no
subgroup, multilingual, robustness, adversarial, or out-of-distribution
evaluation. Inputs are truncated to 96 tokens. Confidence is not a guarantee,
and the broad prototype action mapping inflates action accuracy because many
intents share one action.

## Usage

From the repository root:

```bash
python banking77.py predict "Why was my card payment declined?"
```

## Licensing and attribution

This adapter is distributed under CC BY 4.0 to preserve the attribution terms
of Banking77. Banking77 is by PolyAI and is described in *Efficient Intent
Detection with Dual Sentence Encoders* (Casanueva et al., 2020). The base model
remains subject to Apache License 2.0. See `../THIRD_PARTY_NOTICES.md`.
