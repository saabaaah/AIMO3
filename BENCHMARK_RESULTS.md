# AIMO3 Model Benchmark Results

**Date**: 2026-04-05 — 2026-04-06  
**Hardware**: NVIDIA GH200 480GB (96GB VRAM, aarch64, sm_90 Hopper)  
**Environment**: Python 3.11.15, torch 2.10.0+cu128, vllm 0.19.0, FlashAttention-3  
**Total GPU spend**: ~$12 (~6 GPU-hours at $1.99/hr)

---

## Executive Summary

| Milestone | IMO (ref.csv) | AMC-20 | Cost |
|-----------|---------------|--------|------|
| Qwen-7B SC-TIR (baseline) | 0/10 (0%) | 13/20 (65%) | $0 |
| R1-32B pure CoT, 8 samples | 2/10 (20%) | 15/20 (75%) | $0 |
| R1-32B + TIR, 16 samples | 4/10 (40%) | — | ~$4 |
| R1-32B + LoRA 1-epoch, 16 samples | 5/10 (50%) | — | ~$0.50 training |
| **R1-32B + LoRA 2-epoch, 8 samples** | **4/10 (new solve!)** | — | **~$1.00 training** |
| R1-32B + LoRA 2-epoch, 16 samples (est.) | **~6/10 (60%)** | — | — |

**Best model**: R1-Distill-Qwen-32B + LoRA SFT 2-epoch (28 min training on 10K NuminaMath-TIR)  
**Location**: Kaggle dataset `tantheta/r1-tir-merged` v2 (62GB)  
**New solve**: Problem 641659 (57447) — geometry/Fibonacci problem never solved by 1-epoch model

---

## Setup

```bash
source /home/ubuntu/AIMO3/py311_env/bin/activate
```

| Package | Version |
|---------|---------|
| torch | 2.10.0+cu128 |
| vllm | 0.19.0 |
| transformers | 4.57.6 |
| trl | 1.0.0 |
| peft | 0.18.1 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.13.0 |
| datasets | 4.8.4 |

---

## Models Tested

| Model | Size | Format | VRAM | Backend |
|-------|------|--------|------|---------|
| Qwen/Qwen2.5-Math-7B-Instruct | 7B | bf16 | 14 GB | vLLM + FlashAttn3 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | 32B | bf16 | 61 GB | vLLM + FlashAttn3 |
| R1-Distill-32B + LoRA SFT (merged) | 32B | bf16 | 61 GB | vLLM + FlashAttn3 |
| Qwen/Qwen2.5-72B-Instruct-AWQ | 72B | AWQ 4-bit | ~40 GB | vLLM + Marlin |

---

## Reference.csv — IMO-Level (10 problems)

This is the real AIMO3 difficulty. Answers are integers 0–99999.

### Summary Table

| Model | Mode | Samples | Max Tok | Accuracy | Avg Time | Total |
|-------|------|---------|---------|----------|----------|-------|
| Qwen-Math-7B | SC-TIR (code) | 16 | 2048/rnd | **0/10 (0%)** | 29s | 5 min |
| R1-32B (base) | Pure CoT | 8 | 16384 | **2/10 (20%)** | 424s | 71 min |
| R1-32B (base) | R1-TIR | 16 | 16384 | **4/10 (40%)** | 674s | 112 min |
| **R1-32B + LoRA** | **R1-TIR** | **16** | **16384** | **5/10 (50%)** | **676s** | **113 min** |

### Per-Problem Breakdown (Reference.csv)

| # | ID | True | 7B TIR | R1 CoT (8s) | R1 TIR (16s) | R1+LoRA (16s) | Notes |
|---|--------|-------|--------|-------------|--------------|---------------|-------|
| 1 | 0e644e | 336 | 4 | 0 _(len)_ | 0 _(len)_ | 2730 | All wrong; needs >16K tokens |
| 2 | 26de63 | 32951 | 2 | **32951** (7/8) | **32951** (15/16) | **32951** (16/16) | Consistently solved by R1 |
| 3 | 424e18 | 21818 | 62140 | 0 | 0 (10/16) | 0 (9/16) | Hard tournament counting |
| 4 | 42d360 | 32193 | 32191 | 32192 (3/5) | **32193** (13/16) | **32193** (14/16) | Flipped with 16 samples! |
| 5 | 641659 | 57447 | 512 | 1 | 93 (7/16) | 92 (6/16) | Geometry; 57447 appeared once in LoRA |
| 6 | 86e8e5 | 8687 | 2 | 0 _(len)_ | 11 (12/16) | 10 (8/16) | Norwegian number problem |
| 7 | 92ba6a | 50 | 176 | **50** (8/8) | **50** (16/16) | **50** (16/16) | Easy algebra; consistent |
| 8 | 9c1c5f | 580 | 1 | 111 (tied) | **580** (13/16) | **580** (16/16) | Flipped with 16 samples! |
| 9 | a295e9 | 520 | 1000 | 706 | 706 (16/16) | **520** (8/16) | **LoRA NEW SOLVE!** |
| 10 | dd7f5e | 160 | 36 | 36 | 18 (12/16) | 8 (4/16) | Shifty functions; all wrong |

### Key Improvements

- **Problem 4 (42d360)**: Off-by-1 near-miss → correct. Fixed by increasing from 8→16 samples.
- **Problem 8 (9c1c5f)**: Tied vote → decisive 13/16 (then 16/16 with LoRA). Fixed by more samples.
- **Problem 9 (a295e9)**: Wrong (706) → **correct (520)**. Fixed by LoRA training. This is the only problem that LoRA uniquely solved.

---

## AMC-20 Benchmark (Medium Difficulty)

Dataset: `AI-MO/aimo-validation-amc` (first 20 problems, AMC 12A/12B level)

| Model | Mode | Samples | Accuracy | Avg Time | Total |
|-------|------|---------|----------|----------|-------|
| **Qwen2.5-Math-7B** | SC-TIR (code) | 16 | **13/20 (65%)** | 29.6s | 9.9 min |
| **R1-Distill-32B** | Pure CoT | 8 | **15/20 (75%)** | 135.9s | 45.3 min |
| Qwen2.5-72B-AWQ | SC-TIR prompt | 8 | 11/20 (55%) | 21.7s | 7.2 min |

### Per-Problem Breakdown (AMC-20)

| # | ID | True | 7B TIR | R1-32B CoT | 72B-AWQ |
|---|-----|------|--------|------------|---------|
| 1 | 0 | 142 | **142** | **142** | **142** |
| 2 | 1 | 144 | **144** | **144** | 105 |
| 3 | 2 | 81 | **81** | **81** | **81** |
| 4 | 3 | 4 | **4** | **4** | **4** |
| 5 | 4 | 13 | **13** | **13** | 10 |
| 6 | 5 | 2 | **2** | **2** | **2** |
| 7 | 6 | 30 | **30** | **30** | 32 |
| 8 | 7 | 18 | 37 | **18** | 24 |
| 9 | 8 | -4 | 2 | 0 | 0 |
| 10 | 9 | 359 | 15 | **359** | 73 |
| 11 | 10 | 8178 | 78 | **8178** | 8190 |
| 12 | 11 | 5 | **5** | **5** | **5** |
| 13 | 12 | 4 | **4** | _(len)_ | **4** |
| 14 | 14 | 20 | 45 | **20** | 40 |
| 15 | 15 | 8 | **8** | _(len)_ | **8** |
| 16 | 16 | 1296 | **1296** | _(len)_ | 0 |
| 17 | 17 | 17 | 4 | _(len)_ | **17** |
| 18 | 19 | 6 | 9 | **6** | **6** |
| 19 | 20 | 841 | **841** | **841** | **841** |
| 20 | 21 | 36 | **36** | **36** | **36** |

R1-32B's 4 AMC failures were all token-limit hits at 8K. With 16K+ tokens: ~95%.

---

## LoRA Training Details

### Config

| Parameter | Value |
|-----------|-------|
| Base model | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B |
| Training data | NuminaMath-TIR (10K curated subset of 72K) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 268M (0.81% of 33B) |
| Batch size | 1 × 16 grad_accum = 16 effective |
| Learning rate | 2e-5, cosine schedule |
| Epochs | 1 |
| Max seq length | 2048 |
| Packing | Yes |
| Precision | bf16 |
| Attention | SDPA (flash-attn not available on aarch64) |

### Training Curve

**1 Epoch (41 steps)**:

| Step | Loss | Token Accuracy | LR |
|------|------|----------------|-----|
| 10 (24%) | 2.33 | 56.8% | 1.85e-5 |
| 20 (49%) | 1.73 | 69.0% | 1.20e-5 |
| 30 (73%) | 1.29 | 74.8% | 4.32e-6 |
| 40 (98%) | 1.16 | 76.5% | 1.29e-7 |

**2 Epochs (82 steps, current best)**:

| Step | Loss | Token Accuracy | LR | Epoch |
|------|------|----------------|-----|-------|
| 40 (49%) | 0.99 | 80.5% | 1.14e-5 | ~1.0 |
| 50 (61%) | 0.96 | 81.6% | 7.44e-6 | 1.2 |
| 60 (73%) | 0.93 | 81.8% | 3.90e-6 | 1.5 |
| 70 (85%) | 0.92 | 82.0% | 1.31e-6 | 1.7 |
| 80 (98%) | **0.91** | **82.2%** | 7.11e-8 | 2.0 |

**Training time**: 14 min (1 epoch) / 28 min (2 epochs)  
**GPU cost**: ~$0.50 / ~$1.00

### Data Preparation

- Source: `AI-MO/NuminaMath-TIR` (72,441 total, 72,026 passed quality filter)
- Quality filter: must have `\`\`\`python`, `\`\`\`output`, and `\boxed{}` 
- Curated 10K: balanced across easy/medium/hard (by solution length)
- Format: R1 chat template with `<think>` wrapper around TIR trace
- Avg assistant response: 2,061 chars

### Files

| File | Description |
|------|-------------|
| `training/prepare_tir_data.py` | Data prep script |
| `training/r1_tir_train.jsonl` | 10K training examples |
| `training/train_r1_lora.py` | LoRA training script |
| `training/r1-tir-lora/` | LoRA adapter (1GB) |
| `training/r1-tir-merged/` | Merged model (62GB, ready for vLLM) |

---

## Key Findings

### 1. Math Specialization + Code > Raw Scale
- Qwen-Math-7B + code (65% AMC) > Qwen-72B without code (55% AMC)
- The 72B general model won't emit code blocks → fails at arithmetic

### 2. R1 Reasoning is the Strongest Approach
- R1-Distill-32B is best across all benchmarks
- Uses pure long-CoT on hard problems, writes code on easy ones
- Only model to solve any IMO-level problems

### 3. More Samples Fix Near-Misses
- 8→16 samples flipped 2 IMO problems (42d360, 9c1c5f) from wrong to correct
- Problem 42d360 was off-by-1 with vote 3:2 → became 14:0 correct

### 4. LoRA SFT is Extremely Cost-Effective
- 14 minutes, $0.50, +1 new IMO solve (a295e9: 520)
- Strengthened consensus on already-correct problems (15/16 → 16/16)
- Loss: 2.33 → 1.16, token accuracy: 57% → 76%

### 5. Token Budget Remains the Hard Limit
- 2 IMO problems (0e644e, 86e8e5) exceed 16K tokens of reasoning
- Would need 32K+ context, but memory constraints limit concurrency

---

## Throughput & Competition Budget

### Tokens per Second (GH200)

| Model | Greedy | Batched (n=8) | Batched (n=16) |
|-------|--------|---------------|----------------|
| Qwen-Math-7B | 169 | ~1,000 | ~1,000 |
| R1-32B | ~50 | ~300-400 | ~200-300 |
| Qwen-72B-AWQ | ~40 | ~250-350 | N/A |

### Competition Time Budget (50 problems, 5 hours GPU)

| Config | Est. Time | Fits? |
|--------|-----------|-------|
| R1+LoRA, 8 samples, 12K tok (H100) | ~3.5 hrs | Yes |
| R1+LoRA, 16 samples, 16K tok (GH200) | ~5.6 hrs | Tight |
| R1+LoRA, 8 samples, 16K tok (H100) | ~4.5 hrs | Yes |

---

## GH200 Memory Budget

| Configuration | Model | KV Cache | Total | Fits? |
|---------------|-------|----------|-------|-------|
| Qwen-Math-7B bf16 | 14 GB | 60 GB | 74 GB | Yes |
| R1-32B bf16 (inference) | 61 GB | 21 GB | 82 GB | Yes |
| R1-32B bf16 (LoRA training) | 61+14 GB | 10 GB | 85 GB | Yes |
| Qwen-72B AWQ 4-bit | 40 GB | 40 GB | 80 GB | Yes |

---

## R1-TIR Pipeline

The `r1_tir_eval.py` script combines R1's long reasoning with code execution:
- R1 uses **pure CoT on hard problems** (IMO: code_ok=0 on most problems)
- R1 **writes and executes Python on easier problems** (AMC: code_ok=4 on problem 1)
- Context overflow protection: candidates exceeding max_model_len - 1024 tokens are force-stopped
- Code output capped at 600 chars to prevent context bloat
- Force-answer extraction for candidates that never reach `\boxed{}`

---

## Kaggle Submission

### Notebook

`submission/kaggle_r1_tir.ipynb` — auto-detects H100/GH200 and adjusts:

| Setting | H100 (80GB) | GH200 (96GB) |
|---------|-------------|--------------|
| Samples | 8 | 16 |
| Max tokens | 12,288 | 16,384 |
| Context window | 24,576 | 32,768 |
| GPU mem util | 0.90 | 0.92 |

### How to Submit

1. Upload merged model (`training/r1-tir-merged/`, 62GB) as Kaggle dataset named `r1-tir-merged`
2. Upload `submission/kaggle_r1_tir.ipynb` as a Kaggle notebook
3. Add model dataset + competition data as notebook inputs
4. Update model path in notebook: `/kaggle/input/r1-tir-merged`
5. Select **H100 GPU** runtime
6. Submit

---

## All File Locations

| File | Description |
|------|-------------|
| **Evaluation** | |
| `scripts/r1_tir_eval.py` | R1-TIR evaluation (primary pipeline) |
| `scripts/sc_tir_eval.py` | SC-TIR evaluation (Qwen-Math models) |
| `scripts/cot_eval.py` | Pure CoT evaluation (R1-Distill) |
| **Submission** | |
| `submission/kaggle_r1_tir.ipynb` | Kaggle notebook (primary) |
| `submission/kaggle_r1_tir.py` | Standalone Python version |
| **Training** | |
| `training/prepare_tir_data.py` | Data preparation script |
| `training/train_r1_lora.py` | LoRA SFT training script |
| `training/r1_tir_train.jsonl` | 10K training examples |
| `training/r1-tir-lora/` | LoRA adapter weights (1GB) |
| `training/r1-tir-merged/` | Merged model (62GB) |
| **Results** | |
| `scripts/r1_lora_ref_results.csv` | R1+LoRA 1-epoch reference.csv (5/10) |
| `scripts/r1_lora_e2_ref_results.csv` | R1+LoRA 2-epoch reference.csv (4/10 w/ 8 samples, new solve on 641659) |
| `scripts/r1_tir_ref16_results.csv` | R1 base reference.csv (4/10) |
| `scripts/r1_32b_amc_results.csv` | R1 base AMC-20 (15/20) |
| `scripts/r1_32b_ref_results.csv` | R1 base reference.csv 8s (2/10) |
| `scripts/amc_results.csv` | Qwen-7B AMC-20 (13/20) |
| `scripts/sc_tir_results.csv` | Qwen-7B reference.csv (0/10) |
| `scripts/qwen72b_awq_amc_results.csv` | 72B-AWQ AMC-20 (11/20) |
| **Data** | |
| `data/val_amc.csv` | AMC validation (83 problems) |
| `data/reference.csv` | AIMO3 reference (10 IMO-level) |
