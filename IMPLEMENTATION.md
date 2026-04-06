# AIMO3 Implementation Guide

## Overview

This is a complete implementation for the AI Mathematical Olympiad Progress Prize 3 (AIMO3) Kaggle competition. The system uses **DeepSeek-R1-Distill-Qwen-32B** fine-tuned with LoRA on NuminaMath-TIR data, combining long chain-of-thought reasoning with Python code execution.

**Final score**: 5/10 on IMO-level reference problems (up from 0/10 baseline)

---

## Architecture

```
Problem
  │
  ▼
┌──────────────────────────┐
│  R1-Distill-32B + LoRA   │   <-- Long chain-of-thought reasoning
│  (bf16, 61GB VRAM)       │       with <think>...</think> tags
└──────────┬───────────────┘
           │
     ┌─────▼─────┐
     │ Generate N │   <-- N=8 (H100) or N=16 (GH200) candidates
     │ candidates │       Stop at: ```output, </think>, EOS
     └─────┬─────┘
           │
     ┌─────▼─────────┐
     │ Code blocks?   │──Yes──▶ Execute Python (sympy, numpy)
     │ ```python...```│        Inject ```output...``` result
     └─────┬─────────┘        Continue generation
           │ No
     ┌─────▼─────┐
     │ \boxed{}? │──Yes──▶ Extract answer, mark candidate done
     └─────┬─────┘
           │ No (more rounds)
           ▼
     (repeat up to 5 rounds)
           │
     ┌─────▼──────┐
     │ Force answer│   <-- For candidates that never reached \boxed{}
     │ extraction  │       Append "The final answer is \boxed{" + greedy
     └─────┬──────┘
           │
     ┌─────▼───────┐
     │ Majority     │   <-- Count votes, most common answer wins
     │ Vote         │
     └─────┬───────┘
           │
           ▼
        Answer (int 0-99999)
```

---

## Pipeline Steps

### Step 1: Environment Setup (GH200)

```bash
# Install Python 3.11
sudo apt-get install python3.11 python3.11-venv python3.11-dev

# Create venv
python3.11 -m venv py311_env
source py311_env/bin/activate

# Install packages
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install vllm transformers trl peft bitsandbytes datasets accelerate scipy
```

### Step 2: Data Preparation

```bash
python training/prepare_tir_data.py
```

This downloads NuminaMath-TIR (72K problems) from HuggingFace, filters for quality (must have `\`\`\`python`, `\`\`\`output`, `\boxed{}`), and creates a balanced 10K subset saved to `training/r1_tir_train.jsonl`.

**Format**: Each example is a chat conversation:
```json
{
  "messages": [
    {"role": "user", "content": "Problem...\n\nSolve this step-by-step..."},
    {"role": "assistant", "content": "<think>\nReasoning with ```python code``` and ```output result```\n</think>\n\n\\boxed{answer}"}
  ]
}
```

### Step 3: LoRA Fine-Tuning

```bash
python training/train_r1_lora.py
```

**Configuration**:
- Base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (bf16)
- LoRA: rank=32, alpha=64, targets=all linear layers
- Training: 1 epoch, batch=16, lr=2e-5, cosine schedule
- Duration: **14 minutes** on GH200

**Output**: LoRA adapter at `training/r1-tir-lora/` (1GB)

### Step 4: Merge LoRA into Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    torch_dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(model, "training/r1-tir-lora")
model = model.merge_and_unload()
model.save_pretrained("training/r1-tir-merged", safe_serialization=True)
```

**Output**: Merged model at `training/r1-tir-merged/` (62GB)

### Step 5: Local Evaluation

```bash
# Evaluate on IMO-level reference problems
python -u scripts/r1_tir_eval.py \
  --model training/r1-tir-merged \
  --samples 16 --rounds 5 --temp 0.6 \
  --max-tok 16384 --max-len 24576 --gpu-mem 0.92 \
  --data data/reference.csv \
  --out scripts/r1_lora_ref_results.csv

# Evaluate on AMC validation
python -u scripts/r1_tir_eval.py \
  --model training/r1-tir-merged \
  --samples 16 --rounds 5 --temp 0.6 \
  --max-tok 16384 --max-len 24576 --gpu-mem 0.92 \
  --data data/val_amc.csv \
  --out scripts/r1_lora_amc_results.csv
```

### Step 6: Kaggle Submission

```bash
# Upload merged model to Kaggle
kaggle datasets create -p training/r1-tir-merged/ --dir-mode zip

# Upload notebook
kaggle kernels push -p submission/kaggle-kernel/
```

---

## Kaggle Notebook Details

**File**: `submission/kaggle_r1_tir.ipynb`

The notebook auto-detects GPU and adjusts:

| Setting | H100 (80GB) | GH200 (96GB) |
|---------|-------------|--------------|
| Samples | 8 | 16 |
| Max tokens/round | 12,288 | 16,384 |
| Context window | 24,576 | 32,768 |
| GPU mem util | 0.90 | 0.92 |

**Model loading priority**:
1. `/kaggle/input/r1-tir-merged` (LoRA-merged, uploaded as dataset)
2. `/kaggle/input/r1-distill-qwen-32b` (base model fallback)
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (HF hub, requires internet)

**Safety features**:
- Context overflow protection (force-stop at max_model_len - 1024 tokens)
- Code output capped at 600 chars
- Forbidden operations blocked (subprocess, os.system, etc.)
- Force-answer extraction for candidates that never reach `\boxed{}`
- 10-second code execution timeout

---

## Results Summary

### IMO-Level (reference.csv, 10 problems)

| Config | Score | Key Wins |
|--------|-------|----------|
| Qwen-7B SC-TIR | 0/10 | — |
| R1-32B pure CoT (8 samples) | 2/10 | 26de63, 92ba6a |
| R1-32B + TIR (16 samples) | 4/10 | + 42d360, 9c1c5f |
| **R1-32B + LoRA (16 samples)** | **5/10** | **+ a295e9** |

### AMC Competition (val-amc, 20 problems)

| Config | Score |
|--------|-------|
| Qwen-7B SC-TIR | 13/20 (65%) |
| R1-32B pure CoT (8 samples) | 15/20 (75%) |
| Qwen-72B-AWQ (no code) | 11/20 (55%) |

---

## Cost Breakdown

| Activity | GPU Time | Cost ($1.99/hr) |
|----------|----------|-----------------|
| Environment setup + smoke tests | 1 hr | $2.00 |
| Model benchmarks (7B, 32B, 72B) | 2 hr | $3.98 |
| R1-TIR pipeline development | 1 hr | $1.99 |
| R1-TIR ref.csv eval (16 samples) | 2 hr | $3.98 |
| LoRA training (14 min) | 0.25 hr | $0.50 |
| LoRA model merge | 0.1 hr | $0.20 |
| LoRA eval on ref.csv | 2 hr | $3.98 |
| **Total** | **~8.5 hr** | **~$17** |

---

## File Structure

```
AIMO3/
├── BENCHMARK_RESULTS.md        # Detailed benchmark results
├── IMPLEMENTATION.md           # This file
├── py311_env/                  # Python 3.11 virtual environment
│
├── data/
│   ├── reference.csv           # 10 IMO-level reference problems
│   ├── val_amc.csv             # 83 AMC validation problems
│   └── kaggle_evaluation/      # Kaggle evaluation API
│
├── scripts/
│   ├── r1_tir_eval.py          # Primary evaluation pipeline
│   ├── sc_tir_eval.py          # Qwen-Math SC-TIR evaluation
│   ├── cot_eval.py             # Pure CoT evaluation
│   └── *_results.csv           # All benchmark results
│
├── training/
│   ├── prepare_tir_data.py     # Data preparation script
│   ├── train_r1_lora.py        # LoRA training script
│   ├── r1_tir_train.jsonl      # 10K training examples
│   ├── r1-tir-lora/            # LoRA adapter (1GB)
│   └── r1-tir-merged/          # Merged model (62GB)
│
└── submission/
    ├── kaggle_r1_tir.ipynb     # Kaggle submission notebook
    ├── kaggle_r1_tir.py        # Standalone Python version
    └── kaggle-kernel/          # Kernel metadata for kaggle push
```

---

## Reproducing Results

```bash
# Full pipeline from scratch (~$17, ~8.5 GPU-hours)
source py311_env/bin/activate

# 1. Prepare data (2 min, no GPU)
python training/prepare_tir_data.py

# 2. Train LoRA (14 min)
python -u training/train_r1_lora.py

# 3. Merge (30 sec, CPU only)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
m = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
m = PeftModel.from_pretrained(m, 'training/r1-tir-lora').merge_and_unload()
m.save_pretrained('training/r1-tir-merged', safe_serialization=True)
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', trust_remote_code=True).save_pretrained('training/r1-tir-merged')
"

# 4. Evaluate (2 hrs)
python -u scripts/r1_tir_eval.py \
  --model training/r1-tir-merged \
  --samples 16 --rounds 5 --temp 0.6 \
  --max-tok 16384 --max-len 24576 --gpu-mem 0.92 \
  --data data/reference.csv \
  --out scripts/r1_lora_ref_results.csv

# 5. Upload to Kaggle
kaggle datasets create -p training/r1-tir-merged/ --dir-mode zip
kaggle kernels push -p submission/kaggle-kernel/
```
