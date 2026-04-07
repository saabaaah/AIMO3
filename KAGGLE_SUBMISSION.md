# Kaggle Submission Guide — AIMO3

## Prerequisites on Kaggle

| Item | URL | Status |
|------|-----|--------|
| Notebook | kaggle.com/code/tantheta/aimo3-r1-tir-submission | v3 pushed |
| Model dataset | kaggle.com/datasets/tantheta/r1-tir-merged | v2 (2-epoch LoRA) |
| LoRA adapter | kaggle.com/models/tantheta/r1-distill-qwen-32b-tir-lora | uploaded |

---

## Step-by-Step Submission

### 1. Rename the Notebook
- Open: https://www.kaggle.com/code/tantheta/aimo3-r1-tir-submission
- Click the title at the top (or File > Rename)
- Rename to: **`submission.parquet`**
- This exact name is required by the competition

### 2. Check Inputs are Attached
In the right sidebar under **Input**, verify these are listed:
- `tantheta/r1-tir-merged` (your fine-tuned model, 62GB)
- `ai-mathematical-olympiad-progress-prize-3` (competition data)

If missing, click **Add Input** > search and add them.

### 3. Check Session Options
In the right sidebar under **Session options**:
- **Accelerator**: `GPU H100` (CRITICAL — won't work without GPU)
- **Language**: Python
- **Internet**: Off (required for competition)
- **Environment**: Pin to original environment (2026-03-20)

### 4. Check Dependency Manager
Settings > Dependency Manager should have:
```
pip install -q vllm transformers polars
```
(No `!` prefix — plain pip command)

### 5. Submit
- Click **Submit** button (top right)
- Confirm GPU H100 is selected in the submission dialog
- Wait ~4-5 hours for rerun to complete

---

## What Happens During Rerun

1. **0-2 min**: Kaggle installs dependencies from Dependency Manager
2. **2-5 min**: Notebook cells execute, model loads lazily
3. **5 min**: `inference_server.serve()` starts, gateway connects
4. **5 min - 5 hrs**: Gateway sends 50 problems one at a time
   - Each problem: R1 generates 4 candidates × up to 12K tokens
   - Code blocks executed if present
   - Majority vote → answer returned
   - Per-problem timeout: 6 min max
5. **End**: Gateway writes `submission.parquet`, scoring happens

---

## Current Configuration (H100, 80GB)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | R1-Distill-32B + LoRA (2-epoch) | Best accuracy on IMO problems |
| dtype | bfloat16 | Full precision, H100 supports it |
| num_samples | 4 | Fewer samples, longer context per sample |
| max_tokens_per_round | 12,288 | Deep reasoning per candidate |
| max_model_len | 16,384 | Context window for R1's long thinking |
| gpu_mem_util | 0.95 | Maximize KV cache on H100 |
| temperature | 0.6 | Moderate diversity |
| num_rounds | 3 | Code execution rounds |
| per_problem_timeout | 360s (6 min) | 50 × 6 = 5 hrs budget |
| cutoff_time | 4h 45m | Stop 15 min before time limit |

---

## Troubleshooting

### "Submission file not found: submission.parquet"
- Notebook name must be exactly `submission.parquet`
- OR the model failed to load (check GPU is H100, dataset is attached)

### "Notebook Threw Exception"
- Usually protobuf `MessageFactory` error — fixed in v3 with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
- Or model OOM — reduce `max_model_len` or `gpu_mem_util`

### "Notebook Timeout"
- Rerun exceeded 5 hours — reduce `PER_PROBLEM_TIMEOUT` from 360 to 300

### "Notebook Exceeded Allowed Compute"
- GPU memory exceeded — reduce `gpu_mem_util` from 0.95 to 0.92

### "Notebook Inference Server Never Started"
- `serve()` wasn't called within 15 min — model loading is lazy so this shouldn't happen
- Check if `KAGGLE_IS_COMPETITION_RERUN` env var is set during rerun (it is, by Kaggle)

### Score = 0
- Check GPU was actually assigned (not P100/None)
- Check model loaded from dataset, not from HF hub (no internet during rerun)

### Low Score (1-2)
- Normal for first attempts with 4 samples on H100
- Increase `num_samples` (needs more KV cache → reduce `max_model_len`)
- Or use a quantized model to free VRAM

---

## Improving Score

### Quick Wins (code changes only)
- Increase `num_samples` from 4 to 8, reduce `max_model_len` from 16384 to 10240
- Add multiple prompt templates (like demo notebook 2 uses 5 different system prompts)
- Tune `temperature` (try 0.7-0.8 for more diversity)

### Medium Effort (needs GH200)
- Train 3-4 epochs of LoRA (currently 2)
- Use larger training set (currently 10K of 72K available)
- Train on competition-similar problems if available

### High Effort
- Quantize model to AWQ 4-bit → fits 72B on H100
- Ensemble R1-32B + Qwen-Math-7B
- GRPO training with answer-correctness reward

---

## Model Path Resolution on Kaggle

The notebook checks these paths in order:
```python
"/kaggle/input/datasets/tantheta/r1-tir-merged"   # Dataset input
"/kaggle/input/r1-tir-merged"                       # Alt dataset path
"/kaggle/input/datasets/tantheta/r1-distill-qwen-32b"  # Base model fallback
"/kaggle/input/r1-distill-qwen-32b"                 # Alt base path
"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"         # HF hub (needs internet)
```

During competition rerun (no internet), the HF hub fallback won't work. Make sure the dataset is attached.

---

## Updating the Model

If you train a better model on GH200:

```bash
# 1. Merge new LoRA
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
m = AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
m = PeftModel.from_pretrained(m, 'training/r1-tir-lora').merge_and_unload()
m.save_pretrained('training/r1-tir-merged-new', safe_serialization=True)
AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', trust_remote_code=True).save_pretrained('training/r1-tir-merged-new')
"

# 2. Upload as new version
echo '{"title":"r1-tir-merged","id":"tantheta/r1-tir-merged","licenses":[{"name":"Apache 2.0"}]}' > training/r1-tir-merged-new/dataset-metadata.json
kaggle datasets version -p training/r1-tir-merged-new/ -m "description of changes" --dir-mode zip

# 3. Push notebook (no changes needed — same dataset name)
kaggle kernels push -p submission/kaggle-kernel/
```

---

## Files Reference

| Local File | Purpose |
|------------|---------|
| `submission/kaggle_r1_tir.ipynb` | Kaggle notebook (source of truth) |
| `submission/kaggle-kernel/kernel-metadata.json` | Kaggle kernel push metadata |
| `training/r1-tir-merged-e2/` | 2-epoch merged model (62GB, uploaded) |
| `training/r1-tir-lora/` | LoRA adapter (1GB, uploaded as Kaggle Model) |
| `training/train_r1_lora.py` | Training script |
| `training/prepare_tir_data.py` | Data preparation script |
