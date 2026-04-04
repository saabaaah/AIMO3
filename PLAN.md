# AIMO 3 — Execution Plan

**Deadline:** April 15, 2026 (11 days remaining)
**Budget:** ~$900 Lambda credit (target spend: ~$15-20)

---

## Current Status

| Item | Status |
|------|--------|
| SC-TIR submission notebook | Done (kaggle_notebook_v2.ipynb) |
| SFT training script | Done (training/sft_train.py) |
| GRPO training script | Done (training/grpo_train.py) |
| Model upload script | Done (training/upload_model.py) |
| SFT training run | Not started (lost previous checkpoints) |
| GRPO training run | Not started |
| First Kaggle submission | Not done |

---

## Plan

### Phase 1: Train on Lambda (~$10, ~6 hours)

**Instance:** 1x A100 40GB ($1.48/hr)

```bash
# SSH into Lambda instance
ssh <lambda-host>

# Setup
git clone <your-repo> && cd AIMO3/training
pip install unsloth trl==0.18.1 vllm huggingface_hub
huggingface-cli login

# Stage 1: SFT (~2-3 hours, ~$4)
nohup python sft_train.py > sft_train.log 2>&1 &
tail -f sft_train.log

# Stage 2: GRPO (~2-3 hours, ~$4)
nohup python grpo_train.py > grpo_train.log 2>&1 &
tail -f grpo_train.log

# Upload to HuggingFace
python upload_model.py
```

**What to watch:**
- SFT loss: should drop from ~2.0 → ~0.5-0.8
- GRPO reward/mean: should increase from ~0.2-0.4 → ~0.5-0.7

**If Lambda disconnects:** SSH back in, re-run same command. Both scripts auto-resume from last checkpoint.

### Phase 2: Submit to Kaggle (free)

1. Open `submission/kaggle_notebook_v2.ipynb` on Kaggle
2. Add Input → Models → search your HuggingFace repo
3. Update model path in the notebook to point to your trained model
4. Set GPU accelerator (T4 or P100 for free tier)
5. Turn OFF internet
6. Submit

### Phase 3: Iterate (if time allows)

Ideas to improve score:
- Increase GRPO epochs (3 → 5)
- Add NuminaMath-TIR test split to GRPO data
- Try larger model (Qwen2.5-Math-72B) if budget allows
- Tune num_samples in submission (16 → 32)
- Add code execution in GRPO reward (execute code blocks, check output)

---

## Files

```
training/
  sft_train.py       — Stage 1: SFT on NuminaMath-TIR (72K problems)
  grpo_train.py      — Stage 2: GRPO on AIME+AMC (173 problems)
  upload_model.py    — Merge LoRA + upload to HuggingFace

submission/
  kaggle_notebook_v2.ipynb — SC-TIR inference notebook for Kaggle
  aimo3_submission.py      — Core SC-TIR implementation
```
