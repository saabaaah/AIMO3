# AIMO 3 — Updated Strategy Plan
**Updated: Feb 22, 2026 | Deadline: Apr 15, 2026 | Remaining: ~8 weeks**

---

## Hardware Summary

| Resource | Spec | Use |
|----------|------|-----|
| Mac Studio (local) | M3 Ultra, 512GB unified memory | Training, distillation data generation, local eval |
| Kaggle | H100 (80GB VRAM), 5-hour runtime | Final inference, competition submission |

Key implication: 72B models fit locally in bfloat16 (~144GB). QLoRA on 72B fits easily (~80-120GB). This is the primary advantage over most teams.

---

## Baseline Reality Check

**Current measured baseline (val-aime, 35 problems):**
- Model: Qwen2.5-Math-7B-Instruct
- Settings: 4 samples, 2 generation steps, 1500 max tokens
- Result: **3/35 = 8.6%**

This is far below expected (~25-35% for 7B with proper settings). The low score is due to:
1. `num_samples=4` — self-consistency needs 16+ samples to work
2. Answer extraction may be failing on MPS-generated text
3. `max_new_tokens=1500` may be truncating long AIME solutions

**First priority is fixing this and establishing real baselines.**

---

## Data Assets

| Source | Count | Quality | Status |
|--------|-------|---------|--------|
| NuminaMath-CoT | 859K problems | High (diverse, CoT format) | Downloaded |
| NuminaMath-TIR | 72K problems | High (code+execution format) | Downloaded |
| AoPS AIME/AMC (with answers) | 561 problems | Very high (verified integer answers) | Processed |
| AoPS (no answer extracted) | 152 problems | Medium | Needs re-extraction |
| Contest PDFs (CEMC, SMT, etc.) | ~3,400 records | Low (answers missing) | Partially parsed |
| PDF Books (~538 books) | Unknown | Low (noisy) | Not worth processing |

**Training data priority order**: NuminaMath-TIR > AoPS AIME/AMC > NuminaMath-CoT subset

---

## Strategy Overview

Two parallel tracks:

```
PRIMARY TRACK (Distillation)                 SECONDARY TRACK (72B LoRA)
─────────────────────────────                ──────────────────────────
72B teacher generates solutions              QLoRA on 72B with AoPS data
         │                                            │
Filter to correct answers only               Evaluate on val-aime
         │                                            │
Fine-tune 7B student (LoRA)                  Merge adapter → 4-bit quant
         │                                            │
Fast 7B inference on Kaggle                  72B inference on Kaggle
(16-32 samples, ~30 min for 50 problems)     (8 samples, fits in 5h)
         │                                            │
         └─────────── Compare on val-aime ───────────┘
                      Submit best one
```

---

## Week-by-Week Plan

### WEEK 1 (Feb 22–28) — Fix Baseline + Infrastructure

**Goal: Know our real starting point. Every experiment after this is relative to it.**

#### Task 1.1: Fix the 7B baseline in test_draft.py
Change config to proper settings:
- `num_samples = 16` (minimum for self-consistency)
- `num_generations = 4` (full SC-TIR depth)
- `max_new_tokens = 2048`
- Run full val-aime (90 problems, not just 35)
- Expected result: ~25-35%

#### Task 1.2: Set up mlx-lm for 72B local inference
```bash
pip install mlx-lm
# Download 72B in 4-bit (fits in ~36GB, fast inference)
mlx_lm.convert --hf-path Qwen/Qwen2.5-Math-72B-Instruct \
    --mlx-path ./models/Qwen2.5-Math-72B-4bit -q --q-bits 4
```
Then run a small val-aime eval (10-20 problems) with 72B to establish ceiling.

#### Task 1.3: Measure generation throughput
- 72B at 4-bit on M3 Ultra: estimate tokens/sec
- 7B at bfloat16: estimate tokens/sec
- Use these numbers to plan distillation timeline

#### Task 1.4: Fix AoPS answer extraction gaps
- 152 AoPS problems have no answer extracted (method=no_solution or no_match)
- Re-examine these: some may have answers embedded differently
- Goal: push from 561 → 650+ verified problems

**Week 1 deliverable**: Baselines confirmed, mlx-lm running 72B, throughput measured.

---

### WEEKS 2–3 (Mar 1–14) — Distillation Data Generation

**Goal: Generate 10K–20K high-quality (problem, teacher_solution) training pairs.**

#### How distillation works here
1. Pick problems from NuminaMath-TIR (72K, already TIR-formatted with answers)
2. Generate solutions using 72B via mlx-lm with temperature=0.8
3. Extract answer from each generated solution (`\boxed{}`)
4. **Keep only solutions where predicted answer == ground truth answer**
5. Store as: `{problem, solution_trace, final_answer}`

This is rejection sampling — the teacher may fail 40-60% of problems, but the surviving solutions are high-quality.

#### Target dataset size
- Sample 30K problems from NuminaMath-TIR
- Expected ~50% pass rate → ~15K clean examples
- Supplement with all 561 AoPS verified problems

#### Throughput estimate (72B at 4-bit on M3 Ultra)
- Estimated: ~30-60 tokens/sec at 4-bit
- Average solution: ~600 tokens
- 30K problems × 600 tokens = 18M tokens ÷ 45 tok/sec ≈ **4-5 days continuous**
- Run overnight/background — feasible in 2 weeks with daytime pauses

#### Diversity strategy
- Sample NuminaMath-TIR problems uniformly across sources (AIME, AMC, MATH, Olympiad)
- Use temperature=0.8 (matches competition inference setting)
- Do NOT generate multiple solutions per problem at this stage — one is enough

**Week 2-3 deliverable**: `data/distillation/train.jsonl` with 12K-18K verified examples.

---

### WEEK 4 (Mar 15–21) — 7B Student Fine-tuning (SFT)

**Goal: Fine-tune Qwen2.5-Math-7B-Instruct on distilled data. Target >35% on val-aime.**

#### Tool: mlx-lm LoRA (best for M3 Ultra)
```bash
mlx_lm.lora \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --train \
  --data ./data/distillation \
  --batch-size 4 \
  --lora-layers 16 \
  --num-iterations 5000 \
  --learning-rate 2e-5 \
  --val-batches 50 \
  --save-every 500 \
  --adapter-path ./adapters/7b-distilled
```

#### Training estimate (7B bfloat16, M3 Ultra)
- ~15K examples × 600 tokens = 9M tokens
- Estimated throughput: ~300-500 tok/sec training
- ~5 hours per epoch — run 2-3 epochs overnight

#### Evaluation loop
After each checkpoint:
```bash
python test_draft.py --adapter ./adapters/7b-distilled/500 --limit 30
```
Stop when val-aime accuracy stops improving.

#### Expected result
- Base 7B: ~25-35%
- Distilled 7B: ~35-45%
- If not improving: check data quality, increase dataset diversity

**Week 4 deliverable**: Fine-tuned 7B adapter that beats base 7B on val-aime.

---

### WEEK 5 (Mar 22–28) — 72B Fine-tuning (QLoRA, Optional)

**Goal: Fine-tune 72B on AoPS AIME/AMC data. Higher ceiling than distilled 7B.**

This is the secondary track. Only start if Week 4 is on schedule.

#### Why a small dataset for 72B
- 72B already knows most math — don't want to overfit or forget
- AoPS AIME problems (561 examples) are domain-matched to AIMO 3
- Small, high-quality dataset + low learning rate = safe fine-tuning

#### Tool: mlx-lm LoRA on 72B (4-bit base, fp32 adapters)
```bash
mlx_lm.lora \
  --model ./models/Qwen2.5-Math-72B-4bit \
  --train \
  --data ./data/aops_aime \
  --batch-size 1 \
  --lora-layers 8 \
  --num-iterations 1000 \
  --learning-rate 5e-6 \
  --adapter-path ./adapters/72b-aops
```

#### Training estimate (72B 4-bit)
- 561 examples × 800 tokens = ~450K tokens
- ~50-80 tok/sec training
- ~2-3 hours per epoch — very fast, can do 5+ epochs

#### Risk mitigation
- Evaluate on val-aime after every 200 iterations
- If accuracy drops below base 72B: stop, use smaller lr or fewer lora-layers
- Keep base 72B as fallback

**Week 5 deliverable**: QLoRA-adapted 72B that beats base 72B on val-aime (or confirms base is better).

---

### WEEKS 6–7 (Mar 29 – Apr 11) — Kaggle Notebook + Inference Optimization

**Goal: Working Kaggle submission with best model. Measure real runtime.**

#### Step 6.1: Merge and export adapter
```bash
# Merge LoRA weights into base model
mlx_lm.fuse --model Qwen/Qwen2.5-Math-7B-Instruct \
    --adapter-path ./adapters/7b-distilled \
    --save-path ./models/7b-distilled-merged

# Convert to HuggingFace format for vLLM on Kaggle
mlx_lm.convert --mlx-path ./models/7b-distilled-merged \
    --hf-path ./models/7b-distilled-hf
```
Upload merged model to HuggingFace (private repo) for Kaggle to pull.

#### Step 6.2: Inference hyperparameter sweep (on Mac, not Kaggle quota)
Test locally on val-aime with mlx-lm:
```
num_samples: [8, 16, 32]
temperature: [0.6, 0.8, 1.0]
max_new_tokens: [1024, 2048]
num_generations: [2, 4]
```
Pick best combination. Then verify runtime fits in 5 hours on Kaggle (estimate from throughput).

#### Step 6.3: Kaggle notebook update
- Update `submission/kaggle_notebook_72b.ipynb` with best model + settings
- Verify `IS_SUBMISSION` flag logic works correctly
- Test locally using `submission/test_local.py`
- Submit to leaderboard and record score

#### Step 6.4: Runtime budget (50 problems, 5 hours)
```
Available: 300 minutes
Per problem: 6 minutes max

At 100 tok/sec (72B 4-bit vLLM on H100):
  8 samples × 4 generations × 500 tokens = 16K tokens
  16K ÷ 100 tok/sec = 160 seconds = 2.7 min → SAFE

At 200 tok/sec (7B bfloat16 vLLM on H100):
  32 samples × 4 generations × 500 tokens = 64K tokens
  64K ÷ 200 tok/sec = 320 seconds = 5.3 min → TIGHT, use 16 samples
```

**Week 6-7 deliverable**: Live Kaggle submission, score on public leaderboard.

---

### WEEK 8 (Apr 8–15) — Final Push

#### Apr 8: Entry deadline — confirm team registration

#### Apr 9-12: Final model selection
- Compare val-aime scores: base 7B vs distilled 7B vs base 72B vs fine-tuned 72B
- Select top 2 to submit (get 2 daily submissions)
- Consider ensemble only if both models fit within 4 hours combined

#### Apr 13-14: Robustness checks
- Run val-aime 3 times with same config → verify stable accuracy (±2%)
- Check edge cases: problems with no code blocks, problems with very long solutions
- Verify answer always in 0-99999 range

#### Apr 15: Submit before 11:59 PM UTC

---

## Decision Tree: Which Model to Submit

```
val-aime accuracy comparison:

Fine-tuned 72B > Base 72B?
  YES → Submit fine-tuned 72B (secondary track wins)
  NO  → Use base 72B

Distilled 7B > Base 7B by >5%?
  YES → Consider 7B (faster → more samples → better self-consistency)
  NO  → Skip distillation track

32 samples × 7B vs 8 samples × 72B → which is higher on val-aime?
  → Submit that one
```

---

## Reinforcement Learning (Post-SFT, if time permits)

RL is high-risk on MPS. Only attempt after Week 5 if everything is on track.

**Approach**: GRPO-style with binary reward (correct answer = 1, wrong = 0)
- Use AIME 2000-2024 (AoPS, integer answers 0-999 — easily verified)
- Generate 8 rollouts per problem, compute group-relative advantage
- Update with PPO clipping

**Framework options on MPS** (none are ideal):
1. `trl` GRPO — CUDA-dependent, needs patching for MPS
2. Custom loop: generate with `mlx-lm`, score on CPU, update via `mlx` optimizer
3. Use a cloud GPU (Lambda Labs, RunPod) for RL only

**If RL shows +5% on val-aime after 500 steps → keep. Otherwise abort.**

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Distillation yield too low (<30%) | Medium | High | Use larger NuminaMath sample, lower temperature |
| 72B fine-tune degrades accuracy | Medium | Medium | Use smaller lr, fewer LoRA layers; base model is fallback |
| Kaggle runtime exceeds 5h | Medium | High | Profile early (Week 6), reduce samples |
| mlx-lm → vLLM adapter incompatibility | Low | High | Test on Kaggle in Week 6, not Week 8 |
| val-aime overfitting (tuning to val set) | Medium | Medium | Use val-amc as secondary holdout |
| MPS NaN issues during training | Low | Medium | Use bfloat16 + SanitizeLogits pattern (already in code) |

---

## Immediate Next Actions (This Week)

```
[ ] 1. Fix test_draft.py config: num_samples=16, num_generations=4, max_new_tokens=2048
[ ] 2. Run full val-aime (90 problems) with fixed 7B config → get real baseline
[ ] 3. pip install mlx-lm
[ ] 4. Download/convert Qwen2.5-Math-72B to 4-bit MLX format
[ ] 5. Run 10-problem val-aime sample with 72B → get 72B baseline + throughput
[ ] 6. Measure tok/sec for both 7B and 72B → validate distillation timeline
[ ] 7. Write distillation data generation script
```

---

## Success Metrics

| Week | Metric | Target |
|------|--------|--------|
| 1 | 7B baseline (val-aime, 90 problems, 16 samples) | ≥25% |
| 1 | 72B baseline (val-aime, 20 problems) | ≥45% |
| 3 | Distillation dataset size | ≥10K verified examples |
| 4 | Distilled 7B (val-aime) | ≥35% |
| 5 | Fine-tuned 72B (val-aime) | ≥50% |
| 6 | Live Kaggle submission | Score on public LB |
| 8 | Final submission | Top-100 (stretch: top-50) |
