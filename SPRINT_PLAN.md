# AIMO 3 - 8-Week Sprint Plan
**Start: Feb 15, 2026 | Deadline: Apr 15, 2026**

---

## Current Status
- [x] Competition research complete
- [x] Documentation created
- [ ] Kaggle authentication pending
- [ ] No baseline submission yet
- [ ] No training data downloaded

---

## WEEK 1 (Feb 15-21) - Foundation Sprint

### Day 1-2: Environment Setup
```bash
# Priority 1: Kaggle Authentication
mkdir -p ~/.kaggle
# Download kaggle.json from kaggle.com/settings → API
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Priority 2: Download Competition Data
cd /Users/sabah/ai-content/AIMO/data
kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3
unzip ai-mathematical-olympiad-progress-prize-3.zip
```

### Day 2-3: Study Materials
- [ ] Read reference.csv (10 problems with solutions)
- [ ] Read AIMO3_Reference_Problems.pdf (full solutions)
- [ ] Run AIMO 3 Submission Demo notebook locally
- [ ] Understand problem difficulty and style

### Day 3-5: Baseline Submission
- [ ] Fork submission demo notebook on Kaggle
- [ ] Use existing open model (Qwen2.5-Math-7B or DeepSeekMath-7B)
- [ ] Submit to get initial leaderboard position
- [ ] Verify runtime < 5 hours

### Day 5-7: Resource Setup
- [ ] Apply for Fields Model Initiative (H100 access)
- [ ] Set up HuggingFace account for model downloads
- [ ] Configure local/cloud training environment
- [ ] Download base models for experimentation

---

## WEEK 2 (Feb 22-28) - Data Pipeline

### Training Data Download
```bash
# NuminaMath Datasets (proven to work)
huggingface-cli download AI-MO/NuminaMath-CoT --local-dir ./data/numina-cot
huggingface-cli download AI-MO/NuminaMath-TIR --local-dir ./data/numina-tir

# NVIDIA OpenMath (AIMO 2 winner data)
huggingface-cli download nvidia/OpenMathReasoning --local-dir ./data/openmath

# Validation Sets
huggingface-cli download AI-MO/aimo-validation-amc --local-dir ./data/val-amc
huggingface-cli download AI-MO/aimo-validation-aime --local-dir ./data/val-aime
huggingface-cli download AI-MO/aimo-validation-math-level-4 --local-dir ./data/val-math4
huggingface-cli download AI-MO/aimo-validation-math-level-5 --local-dir ./data/val-math5
```

### Data Processing Tasks
- [ ] Analyze NuminaMath-CoT format
- [ ] Analyze NuminaMath-TIR format (with code blocks)
- [ ] Create unified data loader
- [ ] Build validation evaluation script
- [ ] Run baseline model on validation sets (establish benchmarks)

---

## WEEK 3-4 (Mar 1-14) - Model Training

### Week 3: Stage 1 - CoT Fine-tuning
```python
# Training Config
base_model = "deepseek-ai/deepseek-math-7b-base"  # or Qwen2.5-Math
dataset = "NuminaMath-CoT"
epochs = 3
lr = 2e-5
batch_size = 32
max_length = 2048
```

- [ ] Set up training script (TRL SFTTrainer or Axolotl)
- [ ] Configure DeepSpeed ZeRO-3 for multi-GPU
- [ ] Run Stage 1 training (~10-20 hours on 8xH100)
- [ ] Evaluate on validation sets
- [ ] Save checkpoint

### Week 4: Stage 2 - TIR Fine-tuning
```python
# Training Config
base_model = "stage1_checkpoint"
dataset = "NuminaMath-TIR"
epochs = 4
lr = 2e-5
batch_size = 32
max_length = 1024
```

- [ ] Prepare TIR format data (code + execution results)
- [ ] Run Stage 2 training
- [ ] Evaluate on validation sets
- [ ] Compare CoT-only vs TIR model
- [ ] Submit trained model to leaderboard

---

## WEEK 5-6 (Mar 15-28) - Inference Pipeline

### SC-TIR Implementation
```python
# Core Algorithm Parameters
N_SAMPLES = 48      # Number of solution attempts
M_ITERATIONS = 4    # Code execution rounds per attempt
TEMPERATURE = 0.8
MAX_TOKENS = 2048
TIMEOUT = 5         # seconds per code block
```

### Implementation Tasks
- [ ] Build Python code executor (sandboxed)
- [ ] Implement multi-sample generation with vLLM
- [ ] Add code block extraction (regex: ```python...```)
- [ ] Add execution result concatenation
- [ ] Implement iterative refinement loop
- [ ] Build answer extraction (\boxed{} parsing)
- [ ] Implement majority voting
- [ ] Profile runtime (must fit in 5 hours for 50 problems)

### Optimization
- [ ] Test 8-bit quantization (AutoGPTQ)
- [ ] Test 4-bit quantization if needed
- [ ] Optimize batch sizes
- [ ] Reduce N_SAMPLES if runtime too long
- [ ] Submit optimized version

---

## WEEK 7 (Mar 29 - Apr 4) - Iteration

### Error Analysis
- [ ] Categorize failures by problem type (algebra/geometry/etc.)
- [ ] Identify systematic weaknesses
- [ ] Compare public leaderboard vs internal validation
- [ ] Find patterns in failed problems

### Targeted Improvements
- [ ] Add domain-specific data for weak areas
- [ ] Try different prompting strategies
- [ ] Experiment with larger models if compute allows
- [ ] Test ensemble of multiple models
- [ ] A/B test different hyperparameters

### Submissions
- [ ] Submit multiple variants
- [ ] Track which changes improve score
- [ ] Select best performing approach

---

## WEEK 8 (Apr 5-15) - Final Push

### Apr 5-8: Robustness (Before Entry Deadline)
- [ ] Stress test on all validation sets
- [ ] Verify runtime stability (multiple runs)
- [ ] Handle edge cases (empty output, timeout, etc.)
- [ ] Ensure deterministic behavior for private set reruns

### Apr 8: Entry Deadline
- [ ] Confirm team registration complete
- [ ] Verify all team members accepted rules

### Apr 9-14: Final Optimization
- [ ] Select best checkpoint
- [ ] Fine-tune inference parameters
- [ ] Clean up notebook code
- [ ] Run final validation

### Apr 15: Final Submission
- [ ] Submit before 11:59 PM UTC
- [ ] Verify submission successful
- [ ] Document approach for potential writeup

---

## Parallel Workstreams

### If Team Has Multiple Members:

| Person | Week 1-2 | Week 3-4 | Week 5-6 | Week 7-8 |
|--------|----------|----------|----------|----------|
| **A** | Setup + Baseline | Training | Inference | Final tuning |
| **B** | Data download | Data processing | Error analysis | Robustness |
| **C** | Research competition | Validation suite | Ablations | Documentation |

---

## Critical Path (Must Complete)

```
Week 1: Baseline on leaderboard ──┐
                                  │
Week 2: Training data ready ──────┼──► Week 3-4: Trained model
                                  │
Week 3-4: Model training ─────────┘
           │
           ▼
Week 5-6: SC-TIR inference working
           │
           ▼
Week 7: Error analysis + improvements
           │
           ▼
Week 8: Final submission
```

---

## Quick Wins (Do These First)

1. **Use existing public notebooks** - Don't reinvent the wheel
2. **Start with 7B model** - Faster iteration, scale up later
3. **Use NuminaMath datasets directly** - Proven to work
4. **Copy SC-TIR implementation** - From project-numina GitHub
5. **Submit early and often** - Learn from leaderboard feedback

---

## Commands to Run Now

```bash
# 1. Set up Kaggle (do this first)
mkdir -p ~/.kaggle
# [Download kaggle.json from kaggle.com/settings]
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Download competition data
mkdir -p /Users/sabah/ai-content/AIMO/data
cd /Users/sabah/ai-content/AIMO/data
kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3

# 3. Clone winning solution repo
cd /Users/sabah/ai-content/AIMO
git clone https://github.com/project-numina/aimo-progress-prize.git numina-solution

# 4. Install dependencies
pip install vllm transformers datasets accelerate kaggle huggingface_hub
```

---

## Success Metrics

| Week | Target |
|------|--------|
| 1 | Baseline submission on leaderboard |
| 2 | Validation pipeline working, internal benchmarks established |
| 4 | Trained model scores > baseline on validation |
| 6 | SC-TIR inference complete, submission < 4 hours runtime |
| 8 | Top-100 leaderboard (stretch: top-50) |
