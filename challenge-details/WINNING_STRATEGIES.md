# AIMO - Past Winning Strategies & Approaches

## AIMO 1 Winner: NuminaMath (29/50)
Project Numina, July 2024

### Model Architecture
- **Base Model**: DeepSeekMath-Base 7B
- **Why 7B?**: Better inference speed on T4 GPUs than larger models (20B, 33B)
- **Quantization**: 8-bit (AutoGPTQ) for faster upload and reduced VRAM

### Two-Stage Training

#### Stage 1: Chain of Thought (CoT) Fine-tuning
```
Dataset: ~100k+ problems with CoT solutions
Sources: Chinese high school math, AMC, AIME, IMO problems
Processing: OCR -> Segmentation -> Translation -> CoT realignment -> Answer formatting

Hyperparameters:
- Learning rate: 2.0E-5
- Batch size: 32
- Block size: 2048 tokens
- Epochs: 3
- Scheduler: cosine
- Warmup: 0

Result: 56.3% on MATH benchmark
```

#### Stage 2: Tool-Integrated Reasoning (TIR)
```
Dataset: 60k problems with TORA-format solutions
Generated: GPT-4 created multi-step reasoning with Python code
Filtered: Correct answers only, 3x iterations for consistency

Hyperparameters:
- Learning rate: 2.0E-5
- Batch size: 32
- Block size: 1024 tokens
- Epochs: 4
- Scheduler: cosine
- Warmup: 0.1

Result: 68.2% on MATH benchmark
```

### Inference: SC-TIR Algorithm
Self-Consistency with Tool-Integrated Reasoning

```
Algorithm:
1. Copy input N times (N=48 in winning submission)
2. Sample N diverse completions until Python code block
3. Execute each Python block, append output/tracebacks
4. Repeat M times (M=4) for self-correction
5. Prune incomplete/failed samples
6. Majority voting on final answers

Benefits:
- Reduces variance from random problem ordering
- Enables self-correction via execution feedback
- More robust than simple majority voting
```

### Validation Strategy
| Dataset | Size | Purpose |
|---------|------|---------|
| AMC (2022-2023) | 83 | Representative difficulty |
| AIME (2022-2024) | 90 | Harder problems |
| MATH Level 4 | 754 | Large-scale eval |
| MATH Level 5 | 721 | Large-scale eval |

### What Didn't Work
- KTO: Only +3% improvement
- RLOO (RL): No significant gains
- Model merging (DARE, TIES, WARP): Led to regressions
- Larger models: Too slow on T4 GPUs
- Static KV cache + torch.compile: Errors on Kaggle

---

## AIMO 2 Winner: NemoSkills/NVIDIA (34/50)
April 2025

### Three-Pillar Approach

#### 1. Large-Scale Dataset
- 540K unique high-quality math problems (including olympiad-level)
- 3.2M long-reasoning solutions

#### 2. Tool-Integrated Reasoning
- Novel code execution + long reasoning integration
- Iterative: training -> generation -> quality filtering
- 1.7M high-quality TIR solutions

#### 3. Generative Solution Selection (GenSelect)
- Train models to select most promising solution from candidates
- Significantly outperforms majority voting

### Model
- **OpenMath-Nemotron-14B-Kaggle**: 14.8B parameters
- Fine-tuned on strategically selected subset of OpenMathReasoning dataset
- Combines natural language reasoning + Python code execution

### Hardware
- 4x L4 GPUs
- Solved 34/50 problems in 5 hours

### Resources
- [AIMO-2 Winning Solution Paper (arXiv)](https://arxiv.org/abs/2504.16891)
- [OpenMath-Nemotron Models](https://huggingface.co/nvidia)

---

## Key Success Patterns

### Data Quality
1. High-quality, curated training data
2. Multi-source datasets (competitions, textbooks, synthetic)
3. CoT + TIR format solutions
4. GPT-4 for solution generation and verification

### Model Architecture
1. Math-specialized base models (DeepSeekMath, etc.)
2. Tool-Integrated Reasoning capability
3. Code execution feedback loop
4. Quantization for inference speed

### Inference Strategies
1. Self-consistency with multiple samples
2. Code execution for verification
3. Majority voting on final answers
4. Solution selection/ranking models

### Validation
1. Multiple internal validation sets
2. Avoid overfitting to small public leaderboard
3. Track variance across seeds

---

## AIMO 3 Considerations

### Changes from Previous Competitions
| Aspect | AIMO 1/2 | AIMO 3 |
|--------|----------|--------|
| Answers | 3-digit (0-999) | 5-digit (0-99999) |
| Problems | 50 | 110 (50 public + 50 private + 10 reference) |
| Difficulty | Mixed | National Olympiad to IMO level |
| Hardware | T4/L4 | H100 available |
| Runtime | 9 hours | 5 hours (GPU) / 9 hours (CPU) |

### Implications
1. **5-digit answers**: Guessing is virtually impossible (1/100,000 vs 1/1000)
2. **Higher difficulty**: Need stronger reasoning, not just pattern matching
3. **H100 access**: Can run larger models (20B+) with better inference
4. **Original problems**: Zero contamination advantage - must generalize

### Recommended Approach
1. Use larger, more capable base models (H100 enables this)
2. Strong Tool-Integrated Reasoning
3. High-quality training data (synthetic + real olympiad)
4. Robust validation across multiple difficulty levels
5. Solution selection/ranking for final answer

## Sources
- [NuminaMath Winning Solution](https://huggingface.co/blog/winning-aimo-progress-prize)
- [NVIDIA NemoSkills Paper](https://arxiv.org/abs/2504.16891)
- [NVIDIA Math AI Blog](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)
