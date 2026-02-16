# AIMO 3 - Team Discussion Plan

## Meeting Agenda

### 1. Competition Overview (5 min)

**What is AIMO 3?**
- AI Mathematical Olympiad Progress Prize 3
- Goal: Build open-source AI that solves olympiad-level math problems
- Prize pool: **$2.2M+** (1st place: $262K, Overall Progress Prize: ~$1.6M)
- Deadline: **April 15, 2026** (~2 months remaining)

**Key Facts:**
- 110 original problems (National Olympiad → IMO difficulty)
- 50 public + 50 private test problems
- 5-digit answers (0-99999) - guessing is impossible
- Domains: Algebra, Combinatorics, Geometry, Number Theory

---

### 2. Past Winners Analysis (10 min)

| Competition | Winner | Score | Key Approach |
|-------------|--------|-------|--------------|
| AIMO 1 (Jul 2024) | NuminaMath | 29/50 | 7B model + SC-TIR |
| AIMO 2 (Apr 2025) | NVIDIA NemoSkills | 34/50 | 14B model + GenSelect |

**What worked:**
1. **Tool-Integrated Reasoning (TIR)** - Model generates Python code, executes it, uses output
2. **Self-Consistency** - Generate 32-64 solutions, majority vote on answer
3. **Math-specialized base models** - DeepSeekMath, Qwen-Math
4. **High-quality training data** - CoT + TIR formatted solutions

**AIMO 3 changes:**
- Harder problems (IMO level vs AIME level)
- 5-digit answers (vs 3-digit) - no guessing
- H100 GPUs available - can run larger models
- 5-hour GPU limit (vs 9-hour in AIMO 1)

---

### 3. Technical Architecture (15 min)

#### Submission Flow
```
┌─────────────────────────────────────────────────────────────┐
│  Kaggle Evaluation API                                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ for problem in iter_test():                             ││
│  │     answer = model.predict(problem)  # Your code here   ││
│  │     submit(answer)                                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

#### Recommended Architecture: SC-TIR
```
Input Problem (LaTeX)
        │
        ▼
┌───────────────────┐
│   LLM Generation  │ ← Generate N=48 diverse solutions
│   (with code)     │   Temperature=0.8, max_tokens=2048
└───────────────────┘
        │
        ▼ (for each solution)
┌───────────────────┐
│  Python Executor  │ ← Run code blocks, capture output
│  (sandboxed)      │   5-second timeout per block
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Continue/Retry   │ ← Append output, generate more
│  (M=4 iterations) │   Self-correction on errors
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Answer Extraction│ ← Parse \boxed{} answers
│  & Majority Vote  │   Filter invalid, vote on valid
└───────────────────┘
        │
        ▼
    Final Answer (0-99999)
```

#### Model Options (with H100)
| Model | Size | Pros | Cons |
|-------|------|------|------|
| DeepSeekMath-7B | 7B | Fast, proven | May lack capacity |
| Qwen2.5-Math-72B | 72B | Strong math | Slow inference |
| OpenMath-Nemotron-32B | 32B | AIMO 2 winner base | Requires fine-tuning |
| DeepSeekMath-67B | 67B | Strong reasoning | Memory intensive |

**Recommendation:** Start with 7B for rapid iteration, scale to 32B-72B for final submission.

---

### 4. Resource Assessment (10 min)

#### Questions to Discuss:
- [ ] What GPUs does our team have access to?
- [ ] Can we apply for Fields Model Initiative (128 H100s)?
- [ ] Do we have Kaggle accounts with GPU quota?
- [ ] What's our storage situation for training data?

#### Compute Estimates
| Task | GPU Hours (H100) |
|------|------------------|
| Data processing | 50-100 |
| Stage 1 training (CoT) | 200-400 |
| Stage 2 training (TIR) | 200-400 |
| Inference experimentation | 200 |
| Final tuning | 100 |
| **Total** | **750-1200 hours** |

#### External Resources (Applications Open)
1. **Fields Model Initiative** - Up to 128 H100s for fine-tuning
2. **Tinker API** - Up to $400 in credits
3. **Kaggle H100s** - Available for AIMO 3 notebooks (no internet)

---

### 5. Data Strategy (15 min)

#### Training Data Sources
| Source | Problems | Format |
|--------|----------|--------|
| MATH Dataset | 12.5K | Problems + solutions |
| AMC/AIME | ~500/year | Competition archives |
| IMO Shortlist | ~30/year | High difficulty |
| National Olympiads | Varies | Multiple countries |
| Synthetic (GPT-4/Claude) | Unlimited | Generate + verify |

#### Data Pipeline
```
Raw Problems → OCR/Clean → Format to CoT → Generate TIR → Verify Answers → Filter
```

#### Key Datasets to Acquire
- [ ] NuminaMath-CoT (~860K problems) - HuggingFace
- [ ] NuminaMath-TIR (~70K problems) - HuggingFace
- [ ] OpenMathReasoning (NVIDIA) - 540K problems
- [ ] AoPS forum problems
- [ ] Historical olympiad archives

---

### 6. Role Assignment (10 min)

#### Suggested Roles
| Role | Responsibilities | Skills Needed |
|------|------------------|---------------|
| **Data Lead** | Data collection, processing, quality control | Python, data wrangling |
| **Training Lead** | Model training, hyperparameter tuning | PyTorch, transformers, distributed training |
| **Inference Lead** | SC-TIR implementation, runtime optimization | vLLM, optimization |
| **Validation Lead** | Internal benchmarks, error analysis, ablations | Analysis, statistics |
| **Math Expert** | Problem categorization, solution verification | Math olympiad experience |

#### Discussion:
- Who takes which role?
- Any gaps we need to fill?
- Time commitment per person?

---

### 7. Timeline & Milestones (10 min)

| Week | Milestone | Owner |
|------|-----------|-------|
| **1** | Baseline submission on leaderboard | All |
| **2** | Kaggle credentials + data downloaded | - |
| **2** | Apply for compute resources | - |
| **3** | Training data pipeline complete | Data Lead |
| **4** | First trained model | Training Lead |
| **6** | SC-TIR inference working | Inference Lead |
| **8** | Internal validation suite | Validation Lead |
| **10** | Top-50 leaderboard | All |
| **12** | Ablation studies complete | Validation Lead |
| **14** | Final model selection | All |
| **16** | Final submission (Apr 15) | All |

#### Key Dates
- **Feb 2, 2026** - Longest Leader Prize period ends
- **Feb 9, 2026** - Math Corpus Prize deadline
- **Apr 8, 2026** - Entry & team merger deadline
- **Apr 15, 2026** - Final submission deadline

---

### 8. Risks & Mitigation (5 min)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient compute | High | Apply for Fields Initiative ASAP |
| Overfitting to public LB | High | Strong internal validation (AMC, AIME, MATH) |
| Runtime timeout (5h) | High | Profile early, quantize, optimize |
| Data quality issues | Medium | Verify answers, multiple sources |
| Team availability | Medium | Clear ownership, async communication |

---

### 9. Immediate Action Items (5 min)

#### This Week
- [ ] **Everyone**: Create/verify Kaggle account, accept competition rules
- [ ] **Everyone**: Download competition data (reference.csv, test.csv)
- [ ] **Data Lead**: Start downloading NuminaMath + OpenMathReasoning datasets
- [ ] **Training Lead**: Set up training environment, test on small scale
- [ ] **Inference Lead**: Run submission demo notebook, understand API
- [ ] **One person**: Apply for Fields Model Initiative compute

#### Before Next Meeting
- [ ] Each person reads AIMO 1 winner writeup: https://huggingface.co/blog/winning-aimo-progress-prize
- [ ] Each person reads AIMO 2 winner paper: https://arxiv.org/abs/2504.16891
- [ ] Baseline submission attempted

---

### 10. Open Discussion (10 min)

Questions to consider:
1. Do we aim for top-5 ($16K-$262K) or Overall Progress Prize ($1.6M)?
2. Should we pursue any additional prizes (Math Corpus, Writeup)?
3. What's our unique angle vs other teams?
4. Any math olympiad experience on the team?
5. When do we meet next?

---

## Appendix: Quick Reference

### Submission API Template
```python
import kaggle_evaluation

env = kaggle_evaluation.aimo.make_env()
for problem, _ in env.iter_test():
    # Your inference code here
    answer = model.predict(problem)  # Must return int 0-99999
    env.predict(answer)
```

### Scoring
- Public LB: Raw accuracy (correct/50)
- Private LB: Run twice, penalized accuracy
  - Both correct: 1.0
  - One correct: 0.5
  - Neither correct: 0.0

### Resources
- [Competition Page](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)
- [Submission Demo](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo)
- [NuminaMath Solution](https://huggingface.co/blog/winning-aimo-progress-prize)
- [NVIDIA NemoSkills Paper](https://arxiv.org/abs/2504.16891)
- [Project Numina GitHub](https://github.com/project-numina/aimo-progress-prize)
