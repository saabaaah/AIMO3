# CLAUDE.md - AIMO Competition Context

## What is this Project?

This is a competition codebase for the **AI Mathematical Olympiad - Progress Prize 3 (AIMO 3)**, a Kaggle competition with a **$2.2M+ prize pool**. The goal is to build an AI system that solves International Mathematical Olympiad (IMO)-level math problems.

**Deadline**: April 15, 2026
**Scoring**: Penalized accuracy on 50 hidden problems (answers are integers 0-99999)

## Project Structure

```
AIMO/
├── challenge-details/          # Competition documentation
│   ├── COMPETITION_OVERVIEW.md
│   ├── EVALUATION.md
│   ├── EXECUTION_PLAN.md
│   ├── TECHNICAL_REQUIREMENTS.md
│   └── WINNING_STRATEGIES.md
├── data/                       # Competition data
│   ├── reference.csv           # 10 reference problems with answers
│   ├── test.csv                # Placeholder test problems
│   ├── sample_submission.csv   # Submission format
│   └── kaggle_evaluation/      # Official evaluation API
├── datasets/                   # Training/validation data (932K+ problems)
│   ├── numina-cot/             # Chain-of-Thought (859K problems)
│   ├── numina-tir/             # Tool-Integrated Reasoning (72K problems)
│   ├── val-amc/                # AMC validation (83 problems)
│   └── val-aime/               # AIME validation (90 problems)
├── numina-solution/            # AIMO 1 winner code (Project Numina)
│   ├── kaggle-solution.ipynb   # Original winning notebook
│   └── training/               # SFT training scripts & configs
└── submission/                 # Ready-to-submit Kaggle notebooks
    ├── aimo3_submission.py     # Full SC-TIR implementation
    ├── kaggle_notebook.ipynb   # 7B model (vLLM)
    ├── kaggle_notebook_72b.ipynb # 72B model (H100)
    └── kaggle_notebook_t4.ipynb  # T4 GPU version
```

## Core Algorithm: SC-TIR

**Self-Consistency with Tool-Integrated Reasoning** - the approach that won AIMO 1 and 2:

1. **Problem Input**: Math problem as LaTeX string
2. **Parallel Generation**: Generate N solution candidates (N=16-32)
3. **Code Execution**: Each candidate uses Python code blocks; executed via `PythonREPL` class with 5-10s timeout
4. **Iterative Refinement**: Code output appended to context, allowing M=4 generate-execute loops
5. **Answer Extraction**: Parse `\boxed{}` answers using brace-counting
6. **Majority Voting**: Most common answer wins

## Models

| Variant | Model | Backend | GPU | Samples |
|---------|-------|---------|-----|---------|
| Default | Qwen2.5-Math-7B-Instruct | vLLM | Any | 32 |
| Large | Qwen2.5-Math-72B-Instruct | vLLM+bf16 | H100 | 16 |
| T4 | Qwen2.5-Math-7B-Instruct | Transformers | T4 | 32 |

## Training Pipeline (from numina-solution/)

**Stage 1**: CoT fine-tuning on NuminaMath-CoT (859K problems)
- Base: `deepseek-ai/deepseek-math-7b-base`
- 3 epochs, lr=2e-5, DeepSpeed ZeRO-3

**Stage 2**: TIR fine-tuning on NuminaMath-TIR (72K problems)
- Base: Stage 1 checkpoint
- 4 epochs, lr=2e-5

**Post-training**: AutoGPTQ quantization to 8-bit

## Key Files to Know

- `submission/aimo3_submission.py` - Main SC-TIR implementation with `PythonREPL`, answer extraction, majority voting
- `numina-solution/training/sft.py` - SFT training script
- `numina-solution/training/configs/*.yaml` - Training configurations
- `data/kaggle_evaluation/aimo_3_gateway.py` - Competition API interface

## Evaluation

- **Runtime**: 5 hours GPU / 9 hours CPU for 50 problems
- **Answer format**: Integer 0-99999 (no automatic modulo)
- **Scoring**: Both runs correct = 1.0, one = 0.5, neither = 0.0; max = 50

## Dependencies

**Inference**:
- `vllm` - Fast LLM serving
- `transformers` - Model loading
- `torch`, `pandas`, `sympy`, `numpy`

**Training**:
- `trl==0.8.1` (SFTTrainer)
- `accelerate`, `deepspeed`
- `auto_gptq` (quantization)
- `wandb` (experiment tracking)

## Commands

```bash
# Test local submission (no GPU)
python submission/test_local.py

# Run validation benchmark
python -c "from submission.aimo3_submission import *; benchmark_on_validation()"

# Start Kaggle notebook locally
jupyter notebook submission/kaggle_notebook.ipynb
```

## Competition History

| Competition | Winner | Score | Approach |
|-------------|--------|-------|----------|
| AIMO 1 | NuminaMath | 29/50 | SC-TIR with 7B model |
| AIMO 2 | NVIDIA | 34/50 | Improved prompting + 72B |

## Important Constraints

1. **No internet** during Kaggle evaluation
2. **Code execution safety**: Block dangerous operations (`subprocess`, `os.system`, `eval`, `exec`, `__import__`, `open`)
3. **Memory limits**: T4 has 16GB VRAM, H100 has 80GB
4. **Time limits**: ~6 minutes per problem average

## Development Guidelines

- Test all changes against `val-amc/` and `val-aime/` datasets before submission
- Use `reference.csv` for quick sanity checks
- Monitor accuracy AND runtime when making changes
- Log all experiment results with configurations

## Team Context

- Project lead: Sabah
- Sprint plan: 8 weeks (Feb 15 - Apr 15, 2026)
- See `SPRINT_PLAN.md` and `FIRST_TEAM_MEETING.md` for details
