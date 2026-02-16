# AIMO 3 Submission Package

## Quick Start

### Option 1: Submit on Kaggle (Recommended)

1. Go to [AIMO 3 Competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/code)
2. Click "New Notebook"
3. Copy contents of `kaggle_notebook.ipynb`
4. Enable GPU accelerator (H100 if available)
5. Disable Internet (required)
6. Submit

### Option 2: Test Locally

```bash
# Test the simple baseline (no GPU needed)
python test_local.py

# Test full solution (requires GPU + vLLM)
python aimo3_submission.py
```

## Files

| File | Description |
|------|-------------|
| `kaggle_notebook.ipynb` | Kaggle-ready notebook for submission |
| `aimo3_submission.py` | Full SC-TIR implementation |
| `test_local.py` | Quick local test (no GPU) |

## Solution Architecture: SC-TIR

**S**elf-**C**onsistency with **T**ool-**I**ntegrated **R**easoning

```
Problem (LaTeX)
     │
     ▼
┌─────────────────┐
│ Generate N=32   │  ← Multiple solution attempts
│ candidates      │    Temperature=0.7
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Execute Python  │  ← Run code blocks
│ code blocks     │    5-second timeout
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Iterate M=4     │  ← Self-correction
│ times           │    Append outputs
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Extract \boxed  │  ← Parse answers
│ answers         │    Validate 0-99999
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Majority Vote   │  ← Most common answer
└─────────────────┘
     │
     ▼
  Final Answer
```

## Configuration

Edit `Config` class in the submission files:

```python
@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    num_samples: int = 32        # More = better but slower
    num_generations: int = 4     # Code execution rounds
    temperature: float = 0.7     # Higher = more diverse
    max_new_tokens: int = 2048
```

## Model Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| Qwen2.5-Math-7B-Instruct | 7B | Fast | Good |
| Qwen2.5-Math-72B-Instruct | 72B | Slow | Better |
| DeepSeekMath-7B-Instruct | 7B | Fast | Good |
| AI-MO/NuminaMath-7B-TIR | 7B | Fast | Proven |

## Runtime Budget

- Kaggle GPU limit: **5 hours** for 50 problems
- Target: **6 minutes per problem**
- With N=32, M=4: ~3-4 minutes per problem on H100

## Key Differences from AIMO 1/2

| Aspect | AIMO 1/2 | AIMO 3 |
|--------|----------|--------|
| Answers | 0-999 (mod 1000) | 0-99999 |
| GPU | T4/L4 | H100 |
| Runtime | 9 hours | 5 hours |
| Problems | 50 | 50 public + 50 private |

## Next Steps

1. **Submit baseline** - Get on leaderboard today
2. **Fine-tune model** - Use NuminaMath datasets
3. **Optimize inference** - Increase samples, tune temperature
4. **Validate** - Test on AMC/AIME validation sets

## Resources

- [NuminaMath Solution](https://huggingface.co/blog/winning-aimo-progress-prize)
- [NVIDIA NemoSkills Paper](https://arxiv.org/abs/2504.16891)
- [AIMO 3 Submission Demo](https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo)
