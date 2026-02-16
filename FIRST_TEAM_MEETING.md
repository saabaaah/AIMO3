# AIMO 3 - First Team Meeting

## What Is This?
**AI Mathematical Olympiad** - Build an AI that solves IMO-level math problems.
- **Prize**: $2.2M total (1st place: $262K)
- **Deadline**: April 15, 2026 (8 weeks)
- **Format**: Submit code on Kaggle, AI solves 50 math problems

---

## What's Ready (Done Before This Meeting)

| Item | Status |
|------|--------|
| Competition data | ✅ Downloaded |
| Training data | ✅ 932K math problems |
| Submission code | ✅ Ready to submit |
| Winning solutions studied | ✅ AIMO 1 & 2 |

---

## The Approach (SC-TIR)

```
Math Problem → LLM generates solution with Python code →
Execute code → Repeat 4x → Extract answer → Majority vote
```

This approach won AIMO 1 (29/50) and AIMO 2 (34/50).

---

## Discussion Points

### 1. Team Name Ideas?
- MathMinds
- OlympiadAI
- [Your ideas]

### 2. Role Assignment
| Role | Who? |
|------|------|
| Training Lead | ? |
| Inference Lead | ? |
| Data/Validation | ? |

### 3. Key Decisions
- **Goal**: Top 5 ($16K-$262K) or shoot for 47/50+ ($1.6M)?
- **Model**: Use 7B (fast) or 72B (better)?
- **Compute**: Apply for free H100s from Fields Initiative?

### 4. Immediate Next Steps
1. Submit baseline this week (code is ready)
2. Get on leaderboard, see where we stand
3. Assign roles
4. Schedule next meeting

---

## Timeline

| Week | Goal |
|------|------|
| 1 | Baseline submission |
| 2-4 | Fine-tune model |
| 5-6 | Optimize inference |
| 7-8 | Final push |

---

## Resources Location

All files in: `/Users/sabah/ai-content/AIMO/`

```
submission/           ← Ready-to-submit notebooks
datasets/             ← 932K training problems
challenge-details/    ← Competition docs
SPRINT_PLAN.md        ← 8-week execution plan
```

---

## One-Liner Summary

> We're competing in a $2.2M AI math competition.
> Code is ready, data is downloaded, we need to submit our first baseline and decide our strategy.
