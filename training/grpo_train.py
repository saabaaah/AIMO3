"""
AIMO 3 — Stage 2: GRPO Training
==================================
RL fine-tuning using binary math rewards (correct=1, wrong=0).
Loads from SFT checkpoint, trains on AIME+AMC problems.

Usage:
    nohup python grpo_train.py > grpo_train.log 2>&1 &
    tail -f grpo_train.log

Resumes automatically if a checkpoint exists in ./grpo_checkpoint/
Requires: ./sft_checkpoint/ from Stage 1
Output: ./grpo_checkpoint/ (LoRA adapters with RL updates)
Time: ~2-3 hours on A100 40GB
"""

import os
import torch
from datasets import load_dataset, concatenate_datasets, Value
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ============================================
# CONFIG — edit these if needed
# ============================================
SFT_CHECKPOINT = "./sft_checkpoint"    # from Stage 1
OUTPUT_DIR = "./grpo_checkpoint"
MAX_SEQ_LENGTH = 2048
NUM_GENERATIONS = 8       # solutions per problem
MAX_COMPLETION = 1024     # tokens per solution (prompt + this < 2048)
MAX_PROMPT = 512          # truncate long prompts
BATCH_SIZE = 2            # per device (1 × 2 × 4 = 8 must be divisible by NUM_GENERATIONS)
GRAD_ACCUM = 4
LEARNING_RATE = 5e-6      # 10x lower than SFT for RL stability
NUM_EPOCHS = 3
SAVE_STEPS = 50
LOGGING_STEPS = 1

# ============================================
# GPU CHECK
# ============================================
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================
# LOAD MODEL FROM SFT CHECKPOINT
# ============================================
if not os.path.exists(SFT_CHECKPOINT):
    raise FileNotFoundError(
        f"SFT checkpoint not found at {SFT_CHECKPOINT}\n"
        "Run sft_train.py first."
    )

print(f"\nLoading model from {SFT_CHECKPOINT}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_CHECKPOINT,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
print(f"Model loaded: {model.num_parameters():,} parameters")

# ============================================
# REWARD FUNCTION
# ============================================
def extract_boxed_answer(text):
    """Extract content from \\boxed{...} in model output."""
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None
    i, depth = idx, 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                start = text.index("{", idx) + 1
                return text[start:i].strip()
        i += 1
    return None


def normalize_answer(text):
    """Clean up answer string for comparison."""
    if not text:
        return ""
    for remove in [r"\text{", "}", r"\mathrm{", "$", ",", " ", "%"]:
        text = text.replace(remove, "")
    return text.strip()


def correctness_reward(prompts, completions, answer, **kwargs):
    """
    GRPO reward: 1.0 if correct, 0.0 if wrong.

    TRL passes completions as chat message lists:
      [[{"role": "assistant", "content": "..."}], ...]
    """
    rewards = []
    for completion, true_answer in zip(completions, answer):
        # Extract text from chat message format
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        extracted = extract_boxed_answer(text)
        normalized = normalize_answer(extracted) if extracted else ""
        is_correct = (normalized == str(true_answer).strip())
        rewards.append(1.0 if is_correct else 0.0)

    return rewards

# ============================================
# LOAD & FORMAT GRPO DATA
# ============================================
print("\nLoading AIME + AMC datasets...")
ds_aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
ds_amc = load_dataset("AI-MO/aimo-validation-amc", split="train")
print(f"AIME: {len(ds_aime)} problems, AMC: {len(ds_amc)} problems")


def make_grpo_prompt(example):
    """Format a problem as a chat prompt for GRPO."""
    problem = example["problem"]
    prompt_text = (
        "Solve this math problem step by step. "
        "Use Python code when calculations are needed.\n"
        "Put your final numerical answer in \\boxed{}.\n\n"
        f"Problem: {problem}"
    )
    answer = str(int(float(example["answer"]))) if example["answer"] is not None else "0"
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "answer": answer,
    }


grpo_aime = ds_aime.map(make_grpo_prompt, remove_columns=ds_aime.column_names)
grpo_amc = ds_amc.map(make_grpo_prompt, remove_columns=ds_amc.column_names)

# Force matching column types for concatenation
grpo_aime = grpo_aime.cast_column("answer", Value("string"))
grpo_amc = grpo_amc.cast_column("answer", Value("string"))

grpo_dataset = concatenate_datasets([grpo_aime, grpo_amc])
print(f"GRPO dataset: {len(grpo_dataset)} problems × {NUM_GENERATIONS} rollouts = {len(grpo_dataset) * NUM_GENERATIONS} samples/epoch")

# ============================================
# GRPO TRAINING
# ============================================
FastLanguageModel.for_training(model)

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION,
    max_prompt_length=MAX_PROMPT,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=NUM_EPOCHS,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    use_vllm=False,
    seed=42,
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=[correctness_reward],
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)

# Auto-resume from checkpoint
resume = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        resume = os.path.join(OUTPUT_DIR, latest)
        print(f"\n>>> Resuming from {resume}")

print(f"\nStarting GRPO training...")
print(f"  Problems: {len(grpo_dataset)}")
print(f"  Generations per problem: {NUM_GENERATIONS}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Saving every {SAVE_STEPS} steps to {OUTPUT_DIR}/")

stats = grpo_trainer.train(resume_from_checkpoint=resume)

print(f"\n{'='*50}")
print(f"GRPO COMPLETE!")
print(f"Final loss: {stats.training_loss:.4f}")
print(f"{'='*50}")

# Save final checkpoint
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}/")
print(f"\nNext: run upload_model.py to push to HuggingFace")
