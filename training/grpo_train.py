"""
AIMO 3 — Stage 2: GRPO Training (no Unsloth)
==============================================
RL fine-tuning using binary math rewards (correct=1, wrong=0).
Uses transformers + peft + trl directly.

Usage:
    nohup python grpo_train.py > grpo_train.log 2>&1 &
    tail -f grpo_train.log

Resumes automatically if a checkpoint exists.
Requires: ./sft_checkpoint/ from Stage 1
Output: ./grpo_checkpoint/
Time: ~2-3 hours on A100 40GB
"""

import os
import torch
from datasets import load_dataset, concatenate_datasets, Value
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

# ============================================
# CONFIG
# ============================================
SFT_CHECKPOINT = "./sft_checkpoint"
OUTPUT_DIR = "./grpo_checkpoint"
BASE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
MAX_SEQ_LENGTH = 2048
NUM_GENERATIONS = 4
MAX_COMPLETION = 768
MAX_PROMPT = 512
BATCH_SIZE = 2            # 2 × 2 = 4 divisible by NUM_GENERATIONS=4
GRAD_ACCUM = 2
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1
SAVE_STEPS = 50
LOGGING_STEPS = 1

# ============================================
# GPU CHECK
# ============================================
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# LOAD MODEL FROM SFT CHECKPOINT
# ============================================
if not os.path.exists(SFT_CHECKPOINT):
    raise FileNotFoundError(f"SFT checkpoint not found at {SFT_CHECKPOINT}\nRun sft_train.py first.")

print(f"\nLoading base model + SFT LoRA from {SFT_CHECKPOINT}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

# Load LoRA adapters on top
model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT, is_trainable=True)

tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
    """GRPO reward: 1.0 if correct, 0.0 if wrong."""
    rewards = []
    for completion, true_answer in zip(completions, answer):
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
grpo_aime = grpo_aime.cast_column("answer", Value("string"))
grpo_amc = grpo_amc.cast_column("answer", Value("string"))
grpo_dataset = concatenate_datasets([grpo_aime, grpo_amc])
print(f"GRPO dataset: {len(grpo_dataset)} problems × {NUM_GENERATIONS} rollouts = {len(grpo_dataset) * NUM_GENERATIONS} samples/epoch")

# ============================================
# GRPO TRAINING
# ============================================
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION,
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
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    seed=42,
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=[correctness_reward],
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)

# Auto-resume
resume = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        resume = os.path.join(OUTPUT_DIR, latest)
        print(f"\n>>> Resuming from {resume}")

print(f"\nStarting GRPO training...")
print(f"  Problems: {len(grpo_dataset)}, Generations: {NUM_GENERATIONS}, Epochs: {NUM_EPOCHS}")
print(f"  Saving every {SAVE_STEPS} steps to {OUTPUT_DIR}/")

stats = grpo_trainer.train(resume_from_checkpoint=resume)

print(f"\n{'='*50}")
print(f"GRPO COMPLETE!")
print(f"Final loss: {stats.training_loss:.4f}")
print(f"{'='*50}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}/")
print(f"\nNext: python upload_model.py")
