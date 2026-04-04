"""
AIMO 3 — Stage 1: SFT Training
================================
Fine-tune Qwen2.5-Math-7B on NuminaMath-TIR (72K problems with code solutions).

Usage:
    nohup python sft_train.py > sft_train.log 2>&1 &
    tail -f sft_train.log

Resumes automatically if a checkpoint exists in ./sft_checkpoint/
Output: ./sft_checkpoint/ (LoRA adapters, ~200MB)
Time: ~2-3 hours on A100 40GB
"""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ============================================
# CONFIG — edit these if needed
# ============================================
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR = "./sft_checkpoint"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8          # effective batch = 2 * 8 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
SAVE_STEPS = 500
LOGGING_STEPS = 25

# ============================================
# GPU CHECK
# ============================================
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================
# LOAD MODEL + LoRA
# ============================================
print(f"\nLoading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # auto-detect (bf16 on A100/H100)
)

# Add LoRA adapters (Unsloth applies these automatically if not already present)
model = FastLanguageModel.get_peft_model(
    model,
    r=64,                    # LoRA rank — higher = more capacity
    lora_alpha=16,           # scaling factor
    lora_dropout=0,          # Unsloth optimized — keep at 0
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # saves 60% more VRAM
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)")

# ============================================
# LOAD & FORMAT DATA
# ============================================
print("\nLoading NuminaMath-TIR dataset...")
ds_tir = load_dataset("AI-MO/NuminaMath-TIR", split="train")
print(f"Training examples: {len(ds_tir)}")

def format_for_sft(example):
    """Convert chat messages to model's template string."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds_formatted = ds_tir.map(format_for_sft, num_proc=4)
print(f"Formatted {len(ds_formatted)} examples")

# ============================================
# SFT TRAINING
# ============================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=NUM_EPOCHS,
    bf16=True,
    fp16=False,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
    padding_free=False,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_formatted,
    args=sft_config,
)

# Auto-resume from checkpoint if one exists
resume = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        resume = os.path.join(OUTPUT_DIR, latest)
        print(f"\n>>> Resuming from {resume}")

total_steps = len(ds_formatted) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nStarting SFT training...")
print(f"  Steps: ~{total_steps}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Saving every {SAVE_STEPS} steps to {OUTPUT_DIR}/")

stats = trainer.train(resume_from_checkpoint=resume)

print(f"\n{'='*50}")
print(f"SFT COMPLETE!")
print(f"Final loss: {stats.training_loss:.4f}")
print(f"{'='*50}")

# Save final checkpoint
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}/")
print(f"\nNext: run grpo_train.py")
