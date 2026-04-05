"""
AIMO 3 — Stage 1: SFT Training (no Unsloth)
=============================================
Fine-tune Qwen2.5-Math-7B on NuminaMath-TIR (72K problems).
Uses transformers + peft + trl directly.

Usage:
    nohup python sft_train.py > sft_train.log 2>&1 &
    tail -f sft_train.log

Resumes automatically if a checkpoint exists.
Output: ./sft_checkpoint/ (LoRA adapters, ~200MB)
Time: ~3-4 hours on A100 40GB
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ============================================
# CONFIG
# ============================================
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR = "./sft_checkpoint"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 4          # effective batch = 4 * 4 = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
SAVE_STEPS = 500
LOGGING_STEPS = 25

# ============================================
# GPU CHECK
# ============================================
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# LOAD MODEL in 4-bit + LoRA
# ============================================
print(f"\nLoading {MODEL_NAME} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare for LoRA training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

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
    dataset_text_field="text",
    packing=False,
    seed=42,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
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

# Save final LoRA adapters
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}/")
print(f"\nNext: python grpo_train.py")
