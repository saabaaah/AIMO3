"""
LoRA SFT on DeepSeek-R1-Distill-Qwen-32B with NuminaMath-TIR data.

Teaches R1 to integrate Python code execution into its reasoning.
Uses PEFT LoRA + trl SFTTrainer on single GH200 (96GB).

Memory budget:
  - Model bf16: 61 GB
  - LoRA adapters (rank 32): ~2 GB
  - Optimizer states (AdamW, LoRA only): ~4 GB
  - Activations (grad checkpointing): ~10 GB
  - Total: ~77 GB (fits in 96GB)
"""
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ============= Config =============

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DATA_PATH = "/home/ubuntu/AIMO3/training/r1_tir_train.jsonl"
OUTPUT_DIR = "/home/ubuntu/AIMO3/training/r1-tir-lora"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 1
GRAD_ACCUM = 16           # effective batch = 16
LR = 2e-5
EPOCHS = 2
MAX_SEQ_LEN = 2048        # truncate long examples to save memory
WARMUP_RATIO = 0.03
SAVE_STEPS = 200
LOGGING_STEPS = 10


def load_data(path: str) -> Dataset:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Apply chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data from {DATA_PATH}...")
    dataset = load_data(DATA_PATH)
    print(f"  {len(dataset)} examples")

    # Format with chat template
    dataset = dataset.map(lambda x: format_chat(x, tokenizer), remove_columns=["messages"])
    print(f"  Formatted. Sample length: {len(dataset[0]['text'])} chars")

    print(f"Loading model {MODEL_ID} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Setting up SFTTrainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=True,          # pack short examples together
        report_to="none",      # no wandb for now
        seed=42,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps per epoch: ~{len(dataset) // (BATCH_SIZE * GRAD_ACCUM)}")
    print(f"  Max seq len: {MAX_SEQ_LEN}")

    trainer.train()

    print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done!")
    print(f"To merge: model.merge_and_unload() or load with PeftModel.from_pretrained()")


if __name__ == "__main__":
    main()
