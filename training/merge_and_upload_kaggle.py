"""
AIMO 3 — Merge LoRA + Upload to Kaggle as a Model
==================================================
1. Loads base Qwen2.5-Math-7B + merges GRPO LoRA adapters
2. Saves merged model to ./final_model/
3. Uploads to Kaggle as a private Kaggle Model

Usage:
    # one-time setup:
    #   place ~/.kaggle/kaggle.json (chmod 600), pip install kaggle
    python merge_and_upload_kaggle.py
"""

import os
import json
import subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================
# CONFIG — EDIT THESE
# ============================================
CHECKPOINT = "./grpo_checkpoint"
BASE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
MERGED_DIR = "./final_model"

# Your Kaggle username (from ~/.kaggle/kaggle.json)
KAGGLE_USERNAME = None  # auto-read from kaggle.json if None
MODEL_SLUG = "aimo3-qwen-math-7b-sft-grpo"
MODEL_TITLE = "AIMO3 Qwen2.5-Math-7B SFT+GRPO"
FRAMEWORK = "transformers"

# ============================================
# STEP 1: MERGE LORA INTO BASE
# ============================================
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

if os.path.exists(MERGED_DIR) and any(Path(MERGED_DIR).iterdir()):
    print(f"Merged model already exists at {MERGED_DIR}/, skipping merge.")
else:
    print("Loading base model in fp16 (for clean merge)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapters from {CHECKPOINT}...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT)

    print("Merging LoRA into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}/...")
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.save_pretrained(MERGED_DIR)

    del merged, model, base_model
    torch.cuda.empty_cache()

size_gb = sum(f.stat().st_size for f in Path(MERGED_DIR).rglob("*") if f.is_file()) / 1e9
print(f"Merged model size: {size_gb:.1f} GB")

# ============================================
# STEP 2: UPLOAD TO KAGGLE
# ============================================
# Read username from kaggle.json if not set
if KAGGLE_USERNAME is None:
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if not kj.exists():
        raise FileNotFoundError(
            "~/.kaggle/kaggle.json not found. Download it from "
            "https://www.kaggle.com/settings/account"
        )
    KAGGLE_USERNAME = json.loads(kj.read_text())["username"]

print(f"\nKaggle user: {KAGGLE_USERNAME}")
print(f"Uploading as: {KAGGLE_USERNAME}/{MODEL_SLUG}")

# Write Kaggle model metadata
metadata = {
    "title": MODEL_TITLE,
    "id": f"{KAGGLE_USERNAME}/{MODEL_SLUG}",
    "subtitle": "Fine-tuned Qwen2.5-Math-7B for AIMO3",
    "description": "Qwen2.5-Math-7B-Instruct fine-tuned on NuminaMath-TIR (SFT) and GRPO on AIME+AMC.",
    "isPrivate": True,
    "licenses": [{"name": "apache-2.0"}],
    "frameworks": [FRAMEWORK],
    "task_ids": ["question-answering"],
}
meta_path = Path(MERGED_DIR) / "model-metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"Wrote metadata: {meta_path}")

# Create + upload model (first run creates it, subsequent runs create a new version)
print("\nCreating/uploading Kaggle Model (this will take a while)...")
# Try create first, ignore if exists, then push a new version
subprocess.run(
    ["kaggle", "models", "init", "-p", MERGED_DIR],
    check=False,  # metadata already written
)
r = subprocess.run(
    ["kaggle", "models", "create", "-p", MERGED_DIR],
    capture_output=True, text=True,
)
if r.returncode != 0 and "already exists" not in (r.stderr + r.stdout).lower():
    print("CREATE stdout:", r.stdout)
    print("CREATE stderr:", r.stderr)

# Init a model instance (variant) + push files
# Kaggle Models require: owner/model-slug/framework/variation/version
variation = "default"
instance_meta = {
    "id": f"{KAGGLE_USERNAME}/{MODEL_SLUG}/{FRAMEWORK}/{variation}",
    "versionNotes": "Initial upload: SFT + GRPO on Qwen2.5-Math-7B",
    "framework": FRAMEWORK,
}
(Path(MERGED_DIR) / "model-instance-metadata.json").write_text(json.dumps(instance_meta, indent=2))

r = subprocess.run(
    ["kaggle", "models", "instances", "create", "-p", MERGED_DIR],
    capture_output=True, text=True,
)
print("INSTANCE stdout:", r.stdout)
print("INSTANCE stderr:", r.stderr)

if r.returncode != 0 and "already exists" in (r.stderr + r.stdout).lower():
    # Push as a new version instead
    r = subprocess.run(
        ["kaggle", "models", "instances", "versions", "create",
         f"{KAGGLE_USERNAME}/{MODEL_SLUG}/{FRAMEWORK}/{variation}",
         "-p", MERGED_DIR, "-n", "Updated version"],
        capture_output=True, text=True,
    )
    print("VERSION stdout:", r.stdout)
    print("VERSION stderr:", r.stderr)

print(f"\n{'='*50}")
print("DONE!")
print(f"Model: https://www.kaggle.com/models/{KAGGLE_USERNAME}/{MODEL_SLUG}")
print(f"{'='*50}")
print("\nIn your Kaggle notebook:")
print("  Add Input → Models → search your model → attach")
print("  Then load with:")
print(f'    model_path = "/kaggle/input/{MODEL_SLUG}/transformers/{variation}/1"')
print('    from vllm import LLM; llm = LLM(model=model_path, dtype="bfloat16")')
