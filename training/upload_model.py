"""
AIMO 3 — Upload trained model to HuggingFace
=============================================
Merges LoRA adapters into base model and uploads to HuggingFace.
Then attach this model to your Kaggle submission notebook.

Usage:
    # First login to HuggingFace:
    huggingface-cli login

    # Then run:
    python upload_model.py

Requires: ./grpo_checkpoint/ from Stage 2 (or ./sft_checkpoint/ for SFT-only)
"""

import os
import torch
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

# ============================================
# CONFIG — CHANGE THESE
# ============================================
CHECKPOINT = "./grpo_checkpoint"       # or "./sft_checkpoint" for SFT-only
REPO_ID = "YOUR_USERNAME/aimo3-qwen-math-7b-sft-grpo"  # <-- CHANGE THIS
MERGED_DIR = "./final_model"

# ============================================
# LOAD & MERGE
# ============================================
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

print(f"Loading model from {CHECKPOINT}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

print("Merging LoRA adapters into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}/...")
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Saved! Size: {sum(f.stat().st_size for f in __import__('pathlib').Path(MERGED_DIR).rglob('*') if f.is_file()) / 1e9:.1f} GB")

# ============================================
# UPLOAD TO HUGGINGFACE
# ============================================
print(f"\nUploading to {REPO_ID}...")
api = HfApi()
api.create_repo(REPO_ID, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path=MERGED_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"\n{'='*50}")
print(f"UPLOAD COMPLETE!")
print(f"Model: https://huggingface.co/{REPO_ID}")
print(f"{'='*50}")
print(f"\nNext steps:")
print(f"1. Go to Kaggle submission notebook")
print(f"2. Add Input → Models → search '{REPO_ID}'")
print(f"3. Update model path in the notebook")
print(f"4. Submit!")
