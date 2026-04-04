"""
AIMO 3 — Upload trained model to HuggingFace
=============================================
Merges LoRA adapters into base model and uploads.

Usage:
    huggingface-cli login   # first time only
    python upload_model.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi

# ============================================
# CONFIG — CHANGE THESE
# ============================================
CHECKPOINT = "./grpo_checkpoint"       # or "./sft_checkpoint" for SFT-only
BASE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
REPO_ID = "YOUR_USERNAME/aimo3-qwen-math-7b-sft-grpo"  # <-- CHANGE THIS
MERGED_DIR = "./final_model"

# ============================================
# LOAD & MERGE
# ============================================
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

print(f"Loading base model...")
# Load in float16 for merging (not 4-bit — we want full precision weights)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Loading LoRA from {CHECKPOINT}...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT)

print("Merging LoRA into base model...")
merged = model.merge_and_unload()

print(f"Saving to {MERGED_DIR}/...")
merged.save_pretrained(MERGED_DIR)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.save_pretrained(MERGED_DIR)

size_gb = sum(f.stat().st_size for f in __import__('pathlib').Path(MERGED_DIR).rglob('*') if f.is_file()) / 1e9
print(f"Saved! Size: {size_gb:.1f} GB")

# ============================================
# UPLOAD
# ============================================
print(f"\nUploading to {REPO_ID}...")
api = HfApi()
api.create_repo(REPO_ID, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=MERGED_DIR, repo_id=REPO_ID, repo_type="model")

print(f"\n{'='*50}")
print(f"DONE! Model: https://huggingface.co/{REPO_ID}")
print(f"{'='*50}")
print(f"\nNext: Kaggle → Add Input → Models → '{REPO_ID}' → Submit")
