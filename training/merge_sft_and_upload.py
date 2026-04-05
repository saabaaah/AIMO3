"""
Merge SFT-only LoRA into base and upload to Kaggle as a separate model.
"""
import os, json, subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CHECKPOINT = "./sft_checkpoint"
BASE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
MERGED_DIR = "./sft_final_model"
MODEL_SLUG = "aimo3-qwen-math-7b-sft-only"
MODEL_TITLE = "AIMO3 Qwen2.5-Math-7B SFT-only"
KAGGLE_USERNAME = "tantheta"

# Merge
if not (Path(MERGED_DIR).exists() and any(Path(MERGED_DIR).iterdir())):
    print("Loading base model in fp16...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, device_map="auto"
    )
    print(f"Loading SFT LoRA from {CHECKPOINT}...")
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    print("Merging...")
    merged = model.merge_and_unload()
    print(f"Saving to {MERGED_DIR}/...")
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(CHECKPOINT).save_pretrained(MERGED_DIR)
    del merged, model, base
    torch.cuda.empty_cache()
else:
    print(f"{MERGED_DIR}/ already exists, skipping merge.")

size_gb = sum(f.stat().st_size for f in Path(MERGED_DIR).rglob("*") if f.is_file()) / 1e9
print(f"Merged size: {size_gb:.1f} GB")

# Write Kaggle metadata
Path(MERGED_DIR, "model-metadata.json").write_text(json.dumps({
    "ownerSlug": KAGGLE_USERNAME,
    "title": MODEL_TITLE,
    "slug": MODEL_SLUG,
    "subtitle": "SFT-only fine-tune (no GRPO)",
    "isPrivate": True,
    "description": "Qwen2.5-Math-7B-Instruct + SFT on NuminaMath-TIR (72K). No GRPO.",
}, indent=2))
Path(MERGED_DIR, "model-instance-metadata.json").write_text(json.dumps({
    "ownerSlug": KAGGLE_USERNAME,
    "modelSlug": MODEL_SLUG,
    "instanceSlug": "default",
    "framework": "transformers",
    "overview": "SFT-only merged weights for AIMO3.",
    "usage": "Load with vLLM.",
    "licenseName": "Apache 2.0",
    "fineTunable": False,
    "trainingData": [],
}, indent=2))

# Create + upload
print("\nCreating Kaggle Model...")
r = subprocess.run(["kaggle", "models", "create", "-p", MERGED_DIR],
                   capture_output=True, text=True)
print(r.stdout); print(r.stderr)

print("\nUploading instance (this is the big 14GB upload)...")
r = subprocess.run(["kaggle", "models", "instances", "create", "-p", MERGED_DIR],
                   capture_output=True, text=True)
print(r.stdout); print(r.stderr)

print(f"\nDONE: https://www.kaggle.com/models/{KAGGLE_USERNAME}/{MODEL_SLUG}")
print(f"Kaggle path: /kaggle/input/models/{KAGGLE_USERNAME}/{MODEL_SLUG}/transformers/default/1")
