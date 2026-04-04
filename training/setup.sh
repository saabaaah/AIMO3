#!/bin/bash
# AIMO 3 — Lambda Labs setup script
# Run once: bash setup.sh

set -e
echo "=== AIMO 3 Lambda Setup ==="

# Core training stack (no Unsloth — avoids dependency hell)
pip install --upgrade torch torchvision
pip install --upgrade transformers accelerate peft bitsandbytes
pip install --upgrade trl==0.18.1
pip install --upgrade datasets huggingface_hub Pillow

echo ""
echo "=== Setup complete! ==="
echo "Next: nohup python sft_train.py > sft_train.log 2>&1 &"
