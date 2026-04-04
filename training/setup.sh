#!/bin/bash
# AIMO 3 — Lambda Labs setup script
# Run once after SSH-ing in:
#   bash setup.sh

set -e

echo "=== AIMO 3 Lambda Setup ==="

# Fix torchvision to match torch version
pip install --upgrade torchvision

# Install training dependencies
pip install unsloth
pip install trl==0.18.1
pip install vllm
pip install huggingface_hub

echo ""
echo "=== Setup complete! ==="
echo "Next: nohup python sft_train.py > sft_train.log 2>&1 &"
