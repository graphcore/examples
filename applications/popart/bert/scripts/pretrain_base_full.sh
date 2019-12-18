#!/bin/sh

# Pre-train phase 1 (3 epochs at sequence length 128):
python bert.py --config configs/pretrain_base.json \
    --checkpoint-dir checkpoints/phase1 2>&1 | tee pretrain_phase1_log.txt

# Load checkpoint and run phase 2 (6 epochs at sequence length 384):
python bert.py --config configs/pretrain_base_384.json \
    --onnx-checkpoint checkpoints/phase1/model.onnx \
    --checkpoint-dir checkpoints/phase2 2>&1 | tee pretrain_phase2_log.txt

# Final pre-training result will be in checkpoints/phase2/model.onnx
