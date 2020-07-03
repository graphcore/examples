#!/bin/sh

# Pre-train with sequence length 128
python3 bert.py --config configs/pretrain_base_128.json \
    --checkpoint-dir ckpts/pretrain_base_128 2>&1 | tee pretrain_base_128_log.txt

# Get the timestamped directory from the most recent run
PHASE1_DIR=$(ls ckpts/pretrain_base_128 -1 | tail -n 1)

# Load checkpoint and train with sequence length 384
python3 bert.py --config configs/pretrain_base_384.json \
    --onnx-checkpoint ckpts/pretrain_base_128/$PHASE1_DIR/model.onnx \
    --checkpoint-dir ckpts/pretrain_base_384 2>&1 | tee pretrain_base_384_log.txt

# Final pre-training result will be in ckpts/pretrain_base_384
