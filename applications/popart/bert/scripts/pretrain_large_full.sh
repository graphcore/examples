#!/bin/sh

# Pre-train with sequence length 128
python3 bert.py --config configs/pretrain_large_128.json \
    --checkpoint-dir ckpts/pretrain_large_128 2>&1 | tee pretrain_large_128_log.txt

# Get the timestamped directory from the most recent run
PHASE1_DIR=$(ls ckpts/pretrain_large_128 -1 | tail -n 1)

# Load checkpoint and train with sequence length 384
python3 bert.py --config configs/pretrain_large_384.json \
    --onnx-checkpoint ckpts/pretrain_large_128/$PHASE1_DIR/model.onnx \
    --checkpoint-dir ckpts/pretrain_large_384 2>&1 | tee pretrain_large_384_log.txt

# Final pre-training result will be in ckpts/pretrain_large_384
