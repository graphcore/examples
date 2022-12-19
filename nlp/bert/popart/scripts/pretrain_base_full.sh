#!/bin/sh

# Pre-train with sequence length 128
python3 bert.py --config configs/pretrain_base_128.json \
    --checkpoint-output-dir ckpts/pretrain_base_128 $@ 2>&1 | tee pretrain_base_128_log.txt

# Get the timestamped directory from the most recent run
PHASE1_DIR=$(ls ckpts/pretrain_base_128 -1 | tail -n 1)

# Load checkpoint and train with sequence length 512
python3 bert.py --config configs/pretrain_base_512.json \
    --checkpoint-input-dir ckpts/pretrain_base_128/$PHASE1_DIR/model.onnx \
    --checkpoint-output-dir ckpts/pretrain_base_512 $@ 2>&1 | tee pretrain_base_512_log.txt

# Final pre-training result will be in ckpts/pretrain_base_512
