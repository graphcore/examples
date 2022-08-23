#!/bin/bash
# run gpt2-large on POD16
python train_gpt2.py \
    --model gpt2-large \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 2 2 2 2 2 2 2 2 3 3 3 3 3 3 2 \
    --matmul-proportion 0.2 0.15 0.2 0.2 0.2 0.15 0.15 0.2 0.2 0.15 0.2 0.2 0.2 0.15 0.15 0.2 \
    --ipus-per-replica 16 \
    --replication-factor 1 \
    --epochs 20 \
    --gradient-accumulation 8192 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding False \
    --compile-only False