#!/bin/bash
# run gpt2-small on POD16
python train_gpt2.py \
    --model gpt2 \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 4 4 4 \
    --matmul-proportion 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --replication-factor 4 \
    --epochs 20 \
    --gradient-accumulation 2048 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False