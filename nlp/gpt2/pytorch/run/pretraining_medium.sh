#!/bin/bash
# run gpt2-medium on POD16
python train_gpt2.py \
    --model gpt2-medium \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 3 3 3 3 4 4 4 \
    --matmul-proportion 0.30 0.15 0.15 0.15 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 8 \
    --replication-factor 2 \
    --epochs 20 \
    --gradient-accumulation 4096 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --dataset 'generated' \
    --replicated-tensor-sharding True