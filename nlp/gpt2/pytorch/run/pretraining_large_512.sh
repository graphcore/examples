#!/bin/bash
# run gpt2-large(seq_length=512) on POD16
python train_gpt2.py \
    --model gpt2-large \
    --max-len 512 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 1 5 5 5 5 5 5 5 \
    --matmul-proportion 0.15 0.12 0.15 0.15 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 8 \
    --replication-factor 2 \
    --epochs 20 \
    --gradient-accumulation 4096 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 8 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --dataset 'generated' \
    --replicated-tensor-sharding True
