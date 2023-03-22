#!/bin/bash
# run gpt2 testing
python train_gpt2.py \
    --model gpt2-test \
    --max-len 128 \
    --layers-per-ipu 0 4 \
    --matmul-proportion 0.20 0.20 \
    --ipus-per-replica 2 \
    --replication-factor 1 \
    --epochs 3 \
    --gradient-accumulation 32 \
    --device-iterations 1 \
    --batch-size 1 \
    --enable-sequence-serialized False \
    --embedding-serialization-factor 8 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --dataset 'generated'
