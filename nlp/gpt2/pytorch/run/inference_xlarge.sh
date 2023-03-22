#!/bin/bash
python inference_gpt2.py \
    --model gpt2-xl \
    --max-len 1024 \
    --layers-per-ipu 1 6 6 7 7 7 7 7 \
    --matmul-proportion 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 \
    --ipus-per-replica 8 \
    --replication-factor 2 \
    --epochs 20 \
    --device-iterations 1 \
    --batch-size 1 \
    --gradient-accumulation 1 \
    --embedding-serialization-factor 4 \
    --enable-half-partials True \
    --dataset 'generated' \
    --replicated-tensor-sharding False
