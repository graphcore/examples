#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Half resolution training with poprun.
poprun \
    --mpi-global-args="--allow-run-as-root --tag-output" \
    --numa-aware=yes \
    --num-instances=4 \
    --num-replicas=8 \
    --ipus-per-replica=2 \
python3 train.py \
    --config=efficientnet-b0-g16-gn-pod16 \
    --batch-size 40 \
    --gradient-accumulation 3 \
    --half-res-training \
    --fine-tune-epoch 2 \
    --fine-tune-lr 0.25 \
    --fine-tune-batch-size 20 \
    --fine-tune-gradient-accumulation 5 \
    --fine-tune-first-trainable-layer blocks/5/3 \
    --dataloader-worker=16 \
    $@
