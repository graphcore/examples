#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Half resolution training with poprun.
poprun \
    --num-instances=4 \
    --num-replicas=4 \
    --ipus-per-replica=4 \
python3 train.py \
    --config=efficientnet-b4-g16-gn-pod16 \
    --batch-size 15 \
    --gradient-accumulation 13 \
    --half-res-training \
    --fine-tune-epoch 2 \
    --fine-tune-lr 0.25 \
    --fine-tune-batch-size 6 \
    --fine-tune-gradient-accumulation 32 \
    --fine-tune-first-trainable-layer blocks/6/1 \
    --dataloader-worker=16 \
    $@
