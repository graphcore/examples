#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
poprun \
    --num-instances=2 \
    --num-replicas=8 \
    --ipus-per-replica=2 \
python3 train.py \
    --config=efficientnet-b4-g16-gn-pod16 \
    --dataloader-rebatch-size 256 \
    --dataloader-worker=16 \
    $@
