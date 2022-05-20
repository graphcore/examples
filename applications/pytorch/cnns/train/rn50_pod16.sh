#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false"}' \
poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=8 --numa-aware=yes --num-replicas=16 --ipus-per-replica=1 python3 train.py --config resnet50 --dataloader-worker 14 --dataloader-rebatch-size 256 $@
