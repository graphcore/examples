#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false"}' \
poprun --num-instances=8 --num-replicas=16 python3 train.py --config resnet50 --dataloader-worker 14 --dataloader-rebatch-size 256 $@
