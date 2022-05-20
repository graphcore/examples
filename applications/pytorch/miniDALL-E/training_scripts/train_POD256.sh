#!/bin/bash

TCP_IF_INCLUDE="vlan2800"
HOSTS="lr17-1-poplar-1,lr17-1-poplar-2,lr17-1-poplar-3,lr17-1-poplar-4,lr17-1-poplar-5,lr17-1-poplar-6,lr17-1-poplar-7,lr17-1-poplar-8,lr17-1-poplar-9,lr17-1-poplar-10,lr17-1-poplar-11,lr17-1-poplar-12,lr17-1-poplar-13,lr17-1-poplar-14,lr17-1-poplar-15,lr17-1-poplar-16"
PARTITION="pod256-dalle"

poprun \
    -vv \
    --host $HOSTS \
    --vipu-partition=$PARTITION \
    --update-partition=yes \
    --print-topology=yes \
    --mpi-global-args="--tag-output --allow-run-as-root --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
    --num-replicas=64 \
    --numa-aware=yes \
    --num-ilds=4 \
    --num-instances=16 \
    --ipus-per-replica=4 \
    --executable-cache-path=$POPTORCH_CACHE_DIR \
    python train.py \
        --config L16_POD256 \
        --generated-data \
        --checkpoint-output-dir "" \
        --epochs 2 \
        --byteio True
