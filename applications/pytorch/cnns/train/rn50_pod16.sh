#!/bin/sh
poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=4 --numa-aware=yes --num-replicas=4 --ipus-per-replica=4 python3 train.py --config resnet50_mk2_pipelined $@
