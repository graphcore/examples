#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
HOST1=`ifconfig eno1 | grep "inet " | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | head -1`
OCT1=`echo "$HOST1" | cut -d "." -f 1`
OCT2=`echo "$HOST1" | cut -d "." -f 2`
OCT3=`echo "$HOST1" | cut -d "." -f 3`
OCT4=`echo "$HOST1" | cut -d "." -f 4`
RNIC1=$OCT1.`expr $OCT2 + 4`.`expr $OCT3`.`expr $OCT4`
RNIC2=$OCT1.`expr $OCT2 + 4`.`expr $OCT3 + 1`.`expr $OCT4`
HOSTS=$RNIC1,$RNIC2
VIPU_SERVER=${VIPU_SERVER:=$RNIC1}
FIRST_PARTITION=`vipu-admin list partitions --api-host $HOST1| grep ACTIVE | cut -d '|' -f 2 | cut -d ' ' -f 2 | head -1`
PARTITON=${PARTITION:=$FIRST_PARTITION}
poprun -vv --num-instances=2 --num-replicas=32 \
        --num-ilds=2 \
        --ipus-per-replica=4 \
        --vipu-server-host=$VIPU_SERVER\
        --host=$HOSTS\
        --vipu-partition=$PARTITION \
        --update-partition=yes \
        --remove-partition=no \
        --reset-partition=no \
        --print-topology=yes \
        --mpi-global-args=" --tag-output \
                            --mca btl_tcp_if_include 10.5.0.0/16 \
                            --mca oob_tcp_if_include 10.5.0.0/16" \
python3 run_pretraining.py --config pretrain_large_128_POD128 \
                           --checkpoint-output-dir checkpoints/pretrain_large_128