#!/bin/sh
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
HOST1=`ifconfig eno1 | grep "inet " | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | head -1`
OCT123=`echo "$HOST1" | cut -d "." -f 1,2,3`
OCT4=`echo "$HOST1" | cut -d "." -f 4`
HOST2=$OCT123.`expr $OCT4 + 1`
HOST3=$OCT123.`expr $OCT4 + 2`
HOST4=$OCT123.`expr $OCT4 + 3`
HOSTS=$HOST1,$HOST2,$HOST3,$HOST4
VIPU_SERVER=${VIPU_SERVER:=$HOST1}
FIRST_PARTITION=`vipu-admin list partitions --api-host $VIPU_SERVER| grep ACTIVE | cut -d '|' -f 3 | cut -d ' ' -f 2 | head -1`
PARTITON=${PARTITION:=$FIRST_PARTITION}
# POPLAR options saves a bit of memory.
POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout": 900}' \
poprun --vv \
       --num-instances=16 \
       --num-replicas=64 \
       --vipu-server-host=$VIPU_SERVER\
       --host=$HOSTS\
       --vipu-partition=$PARTITION \
       --reset-partition=no \
       --executable-cache-path cache/rn50-pod64 \
       --mpi-global-args="--mca btl_tcp_if_include eno1" \
       --mpi-local-args="-x LD_LIBRARY_PATH \
                         -x OPAL_PREFIX \
                         -x PATH \
                         -x CPATH \
                         -x PYTHONPATH \
                         -x IPUOF_VIPU_API_TIMEOUT=800 \
                         -x POPLAR_ENGINE_OPTIONS" \
python3 train.py --config resnet50-pod64 --dataloader-worker 14 --dataloader-rebatch-size 256 $@