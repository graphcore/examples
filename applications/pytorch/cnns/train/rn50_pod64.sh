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
FIRST_PARTITION=`vipu-admin list partitions --api-host $VIPU_SERVER| grep ACTIVE | cut -d '|' -f 2 | cut -d ' ' -f 2 | head -1`
PARTITON=${PARTITION:=$FIRST_PARTITION}
# POPLAR options saves a bit of memory.
POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout": 900}' \
poprun -vv --num-instances=16 --num-replicas=64 \
       --ipus-per-replica=1 \
       --vipu-server-host=$VIPU_SERVER\
       --host=$HOSTS\
       --vipu-server-port 8090 \
       --num-ilds=1 \
       --vipu-partition=$PARTITION \
       --numa-aware=yes \
       --update-partition=no \
       --remove-partition=no \
       --reset-partition=no \
       --print-topology=yes \
       --executable-cache-path cache/rn50_pod64 \
       --mpi-global-args="--tag-output \
                          --allow-run-as-root \
                          --mca btl_tcp_if_include eno1" \
       --mpi-local-args="-x LD_LIBRARY_PATH \
                         -x OPAL_PREFIX \
                         -x PATH \
                         -x CPATH \
                         -x PYTHONPATH \
                         -x IPUOF_VIPU_API_TIMEOUT=800 \
                         -x POPLAR_ENGINE_OPTIONS" \
python3 train.py --config resnet50_mk2_pod64 --dataloader-worker 24 --dataloader-rebatch-size 256 --webdataset-memory-cache-ratio 0.95 $@