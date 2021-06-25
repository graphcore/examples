#!/bin/sh
HOST_PREFIX=`ifconfig eno1 | grep inet | cut -d '.' -f 1,2,3 | rev | cut -d ' ' -f 1 | rev`
HOST1=$HOST_PREFIX.101
HOST2=$HOST_PREFIX.102
HOST3=$HOST_PREFIX.103
HOST4=$HOST_PREFIX.104
HOSTS=$HOST1,$HOST2,$HOST3,$HOST4
PARTITION=`vipu-admin list partitions --api-host $HOST1 | grep ACTIVE | cut -d '|' -f 2 | cut -d ' ' -f 2`

poprun -vv --num-instances=16 --num-replicas=16 \
       --ipus-per-replica=4 \
       --vipu-server-host=$HOST1 \
       --host=$HOSTS\
       --vipu-server-port 8090 \
       --num-ilds=1 \
       --vipu-partition=$PARTITION \
       --numa-aware=yes \
       --update-partition=yes \
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
                         -x IPUOF_VIPU_API_TIMEOUT=800" \
python3 train.py --config resnet50_mk2_pipelined_pod64 $@
