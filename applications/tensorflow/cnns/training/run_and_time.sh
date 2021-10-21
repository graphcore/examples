#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

# Host ips, partition name, server ip and the net mask will be used in the command line.
# Host ips are usually xxx.xxx.xxx.0, xxx.xxx.xxx.1, xxx.xxx.xxx.2, and xxx.xxx.xxx.3
# A net mask can be xxx.xxx.xxx.0/24
# Example use case:
# ./run_and_time.sh 16 42 host0-ip host1-ip host2-ip host3-ip partition server-ip netmask
# Example use case with the option of uploading to wandb:
# ./run_and_time.sh 16 42 host0-ip host1-ip host2-ip host3-ip partition server-ip netmask --upload

if [[ "$#" -gt 7 ||  "$#" == 0 ]]
then
    echo "Usage: $0 NUM-REPLICAS SEED HOST0 PARTITION SERVER NETMASK [--upload]"
    exit 1
fi

REPLICAS=$1
INSTANCES=$(echo $REPLICAS / 2 | bc)
SEED=$2

# machine identifiers
HOST0=$3
IP1=`echo $HOST0 | cut -d "." -f 1`
IP2=`echo $HOST0 | cut -d "." -f 2`
IP3=`echo $HOST0 | cut -d "." -f 3`
IP4=`echo $HOST0 | cut -d "." -f 4`
HOST1="$IP1.$IP2.$IP3.$((IP4+1))"
HOST2="$IP1.$IP2.$IP3.$((IP4+2))"
HOST3="$IP1.$IP2.$IP3.$((IP4+3))"
HOST4="$IP1.$IP2.$((IP3+1)).$IP4"
HOST5="$IP1.$IP2.$((IP3+1)).$((IP4+1))"
HOST6="$IP1.$IP2.$((IP3+1)).$((IP4+2))"
HOST7="$IP1.$IP2.$((IP3+1)).$((IP4+3))"
HOSTS_4=$HOST0,$HOST1,$HOST2,$HOST3
HOSTS_8=$HOST0,$HOST1,$HOST2,$HOST3,$HOST4,$HOST5,$HOST6,$HOST7
MAINHOST=$HOST0
PARTITION=$4
VIPU_SERVER_HOST=$5
NETMASK=$6

echo "CLEARING THE CACHE FOR POD ..."

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TF_POPLAR_FLAGS=--executable_cache_path=/localdata/$USER/exec_cache
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/imagenet-data
export POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.deterministicWorkers":"portable", "target.hostSyncTimeout":"900"}'
export POPLAR_TARGET_OPTIONS='{"gatewayMode":"false"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include "$NETMASK" --mca btl_tcp_if_include "$NETMASK"' \
    --mpi-local-args=' -x OPAL_PREFIX -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_ENGINE_OPTIONS -x TF_POPLAR_FLAGS -x POPLAR_TARGET_OPTIONS' \
    --update-partition=no --reset-partition=no \
    --ipus-per-replica 1 --only-output-from-instance 0 \
    --vipu-server-host "$VIPU_SERVER_HOST" --vipu-partition=$PARTITION "

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p /localdata/$USER/
mkdir -p /localdata/$USER/tmp

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

LOGS_PATH="./logs/bs20_ipu"$REPLICAS"_$TIMESTAMP"
if [[ $REPLICAS -eq "64" ]]
then
  # POD64 through poprun
  TRAIN="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $HOSTS_4 $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod64_bs20 --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" --no-validation"
  # Use less replicas (16) to exactly fit 50k validation samples (16*25*125)
  VALIDATE="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances 8 --num-replicas 16 \
    python validation.py --restore-path "$LOGS_PATH" --logs-path "$LOGS_PATH" \
    --config mk2_resnet50_mlperf_pod16 --data-dir "$DATA_DIR" --no-stochastic-rounding   \
    --batch-size 25 --seed "$SEED" --available-memory-proportion 0.6 --epochs 41"
elif [[ $REPLICAS -eq "128" ]]
then
  # POD128 through poprun
  export POPLAR_TARGET_OPTIONS='{"gatewayMode":"true"}'
  TRAIN=poprun \
    -vv --host $HOSTS_8 $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod128_lars --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" --no-validation"
  VALIDATE=poprun \
    -vv --host $HOSTS_8 $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python validation.py --restore-path "$LOGS_PATH" --logs-path "$LOGS_PATH" \
    --config mk2_resnet50_mlperf_pod128 --data-dir "$DATA_DIR" --no-stochastic-rounding  \
    --batch-size 50 --seed "$SEED" --available-memory-proportion 0.6"
elif [[ $REPLICAS -eq "16" ]]
then
  TRAIN="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod16_lars --logs-path "$LOGS_PATH" \
    --identical-replica-seeding --seed "$SEED" --data-dir "$DATA_DIR" --no-validation "
  VALIDATE="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python validation.py --restore-path "$LOGS_PATH" --logs-path "$LOGS_PATH" \
    --config mk2_resnet50_mlperf_pod16_lars --data-dir "$DATA_DIR" --no-stochastic-rounding  \
    --batch-size 25 --seed "$SEED" --available-memory-proportion 0.6"
else
  echo "Not implemented for "$REPLICAS" replicas"
  exit
fi

echo "Running training:"
echo $TRAIN
eval $TRAIN
echo "Running validation:"
echo $VALIDATE
eval $VALIDATE

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"

if [[ $7 == "--upload" ]]
then
  echo "Running wandb upload:"
  WANDB="python upload_run.py --base-folder "$LOGS_PATH" \
    --name dbn2_bs20_"$REPLICAS"r_44e_3840tbs_wd25_8io_aelr_poprun"$INSTANCES"_s"$SEED" \
    --project mlperf-rn50"
  echo $WANDB
  eval $WANDB
fi
