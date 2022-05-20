#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

# Host names, partition name, server ip and the net mask will be used in the command line.
# Host names are usually xxx-[m-n], or xxx,yyy,zzz,www
# A net mask can be xxx.xxx.xxx.0/16
# Example use case:
# ./run_and_time.sh 16 42 hosts partition server netmask
# Example use case with the option of uploading to wandb:
# ./run_and_time.sh 16 42 hosts partition server netmask --upload

if [[ "$#" -gt 8 ||  "$#" == 0 ]]
then
    echo "Usage: $0 NUM-REPLICAS SEED HOST0 PARTITION SERVER NETMASK SR [--upload]"
    exit 1
fi

REPLICAS=$1
INSTANCES=$(echo $REPLICAS / 2 | bc)
SEED=$2

HOSTS=$3
PARTITION=$4
VIPU_SERVER_HOST=$5
NETMASK=$6

STOCHASTIC_ROUNDING=$7

echo "CLEARING THE CACHE FOR POD ..."

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/imagenet-data
export EXECUTABLE_CACHE=/localdata/$USER/executable_cache
export POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false", "target.hostSyncTimeout":"900"}'
export POPLAR_RUNTIME_OPTIONS='{"streamCallbacks.maxLookahead":"unlimited"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include "$NETMASK" --mca btl_tcp_if_include "$NETMASK"' \
    --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=INFO -x POPLAR_ENGINE_OPTIONS -x POPLAR_RUNTIME_OPTIONS ' \
    --update-partition=yes --reset-partition=no --vipu-server-timeout 600 \
    --ipus-per-replica 1 --numa-aware 1 --vipu-server-host "$VIPU_SERVER_HOST" \
    --vipu-partition="$PARTITION" \
    --executable-cache-path "$EXECUTABLE_CACHE" "

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p /localdata/$USER/tmp

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# The default path of logs
LOGS_PATH="/localdata/"$USER"/POD"$REPLICAS"/s"$SEED"_$TIMESTAMP"
if [[ $REPLICAS -eq "16" ]]
then
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod16_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR" --stochastic-rounding "$STOCHASTIC_ROUNDING" "
elif [[ $REPLICAS -eq "64" ]]
then
  # POD64 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod64_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR" --stochastic-rounding "$STOCHASTIC_ROUNDING" "
elif [[ $REPLICAS -eq "128" ]]
then
  # POD128 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod128_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR"  --stochastic-rounding "$STOCHASTIC_ROUNDING" "
elif [[ $REPLICAS -eq "256" ]]
then
  # POD256 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod256_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR"  --stochastic-rounding "$STOCHASTIC_ROUNDING" "
else
  echo "Not implemented for "$REPLICAS" replicas"
  exit
fi

# if prng seed management is disabled, add identical replica seeding arg
if [[ $STOCHASTIC_ROUNDING == 'OFF' || $STOCHASTIC_ROUNDING == 'ON' ]]; then
  TRAIN="$TRAIN --identical-replica-seeding "
fi

echo "Running training and validation:"
echo $TRAIN
eval $TRAIN

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"

if [[ $8 == "--upload" ]]
then
  echo "Running wandb upload:"
  WANDB="python upload_run.py --base-folder "$LOGS_PATH" \
    --name dbn2_bs20_"$REPLICAS"r_44e_3840tbs_wd25_8io_aelr_poprun"$INSTANCES"_s"$SEED" \
    --project mlperf-rn50"
  echo $WANDB
  eval $WANDB
fi
