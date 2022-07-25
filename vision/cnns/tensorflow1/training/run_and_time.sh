#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

# Host names, partition name, server ip and the net mask will be used in the command line.
# Host names are usually xxx-[m-n], or xxx,yyy,zzz,www
# A net mask can be xxx.xxx.xxx.0/16
# Example use case:
# ./run_and_time.sh 16 42 hosts partition server netmask ON
# Example use case with the option of uploading to wandb:
# ./run_and_time.sh 16 42 hosts partition server netmask ON --upload

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

# Create tmp directories on hosts:
TEMP_DIR="/localdata/$USER/tmp"
if [[ $HOSTS == *"["* ]]; then
    HOSTS_LIST=$(echo $HOSTS | sed -r 's:\[(.*)\]:{\1}:; s:\{([0-9]+)-([0-9]+)\}:\{\1..\2\}:g')
    for host in $(eval echo $HOSTS_LIST); do
        echo -e "\tCreating $TEMP_DIR on $host..."
        ssh $host "mkdir -p $TEMP_DIR"
    done
elif [[ $HOSTS == *","* ]]; then 
    HOSTS_LIST=$(echo $HOSTS | tr "," "\n")
    for host in $(eval echo $HOSTS_LIST); do
        echo -e "\tCreating $TEMP_DIR on $host..."
        ssh $host "mkdir -p $TEMP_DIR"
    done
else
    echo -e "\tCreating $TEMP_DIR on $HOSTS..."
    mkdir -p $TEMP_DIR
fi

export IPUOF_LOG_LEVEL="WARN"
export POPLAR_LOG_LEVEL="WARN"
export IPUOF_VIPU_API_TIMEOUT=300
export TEMPDIR=$TEMP_DIR
export TMPDIR=$TEMP_DIR
export TEMP=$TEMP_DIR
export TMP=$TEMP_DIR
export DATA_DIR=/localdata/datasets/imagenet-data
export EXECUTABLE_CACHE=/localdata/$USER/executable_cache
export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout":"900"}'
export POPLAR_RUNTIME_OPTIONS='{"streamCallbacks.maxLookahead":"unlimited"}'

MPI_LOCAL_ARGS="' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH \
  -x IPUOF_VIPU_API_TIMEOUT -x POPLAR_LOG_LEVEL -x IPUOF_LOG_LEVEL -x TEMPDIR \
  -x TMPDIR -x TEMP -x TMP -x POPLAR_ENGINE_OPTIONS -x POPLAR_RUNTIME_OPTIONS \
  -x TF_POPLAR_FLAGS'"
MPI_GLOBAL_ARGS="'--mca oob_tcp_if_include \
  $NETMASK --mca btl_tcp_if_include $NETMASK '"
MPI_SETTINGS="--update-partition=yes --reset-partition=no --vipu-server-timeout 600 \
  --vipu-server-host $VIPU_SERVER_HOST \
  --vipu-partition $PARTITION --executable-cache-path $EXECUTABLE_CACHE"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# The default path of logs
LOGS_PATH="/localdata/"$USER"/POD"$REPLICAS"/s"$SEED"_$TIMESTAMP"
if [[ $REPLICAS -eq "16" ]]
then
  TRAIN=" poprun \
    -vv --host $HOSTS --mpi-global-args $MPI_GLOBAL_ARGS --mpi-local-args $MPI_LOCAL_ARGS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config resnet50_mlperf_pod16_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR" --stochastic-rounding "$STOCHASTIC_ROUNDING" "
elif [[ $REPLICAS -eq "32" ]]
then
  TRAIN=" poprun \
    -vv --host $HOSTS --mpi-global-args $MPI_GLOBAL_ARGS --mpi-local-args $MPI_LOCAL_ARGS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config resnet50_mlperf_pod32_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR" --stochastic-rounding "$STOCHASTIC_ROUNDING" "
elif [[ $REPLICAS -eq "64" ]]
then
  # POD64 through poprun
  TRAIN=" poprun \
    -vv --host $HOSTS --mpi-global-args $MPI_GLOBAL_ARGS --mpi-local-args $MPI_LOCAL_ARGS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config resnet50_mlperf_pod64_lars --logs-path "$LOGS_PATH" \
    --seed "$SEED" --data-dir "$DATA_DIR" --stochastic-rounding "$STOCHASTIC_ROUNDING" "
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
