#! /bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# exit when any command fails
set -e

# Example use case: ./run_and_time.sh 16 42
# Example use case: ./run_and_time.sh 16 42 --upload to send to wandb
if [[ "$#" > 3 ||  "$#" == 0 ]]
then
    echo "Usage: $0 NUM-REPLICAS SEED [--upload]"
    exit 1
fi

REPLICAS=$1
INSTANCES=$(echo $REPLICAS / 2 | bc)
SEED=$2

# machine identifiers
LR=`hostname | cut -c 3-4`
HOST0=10.10.$LR.150
HOST1=10.10.$LR.151
HOST2=10.10.$LR.152
HOST3=10.10.$LR.153
HOSTS=$HOST0,$HOST1,$HOST2,$HOST3
MAINHOST=$HOST0
PARTITION=`vipu-admin --api-host localhost list partitions | grep ACTIVE | cut -d '|' -f 2 | cut -d ' ' -f 2`
VIPU_SERVER_HOST=10.3.$LR.150

echo "CLEARING THE CACHE FOR POD LR$LR..."
mpirun --tag-output --prefix $OPAL_PREFIX --allow-run-as-root --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16 --host $HOSTS sshpass -f pass.file ssh -o userknownhostsfile=/dev/null -o stricthostkeychecking=no ipuuser@127.0.0.1 "sudo sh -c \"sync; echo 3 > /proc/sys/vm/drop_caches\""

export IPUOF_LOG_LEVEL=WARN
export IPUOF_VIPU_API_TIMEOUT=300
export TF_POPLAR_FLAGS=--executable_cache_path=/localdata/$USER/exec_cache
export TEMP=/localdata/$USER/tmp
export DATA_DIR=/localdata/datasets/imagenet-data
export POPLAR_ENGINE_OPTIONS='{"opt.useAutoloader":"true","target.syncReplicasIndependently":"true", "streamCallbacks.numWorkerThreads" : "0", "streamCallbacks.multiThreadMode": "dedicated", "opt.enableMultiAccessCopies":"false", "target.deterministicWorkers":"true"}'
export POPLAR_TARGET_OPTIONS='{"gatewayMode":"false"}'
MPI_SETTINGS="--mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include 10.10.0.0/16 --mca btl_tcp_if_include 10.10.0.0/16' \
    --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_ENGINE_OPTIONS -x TF_POPLAR_FLAGS -x POPLAR_TARGET_OPTIONS' \
    --update-partition=no --reset-partition=no --vipu-server-timeout 300 \
    --ipus-per-replica 1 --numa-aware 1 --only-output-from-instance 0 \
    --vipu-server-host "$VIPU_SERVER_HOST" --vipu-partition=$PARTITION "

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p /localdata/$USER/
mkdir -p /localdata/$USER/tmp

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

if [[ $REPLICAS -eq "64" ]]
then
  # POD64 through poprun
  LOGS_PATH="./logs/bs16_pod64_$TIMESTAMP"
  INSTANCES=16
  TRAIN="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $HOSTS $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod64_bs20 --logs-path "$LOGS_PATH" \
    --epochs-per-sync 20 --seed "$SEED" --data-dir "$DATA_DIR" --no-validation"
  # Use less replicas (16) to exactly fit 50k validation samples (16*25*125)
  VALIDATE="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances 8 --num-replicas 16 \
    python validation.py --restore-path "$LOGS_PATH" --logs-path "$LOGS_PATH" \
    --config mk2_resnet50_mlperf_pod16 --data-dir "$DATA_DIR" --no-stochastic-rounding   \
    --batch-size 25 --seed "$SEED" --available-memory-proportion 0.6 --epochs 45"
elif [[ $REPLICAS -eq "16" ]]
then
  # POD16 through poprun
  LOGS_PATH="./logs/bs20_pod16_$TIMESTAMP"
  TRAIN="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python train.py --config mk2_resnet50_mlperf_pod16_bs20 --logs-path "$LOGS_PATH" \
    --epochs-per-sync 20 --seed "$SEED" --data-dir "$DATA_DIR" --no-validation  "
  VALIDATE="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python validation.py --restore-path "$LOGS_PATH" --logs-path "$LOGS_PATH" \
    --config mk2_resnet50_mlperf_pod16 --data-dir "$DATA_DIR" --no-stochastic-rounding  \
    --batch-size 25 --seed "$SEED" --available-memory-proportion 0.6"
elif [[ $REPLICAS -eq "4" ]]
then
  # POD4 through poprun
  LOGS_PATH="./logs/bs16_pod4_$TIMESTAMP"
  TRAIN="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
  python train.py --config mk2_resnet50_mlperf_pod4_bs20 --logs-path "$LOGS_PATH" \
    --epochs-per-sync 20 --seed "$SEED" --data-dir "$DATA_DIR" --no-validation"
  VALIDATE="POPLAR_ENGINE_OPTIONS='"$POPLAR_ENGINE_OPTIONS"' poprun \
    -vv --host $MAINHOST $MPI_SETTINGS \
    --num-instances "$INSTANCES" --num-replicas "$REPLICAS" \
    python validation.py --restore-path "$LOGS_PATH" --logs-path "$LOGS_PATH" \
    --config mk2_resnet50_mlperf_pod4_bs20 --data-dir "$DATA_DIR" --no-stochastic-rounding \
    --batch-size 50 --seed "$SEED" --available-memory-proportion 0.6"
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

if [[ $3 == "--upload" ]]
then
  echo "Running wandb upload:"
  WANDB="python upload_run.py --base-folder "$LOGS_PATH" \
    --name dbn2_bs20_"$REPLICAS"r_44e_3840tbs_wd25_8io_aelr_poprun"$INSTANCES"_LR"$LR"_s"$SEED" \
    --project mlperf-rn50"
  echo $WANDB
  eval $WANDB
fi
