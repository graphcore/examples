#! /bin/bash
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# exit when any command fails
set -e

POSITIONAL_ARGS=()

# default values
SEED=$((1 + $RANDOM % 1000000))
EXPERIMENT_NAME=""
CONFIG="resnet50_16ipus_8k_bn_pipeline"
INSTANCES=""
CACHE=false
WANDB=true
POPRUN=true
TOTAL_IPUS=16
VALIDATION=true
CKPT_ALL_INSTANCES=false
IPUS_PER_REPLICA=4

# multi host run
HOSTS=""
PARTITION=""
VIPU_SERVER_HOST=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift; shift
      ;;
    --seed)
      SEED="$2"
      shift; shift
      ;;
    --total-ipus)
      TOTAL_IPUS="$2"
      shift; shift
      ;;
    --ipus-per-replica)
      IPUS_PER_REPLICA="$2"
      shift; shift
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift; shift
      ;;
    --num-instances)
      INSTANCES="$2"
      shift; shift
      ;;
    --partition)
      PARTITION="$2"
      shift; shift
      ;;
    --vipu-host)
      VIPU_SERVER_HOST="$2"
      shift; shift
      ;;
    --hosts)
      HOSTS="$2"
      shift; shift
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift; shift
      ;;
    ---wandb)
      WANDB="$2"
      shift; shift
      ;;
    --poprun-off)
      POPRUN=false
      shift
      ;;
    --cache)
      CACHE=true
      shift
      ;;
    -nv|--no-validation)
      VALIDATION=false
      shift
      ;;
    --ckpt-all-instances)
      CKPT_ALL_INSTANCES=true
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ${#POSITIONAL_ARGS} -gt 0 ]; then echo "Error. Unknown arguments: $POSITIONAL_ARGS[@]"; exit 1; fi

# generate configs
EXECUTABLE_CACHE="/localdata/$USER/exec_cache"
BASE_DIR="/localdata/"$USER"/POD"$TOTAL_IPUS""
[[ $CONFIG == *"pipeline"* ]] && IPUS_PER_REPLICA=4 || IPUS_PER_REPLICA=1
NUM_REPLICAS=$(echo $TOTAL_IPUS / $IPUS_PER_REPLICA | bc)
[ -z "$INSTANCES" ] && INSTANCES=$(echo $NUM_REPLICAS / 2 | bc)
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
[ -z "$EXPERIMENT_NAME" ] && CKPT_DIR="$BASE_DIR/s${SEED}_${TIMESTAMP}" || CKPT_DIR="$BASE_DIR/$EXPERIMENT_NAME_$TIMESTAMP"
[ $CACHE = true ] && TF_POPLAR_FLAGS="--executable-cache-path $EXECUTABLE_CACHE" || TF_POPLAR_FLAGS=""
[ -z "$NUM_EPOCHS" ] && NUM_EPOCHS="" || NUM_EPOCHS="--num-epochs $NUM_EPOCHS"


if [[ $PARTITION == "" ]] || [[ $VIPU_SERVER_HOST == "" ]] || [[ $HOSTS == "" ]]; then
  MULTIHOST=false
  echo "Mutli-host run disabled because --partition, --vipu-host or --hosts wasnt specified."
else
  MULTIHOST=true
fi

echo "Arguments:"
echo "seed = $SEED"
echo "wandb = $WANDB"
echo "cache = $CACHE"
echo "poprun = $POPRUN"
echo "config = $CONFIG"
echo "ckpt_dir = $CKPT_DIR"
echo "validation = $VALIDATION"
echo "num_epochs = $NUM_EPOCHS"
echo "total_ipus = $TOTAL_IPUS"
echo "num_instances = $INSTANCES"
echo "num_replicas = $NUM_REPLICAS"
echo "experiment_name = $EXPERIMENT_NAME"
echo "ipus_per_replica = $IPUS_PER_REPLICA"
echo "ckpt_all_instances = $CKPT_ALL_INSTANCES"
if [ $MULTIHOST = true ]; then
  echo "hosts = $HOSTS"
  echo "partition = $PARTITION"
  echo "vipu_server_host = $VIPU_HOST"
fi
echo ""

PROJECT_ROOT_PATH=$( cd -- "$( dirname "$( dirname -- "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )

if [ $MULTIHOST = true ]; then
  # split string by comma
  delimiter=","  
  s=$HOSTS$delimiter  
  host_array=()
  while [[ $s ]]
    do  
    host_array+=( "${s%%"$delimiter"*}" )
    s=${s#*"$delimiter"} 
  done

  declare -p host_array
  echo "${host_array[@]}"

  echo "bash ../../../utils/distributed_training/copy_ssh.sh ${host_array[@]}"
  eval "bash ../../../utils/distributed_training/copy_ssh.sh ${host_array[@]}"

  echo "bash ../../../utils/distributed_training/config_pod.sh $PARTITION ${host_array[@]}"
  eval "bash ../../../utils/distributed_training/config_pod.sh $PARTITION ${host_array[@]}"

  MULTIHOST_SETTINGS="-vv --host $HOSTS --numa-aware 1 \
      --mpi-global-args='--tag-output --allow-run-as-root --mca oob_tcp_if_include eno1 --mca btl_tcp_if_include eno1' \
      --vipu-partition="$PARTITION" --update-partition=yes --reset-partition=no --vipu-server-host "$VIPU_SERVER_HOST" "
fi

PYTHON_BASE="python3 ${PROJECT_ROOT_PATH}/train.py --config $CONFIG --seed $SEED --on-demand False --wandb $WANDB"
POPRUN_BASE="poprun $MULTIHOST_SETTINGS --num-instances $INSTANCES $TF_POPLAR_FLAGS --only-output-from-instance 0" 

# training
if [ $POPRUN = true ]
then export poprun_prefix="$POPRUN_BASE --num-replicas $NUM_REPLICAS --ipus-per-replica $IPUS_PER_REPLICA"
else export poprun_prefix=""; fi
export python_cmd="$PYTHON_BASE $NUM_EPOCHS --ckpt-all-instances $CKPT_ALL_INSTANCES --validation $VALIDATION --checkpoint-dir $CKPT_DIR --pipeline-validation-model"
echo "Running training:"
echo $poprun_prefix $python_cmd
eval $poprun_prefix $python_cmd
echo ""
