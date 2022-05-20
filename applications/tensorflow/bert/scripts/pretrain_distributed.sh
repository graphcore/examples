#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.

# This script can be used to pretrain BERT Large on IPU-POD64
set -x

display_usage() {
	echo "Pretrain BERT.Large on IPU-POD32, IPU-POD64 or IPU-POD128"
	echo "Usage: $0 CONFIG VIPU_HOST HOST1 HOST2 ..."
	echo "CONFIG: \"POD32\", \"POD64\" or \"POD128\"."
	echo "VIPU_HOST: VIPU IP address."
	echo "HOSTn: a list of IP addresses on which the instances will be run."
}

if [[ $# -lt 3 ]]; then
   display_usage
   exit 1
fi

# First argument is VIPU server IPU address.
CONFIG="$1"
VIPU_SERVER_IP="$2"
# Parse arguments $3 to end
shift
shift
HOSTS="$@"
NUM_INSTANCES="$#"

HOSTS_LIST="${HOSTS[@]}"
echo ${HOSTS_LIST}
HOSTS_LIST="${HOSTS_LIST// /,}"

NETIF_NETMASK__IP_PART=(${HOSTS[0]//./ })
TCP_IF_NETMASK="${NETIF_NETMASK__IP_PART[0]}.${NETIF_NETMASK__IP_PART[1]}.0.0/16"
if [[ "${CONFIG}" == "POD32" ]]; then
   if [[ "${NUM_INSTANCES}" -ne 1 ]]; then
      echo "To run on POD32 use a single host."
      exit 1
   fi
   NUM_REPLICAS=8
   NUM_ILDS=1
   PHASE1_CONFIG_PATH='configs/pretrain_large_128_phase1_POD32.json'
   PHASE2_CONFIG_PATH='configs/pretrain_large_384_phase2_POD32.json'
   WANDB_NAME="POD32 Large"
   VIPU_PARTITION_NAME='pod32'
elif [[ "${CONFIG}" == "POD64" ]]; then
   if [[ "${NUM_INSTANCES}" -ne 1 ]]; then
      echo "To run on POD64 use a single host."
      exit 1
   fi
   NUM_REPLICAS=16
   NUM_ILDS=1
   PHASE1_CONFIG_PATH='configs/pretrain_large_128_phase1_POD64.json'
   PHASE2_CONFIG_PATH='configs/pretrain_large_384_phase2_POD64.json'
   WANDB_NAME="POD64 Large"
   VIPU_PARTITION_NAME='pod64'
elif [[ "${CONFIG}" == "POD128" ]]; then
   if [[ "${NUM_INSTANCES}" -ne 2 ]]; then
      echo "To run on POD128 use two hosts, one on each combined POD64."
      exit 1
   fi
   NUM_REPLICAS=32
   NUM_ILDS=2
   PHASE1_CONFIG_PATH='configs/pretrain_large_128_phase1_POD128.json'
   PHASE2_CONFIG_PATH='configs/pretrain_large_384_phase2_POD128.json'
   WANDB_NAME="POD128 Large"
   VIPU_PARTITION_NAME='gcl128'
else
   echo "Unknown configuration. Must be POD64 or POD128"
   exit 1
fi

# IPU-POD hosts
echo "Running ${NUM_INSTANCES} instances on hosts ${HOSTS[*]}"

LOCAL_HOME="/localdata/${USER}"
TFBERT_PATH="${LOCAL_HOME}/git/public_examples/applications/tensorflow/bert"

copy_ssh() {
   for host in ${HOSTS}; do
      ssh-copy-id "${host}"
   done
}

sync_venvs_and_code() {
   for host in ${HOSTS}; do
      echo "Syncing local code with ${host}"
      rsync --stats --exclude bert/checkpoint -av "${LOCAL_HOME}/git/public_examples" "${host}:${LOCAL_HOME}/git/"
      echo "Syncing local venvs with ${host}"
      rsync --stats -av "${LOCAL_HOME}/venvs" "${host}:${LOCAL_HOME}/"
      echo "Syncing local SDK with ${host}"
      rsync --stats -av "${LOCAL_HOME}/sdks" "${host}:${LOCAL_HOME}/"
   done
}

# Get the real directory path, avoiding symlinks, etc.
cd "$(realpath .)"

# Synchronise code, SDK, Python venv.
copy_ssh
sync_venvs_and_code

# Poplar options
# MPI options
MPI_GLOBAL_ARGS="--tag-output --mca btl_tcp_if_include ${TCP_IF_NETMASK} --mca oob_tcp_if_include ${TCP_IF_NETMASK}"
MPI_LOCAL_ARGS="-x TF_CPP_VMODULE='poplar_compiler=0, poplar_executor=0' -x HOROVOD_LOG_LEVEL=WARN -x IPUOF_LOG_LEVEL=WARN -x POPLAR_LOG_LEVEL=WARN -x CPATH -x GCL_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS=--executable_cache_path=${LOCAL_HOME}/exec_cache"

# Comma-separated hosts
# BERT training options
# Phase 1
PHASE1_TRAIN_FILE='/localdata/datasets/tf_wikipedia/tokenised_128_dup5_mask20/*.tfrecord'

POPRUN_OPTIONS="--host ${HOSTS_LIST} --num-ilds ${NUM_ILDS} --num-instances ${NUM_INSTANCES} --num-replicas ${NUM_REPLICAS} --ipus-per-replica 4 --vipu-server-host=${VIPU_SERVER_IP} --reset-partition=yes --update-partition=yes --remove-partition=no --vipu-partition=${VIPU_PARTITION_NAME} --print-topology=yes"
TIMESTAMP="$(date +%s)"

poprun ${POPRUN_OPTIONS} \
      --mpi-global-args="${MPI_GLOBAL_ARGS}" \
      --mpi-local-args="${MPI_LOCAL_ARGS}" \
      python "${TFBERT_PATH}/run_pretraining.py" --config "${PHASE1_CONFIG_PATH}" --train-file "${PHASE1_TRAIN_FILE}" --wandb --wandb-name "${WANDB_NAME} phase 1" 2>&1 | tee "pretrain_large_128_log_${TIMESTAMP}.txt"

phase1_ret_code="$?"
echo "Phase 1 exit code: ${phase1_ret_code}"
if [[ ${phase1_ret_code} != 0 ]]; then
   echo "Phase 1 run failed. Exiting."
   exit $?
fi

# Phase 2
PHASE2_TRAIN_FILE='/localdata/datasets/tf_wikipedia/tokenised_384_dup5_mask58/*.tfrecord'

PHASE1_CKPT_BASENAME='ckpt-7031'

LAST_CHANGED_CKPT_FILE=$(find "checkpoint/phase1/" -name "${PHASE1_CKPT_BASENAME}*" | tail -1)
PHASE1_CHECKPOINT_DIR=$(dirname "${LAST_CHANGED_CKPT_FILE}")
PHASE1_CHECKPOINT_DIR=$(realpath "${PHASE1_CHECKPOINT_DIR}")
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT_DIR}/${PHASE1_CKPT_BASENAME}"


# Synchronise phase 1 checkpoint with all hosts.
for host in ${HOSTS}; do
   echo "Syncing checkpoint ${PHASE1_CHECKPOINT} with ${host}:${PHASE1_CHECKPOINT_DIR}/"
   rsync -a --include "${PHASE1_CKPT_BASENAME}*" --exclude='*' "${PHASE1_CHECKPOINT_DIR}/" "${host}:${PHASE1_CHECKPOINT_DIR}/"
done

poprun ${POPRUN_OPTIONS} \
      --mpi-global-args="${MPI_GLOBAL_ARGS}" \
      --mpi-local-args="${MPI_LOCAL_ARGS}" \
      python "${TFBERT_PATH}/run_pretraining.py" --config "${PHASE2_CONFIG_PATH}" --train-file "${PHASE2_TRAIN_FILE}" --init-checkpoint "${PHASE1_CHECKPOINT}" --wandb --wandb-name "${WANDB_NAME} phase 2"  2>&1 | tee "pretrain_large_384_log_${TIMESTAMP}.txt"
