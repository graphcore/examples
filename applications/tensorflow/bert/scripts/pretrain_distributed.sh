#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.

# This script can be used to pretrain BERT Large on IPU-POD64
set +eux
display_usage() {
	echo "Pretrain BERT.Large on IPU-POD64"
	echo "Usage: $0 VIPU_HOST"
	echo "VIPU_HOST: VIPU server host IP address."

}

if [[ $# -lt 1 ]]; then
   display_usage
   exit 1
fi

# First argument is VIPU server IPU address.
VIPU_SERVER_IP="$1"
HOSTS=()

IP_PART_A="${VIPU_SERVER_IP%*.*}"
IP_PART_B="${VIPU_SERVER_IP#*.*.*.*}"

for k in $(seq 0 3); do
   HOSTS+=("${IP_PART_A}.$((IP_PART_B+k))")
done

# IPU-POD hosts
echo "Poprun will use hosts: ${HOSTS[*]}"

LOCAL_HOME="/localdata/${USER}"
TFBERT_PATH="${LOCAL_HOME}/public_examples/applications/tensorflow/bert"

copy_ssh() {
   for host in "${HOSTS[@]}"; do
      ssh-copy-id "${host}"
   done
}

sync_venvs_and_code() {
   for host in "${HOSTS[@]}"; do
      echo "Syncing local code with ${host}"
      rsync --stats --exclude bert/checkpoint -av "${LOCAL_HOME}/public_examples" "${host}:${LOCAL_HOME}/"
      echo "Syncing local venvs with ${host}"
      rsync --stats -av "${LOCAL_HOME}/ENVS" "${host}:${LOCAL_HOME}/"
      echo "Syncing local SDK with ${host}"
      rsync --stats -av "${LOCAL_HOME}/SDKS" "${host}:${LOCAL_HOME}/"
   done
}

# Get the real directory path, avoiding symlinks, etc.
cd "$(realpath .)"

# Synchronise code, SDK, Python venv.
copy_ssh
sync_venvs_and_code

# Poplar options
# MPI options
MPI_GLOBAL_ARGS="--tag-output --mca btl_tcp_if_include ${VIPU_SERVER_IP}/24"
MPI_LOCAL_ARGS="-x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x OPAL_PREFIX -x TF_CPP_VMODULE='poplar_compiler=1, poplar_executor=1' -x HOROVOD_LOG_LEVEL=WARN -x IPUOF_LOG_LEVEL=WARN -x  IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x CPATH -x GCL_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS=--executable_cache_path=${LOCAL_HOME}/exec_cache"

# Comma-separated hosts
HOSTS_LIST="${HOSTS[@]}"
HOSTS_LIST="${HOSTS_LIST// /,}"

# BERT training options
# Phase 1
PHASE1_CONFIG_PATH='configs/pretrain_large_128_phase1_POD64.json'
PHASE1_TRAIN_FILE='/localdata/datasets/tf_wikipedia/tokenised_128_dup5_mask20/*.tfrecord'
VIPU_PARTITION_NAME='pod64'

POPRUN_OPTIONS="--host ${HOSTS_LIST} --num-ilds 1 --num-instances 4 --num-replicas 16 --ipus-per-replica 4 --numa-aware=yes --vipu-server-host=${VIPU_SERVER_IP} --reset-partition=no --update-partition=yes --vipu-partition=${VIPU_PARTITION_NAME} --vipu-server-timeout=600"

poprun ${POPRUN_OPTIONS} --mpi-global-args="${MPI_GLOBAL_ARGS}" --mpi-local-args="${MPI_LOCAL_ARGS}" python "${TFBERT_PATH}/run_pretraining.py" --config "${PHASE1_CONFIG_PATH}" --train-file "${PHASE1_TRAIN_FILE}" --wandb --wandb-name "POD64 Large"  2>&1 | tee pretrain_large_128_log.txt

# Phase 2
PHASE2_CONFIG_PATH='configs/pretrain_large_384_phase2_POD64.json'
PHASE2_TRAIN_FILE='/localdata/datasets/tf_wikipedia/tokenised_384_dup5_mask56/*.tfrecord'

PHASE1_CKPT_BASENAME='ckpt-7031'

LAST_CHANGED_CKPT_FILE=$(find "checkpoint/phase1/" -name "${PHASE1_CKPT_BASENAME}*" | tail -1)
PHASE1_CHECKPOINT_DIR=$(dirname "${LAST_CHANGED_CKPT_FILE}")
PHASE1_CHECKPOINT_DIR=$(realpath "${PHASE1_CHECKPOINT_DIR}")
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT_DIR}/${PHASE1_CKPT_BASENAME}"

# Synchronise phase 1 checkpoint with all hosts.
for host in "${HOSTS[@]}"; do
   echo "Syncing checkpoint ${PHASE1_CHECKPOINT} with ${host}:${PHASE1_CHECKPOINT_DIR}/"
   rsync -a --include "${PHASE1_CKPT_BASENAME}*" --exclude='*' "${PHASE1_CHECKPOINT_DIR}/" "${host}:${PHASE1_CHECKPOINT_DIR}/"
done

poprun ${POPRUN_OPTIONS} --mpi-global-args="${MPI_GLOBAL_ARGS}" --mpi-local-args="${MPI_LOCAL_ARGS}" python "${TFBERT_PATH}/run_pretraining.py" --config "${PHASE2_CONFIG_PATH}" --train-file "${PHASE2_TRAIN_FILE}" --init-checkpoint "${PHASE1_CHECKPOINT}" --wandb --wandb-name "Large phase 2 - 2 instances"  2>&1 | tee pretrain_large_384_log.txt
