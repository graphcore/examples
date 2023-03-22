#!/bin/bash
# Copyright (c) 2022 Graphcore Ltd. All Rights Reserved.

# This script can be used to pretrain BERT base or large on IPU-POD64

# Directories of sdk, ven, dataset, and profile folder
# ------------------------------------
SDK_PATH="/localdata/${USER}/sdks"
VENV_PATH="/localdata/${USER}/venvs"
WIKIPEDIA_DATA_PATH="$DATASETS_DIR/wikipedia/128"
LOCAL_HOME="/localdata/${USER}"
PROFILE_PATH="profile"


# Parse options
# -------------
display_usage() {
	echo "Pretrain BERT on IPU-POD64"
	echo "Run from your local machine."
	echo "Usage: $0 MODEL POD_SIZE VIPU_ALLOCATION_NAME VIPU_PARTITION_NAME VIPU_HOST HOST1 HOST2 ..."
	echo "    MODEL: \"base\" or \"large\"."
	echo "    POD_SIZE: \"POD64\"."
	echo "    VIPU_ALLOCATION_NAME: Name of the VIPU allocation."
	echo "    VIPU_PARTITION_NAME: Name of the VIPU partition."
	echo "    VIPU_HOST: VIPU IP address."
	echo "    HOSTn: blank space separated list of IP addresses on which the instances will be run separated."
}

if [[ $# -lt 6 ]]; then
   display_usage
   exit 1
fi
# First arguments are:
# 1. Model type;
# 2. Pod size;
# 3. VIPU allocation name;
# 4. VIPU partition name;
# 5. VIPU server IPU address;
MODEL="$1"
POD_SIZE="$2"
VIPU_ALLOCATION_NAME="$3"
VIPU_PARTITION_NAME="$4"
VIPU_SERVER_IP="$5"

# Parse rest of the arguments as a list of hosts.
shift 5
HOSTS="$@"

NUM_HOSTS="${#@}"

HOSTS_LIST="${HOSTS[@]}"
HOSTS_LIST="${HOSTS_LIST// /,}"

NETIF_NETMASK__IP_PART=(${HOSTS[0]//./ })
TCP_IF_NETMASK="${NETIF_NETMASK__IP_PART[0]}.${NETIF_NETMASK__IP_PART[1]}.0.0/16"

if [ "${MODEL}" != "base" ] && [ "${MODEL}" != "large" ]; then
   echo "Unknown model. Must be 'base' or 'large'"
   exit 1
fi

if [[ "${POD_SIZE}" == "POD64" ]]; then
   NUM_REPLICAS=16
   NUM_ILDS=1
   NUM_INSTANCES=${NUM_HOSTS}
else
   echo "Unknown configuration. Must be 'POD64'"
   exit 1
fi
PHASE1_CONFIG_PATH='configs/pretrain_'"${MODEL}"'_128_phase1_'"${POD_SIZE}"'.json'

echo "Running ${NUM_INSTANCES} instances of BERT ${MODEL} on hosts ${HOSTS_LIST} of ${POD_SIZE} with cluster ${VIPU_ALLOCATION_NAME} and partition ${VIPU_PARTITION_NAME}"

# Synchronise code, SDK, Python venv, and wikipedia dataset
# ---------------------------------------------------------
# Code directory obtained relative to this script.
TFBERT_PATH=$( cd -- "$( dirname "$( dirname -- "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )
TFBERT_PARENT_PATH=$( dirname "${TFBERT_PATH}")

VENV_PARENT_PATH=$( dirname "${VENV_PATH}")
SDK_PARENT_PATH=$( dirname "${SDK_PATH}")
WIKIPEDIA_DATA_PARENT_PATH=$( dirname "${WIKIPEDIA_DATA_PATH}")

echo "Directories:"
echo "    code: ${TFBERT_PATH}"
echo "    sdk: ${SDK_PATH}"
echo "    venv: ${VENV_PATH}"

copy_ssh() {
   for host in ${HOSTS}; do
      ssh-copy-id "${host}"
   done
}

sync_venvs_and_code() {
   for host in ${HOSTS}; do
      echo; echo "Syncing local code at ${TFBERT_PATH} with host ${host}"
      rsync --stats --exclude bert/checkpoints --exclude wandb --exclude "${PROFILE_PATH}/*" -av --rsync-path="mkdir -p ${TFBERT_PARENT_PATH} && rsync" "${TFBERT_PATH}" "${host}:${TFBERT_PARENT_PATH}"
      echo; echo "Syncing local venv at ${VENV_PATH} with host ${host}"
      rsync --stats -av "${VENV_PATH}" "${host}:${VENV_PARENT_PATH}/"
      echo; echo "Syncing local SDK at ${SDK_PATH} with host ${host}"
      rsync --stats -av "${SDK_PATH}" "${host}:${SDK_PARENT_PATH}/"
   done
}

sync_wikipedia_dataset() {
   for host in ${HOSTS}; do
      echo; echo "Syncing Wikipedia Dataset at ${WIKIPEDIA_DATA_PATH} with host ${host}"
      rsync --stats -av --copy-dirlinks "${WIKIPEDIA_DATA_PATH}" "${host}:${WIKIPEDIA_DATA_PARENT_PATH}"
   done
}

cd "$(realpath .)"  # Get the real directory path, avoiding symlinks, etc.

copy_ssh
sync_venvs_and_code
sync_wikipedia_dataset

TIMESTAMP="$(date +%s)"

# Build and run poprun command
# ----------------------------
# MPI options
MPI_GLOBAL_ARGS="--mca btl_tcp_if_include ${TCP_IF_NETMASK} --mca oob_tcp_if_include ${TCP_IF_NETMASK}"

POPRUN_OPTIONS="-vv --host ${HOSTS_LIST} --num-ilds ${NUM_ILDS} --num-instances ${NUM_INSTANCES} --num-replicas ${NUM_REPLICAS} --ipus-per-replica 4 --vipu-server-host=${VIPU_SERVER_IP} --vipu-allocation=${VIPU_ALLOCATION_NAME} --reset-partition=yes --update-partition=yes --remove-partition=no --vipu-partition=${VIPU_PARTITION_NAME}"

set -x
TF_POPLAR_FLAGS=--executable_cache_path=${LOCAL_HOME}/exec_cache \
poprun ${POPRUN_OPTIONS} \
   --mpi-global-args="${MPI_GLOBAL_ARGS}" \
   python "${TFBERT_PATH}/run_pretraining.py"  "--config" "${PHASE1_CONFIG_PATH}" 2>&1 | tee "pretrain_""${MODEL}""_128_log_${TIMESTAMP}.txt"

phase1_ret_code="$?"
echo "Phase 1 exit code: ${phase1_ret_code}"
if [[ ${phase1_ret_code} != 0 ]]; then
   echo "Phase 1 run failed. Exiting."
   exit $?
fi
