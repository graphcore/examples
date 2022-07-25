#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.

# Pretrain BERT Large on POD16
set +eux
display_usage() {
	echo "Pretrain BERT on IPU-POD16"
	echo "Usage: $0 model"
	echo "model: large / base to specify BERT architecture."
}

if [[ $# -lt 1 ]]; then
   display_usage
   exit 1
fi

if [[ $1 != "base" ]]&&[[ $1 != "large" ]]; then
    echo "Expecting model archetecture to be given."
    echo "Specify 'base' or 'large'"
    exit 1
fi

# Get the real directory path, avoiding symlinks, etc.
cd "$(realpath .)"
# Poplar options
export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "600"}' ;
export POPLAR_LOG_LEVEL=INFO
export TF_POPLAR_FLAGS="--executable_cache_path=/localdata/$USER/exec_cache"

# Specify the wandb names
WANDB_NAME_PHASE1="BERT $1 Phase 1"
WANDB_NAME_PHASE2="BERT $1 Phase 2"

# BERT training options
# Phase 1
if [[ $1 == 'large' ]]; then
    PHASE1_CONFIG_PATH='configs/pretrain_large_128_phase1.json'
elif
    PHASE1_CONFIG_PATH='configs/pretrain_base_128_phase1.json'
else
    echo "Unknown model size"
    exit 1
fi
PHASE1_TRAIN_FILE='data/tf_wikipedia/tokenised_128_dup5_mask20/*.tfrecord'

python3 run_pretraining.py --config "${PHASE1_CONFIG_PATH}" --train-file "${PHASE1_TRAIN_FILE}" --wandb --wandb-name "${WANDB_NAME_PHASE1}" 2>&1 | tee pretrain_large_128_log.txt

# Phase 2
if [[ $1 == 'large' ]]; then
    PHASE2_CONFIG_PATH='configs/pretrain_large_384_phase2.json'
elif
    PHASE2_CONFIG_PATH='configs/pretrain_base_384_phase2.json'
else
    echo "Unknown model size"
    exit 1
fi
PHASE2_TRAIN_FILE='data/tf_wikipedia/tokenised_384_dup5_mask56/*.tfrecord'

LAST_CHANGED_CKPT_FILE=$(find "checkpoint/phase1/" -name "ckpt-7031*" | tail -1)
echo "Using phase 1 checkpoint:"
ls -l ${LAST_CHANGED_CKPT_FILE}
PHASE1_CHECKPOINT_DIR=$(dirname "${LAST_CHANGED_CKPT_FILE}")
PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT_DIR}/ckpt-7031"

python3 run_pretraining.py --config "${PHASE2_CONFIG_PATH}" --train-file "${PHASE2_TRAIN_FILE}" --init-checkpoint "${PHASE1_CHECKPOINT}" --wandb --wandb-name "${WANDB_NAME_PHASE2}"
