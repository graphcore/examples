#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
set +eux
display_usage() {
	echo "Fine-tune BERT on IPU-POD16"
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

if [[ $2 != "v1" ]]&&[[ $2 != "v2" ]]; then
    echo "Expecting SQuAD version to be given."
    echo "Specify 'v1' or 'v2'"
    exit 1
fi

export TF_POPLAR_FLAGS="--executable_cache_path=/localdata/$USER/exec_cache"
export POPLAR_LOG_LEVEL=INFO

# Fine-tuning from pretrained checkpoint
LAST_PHASE2_CKPT_FILE=$(find "checkpoint/phase2/" -name "ckpt-2098*" | tail -1)
PHASE2_CKPT_DIR=$(dirname "${LAST_PHASE2_CKPT_FILE}")
PHASE2_CKPT_FILE="${PHASE2_CKPT_DIR}/ckpt-2098"

WANDB_NAME="SQuAD $1 fine tuning $2"
echo "Using pretrained checkpoint: ${PHASE2_CKPT_FILE}"


if [[ $1 == 'large' ]]; then
	if [[ $2 == 'v1' ]]; then
		CONFIG_PATH='configs/squad_large.json'
	elif [[ $2 == 'v2' ]]; then
		CONFIG_PATH='configs/squad_large_V2.json'
	else
		exit -1
	fi
elif [[ $1 == 'base' ]]; then
	if [[ $2 == 'v1' ]]; then
		CONFIG_PATH='configs/squad_base.json'
	elif [[ $2 == 'v2' ]]; then
		CONFIG_PATH='configs/squad_base_V2.json'
	else
		exit -1
	fi
else 
	exit -1
fi

./run_squad.py --config "${CONFIG_PATH}" --do-training --do-predict --do-evaluation --init-checkpoint "${PHASE2_CKPT_FILE}" --wandb --wandb-name "${WANDB_NAME}"
