#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
set +eux
display_usage() {
	echo "Fine-tune BERT on IPU-POD16 for GLUE Tasks"
	echo "Usage: $0 <model size> <task>"
	echo "model: large / base to specify BERT architecture."
}

if [[ $# -lt 1 ]]; then
   display_usage
   exit 1
fi

if [[ $1 != "base" ]]&&[[ $1 != "large" ]]&&[[ $1 != "tiny" ]]; then
    echo "Expecting model archetecture to be given."
    echo "Specify 'base' or 'large'"
    exit 1
fi

if [[ $1 == 'base' ]]; then
    CONFIG_PATH='configs/glue_base.json'
elif [[ $1 == 'large' ]]; then
    if [[ $2 == 'stsb' ]]; then
        CONFIG_PATH='configs/glue_large_regression.json'
    else
        CONFIG_PATH='configs/glue_large.json'
    fi
elif [[ $1 == 'tiny' ]]; then
    CONFIG_PATH='configs/glue_tiny.json'
fi

if [[ $2 == 'cola' ]]; then
    DATA_NAME="CoLA"
elif [[ $2 == 'mrpc' ]]; then
    DATA_NAME="MRPC"
elif [[ $2 == 'mnli' ]]; then
    DATA_NAME="MNLI"
elif [[ $2 == 'mnli-mm' ]]; then
    DATA_NAME="MNLI"
elif [[ $2 == 'ax' ]]; then
    DATA_NAME="AX"        
elif [[ $2 == 'qnli' ]]; then
    DATA_NAME="QNLI"
elif [[ $2 == 'qqp' ]]; then
    DATA_NAME="QQP"
elif [[ $2 == 'rte' ]]; then
    DATA_NAME="RTE"
elif [[ $2 == 'sst2' ]]; then
    DATA_NAME="SST-2"
elif [[ $2 == 'stsb' ]]; then
    DATA_NAME="STS-B"
elif [[ $2 == 'wnli' ]]; then
    DATA_NAME="WNLI"
else
    DATA_NAME=''
fi
CUSTOM_NAME=$3

export TF_POPLAR_FLAGS="--executable_cache_path=/localdata/$USER/exec_cache"
# export POPLAR_LOG_LEVEL=INFO
# export POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "autoReport.all": "true", "autoReport.directory":"./profiles/'${CUSTOM_NAME}'"}'
export XLA_FLAGS="--xla_disable_hlo_passes=embeddings-gradient-optimizer"


# Fine-tuning from pretrained checkpoint
LAST_PHASE2_CKPT_FILE=$(find "checkpoint/phase2/" -name "ckpt-2000*" | tail -1)
PHASE2_CKPT_DIR=$(dirname "${LAST_PHASE2_CKPT_FILE}")
PHASE2_CKPT_FILE="${PHASE2_CKPT_DIR}/ckpt-2098"
WANDB_NAME="GLUE $1 fine tuning ${DATA_NAME} ${CUSTOM_NAME}"
echo "Using pretrained checkpoint: ${PHASE2_CKPT_FILE}"

./run_classifier.py --config "${CONFIG_PATH}" --do-training --do-evaluation --do-predict --data-dir glue_data/"${DATA_NAME}"/ --task-name $2 --output-dir tmp/$2/  --wandb --wandb-name "${WANDB_NAME}" --init-checkpoint "${PHASE2_CKPT_FILE}"
