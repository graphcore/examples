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

if [[ $1 != "base" ]]&&[[ $1 != "large" ]]; then
    echo "Expecting model archetecture to be given."
    echo "Specify 'base' or 'large'"
    exit 1
fi

if [[ $1 == 'base' ]]; then
    if [[ $2 == 'stsb' ]]; then
    	CONFIG_PATH='configs/groupbert/glue_base_regression.json'	  
    else	
    	CONFIG_PATH='configs/groupbert/glue_base.json'
    fi
elif [[ $1 == 'large' ]]; then
    if [[ $2 == 'stsb' ]]; then
        CONFIG_PATH='configs/groupbert/glue_large_regression_8IPU.json'
    else
        CONFIG_PATH='configs/groupbert/glue_large_8IPU.json'
    fi
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
if [[ $1 == 'base' ]]; then
	LAST_PHASE2_CKPT_FILE=$(find "checkpoint/phase2/groupbert/base/" -name "model.ckpt-207505*" | tail -1)
elif [[ $1 == 'large' ]]; then
	LAST_PHASE2_CKPT_FILE=$(find "checkpoint/phase2/groupbert/large/" -name "model.ckpt-207505*" | tail -1)
fi
PHASE2_CKPT_DIR=$(dirname "${LAST_PHASE2_CKPT_FILE}")
PHASE2_CKPT_FILE="${PHASE2_CKPT_DIR}/model.ckpt-207505"
WANDB_NAME="GLUE $1 fine tuning ${DATA_NAME} ${CUSTOM_NAME}"
echo "Using pretrained checkpoint: ${PHASE2_CKPT_FILE}"

./run_classifier.py --config "${CONFIG_PATH}" --do-training --do-eval --do-predict --data-dir glue_data/"${DATA_NAME}"/ --task-name $2 --output-dir tmp/groupbert/$2/  --wandb --wandb-name "${WANDB_NAME}" --init-checkpoint "${PHASE2_CKPT_FILE}"
