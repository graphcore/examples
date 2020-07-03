#!/bin/bash

# Pre-processes a batch of pretrained checkpoints, then runs SQuAD fine-tuning on them and validates the result.

MODEL_DIR=$1
CONFIG=$2

if [ "$#" -ne 2 ]; then
    echo "Usage: tools/run_pretraining_squad.sh <ckpt_output_dir> <squad_config_path>"
    exit
fi

python3 tools/transpose_embeddings_only.py --config $CONFIG --model-dir $MODEL_DIR --output-dir $MODEL_DIR/transposed_embeddings --num-processes 48
python3 tools/bert_multi_checkpoint_squad.py --config $CONFIG --checkpoint-dir $MODEL_DIR/transposed_embeddings
python3 tools/bert_multi_checkpoint_validation.py --config $CONFIG --checkpoint-dir $MODEL_DIR/transposed_embeddings/squad_output

echo "Results output to ${MODEL_DIR}/transposed_embeddings/squad_output/validation_result.json"
