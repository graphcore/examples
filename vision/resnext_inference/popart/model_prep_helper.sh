#!/bin/bash
set -eo pipefail

PATH_TO_COCO_DATASET=$1
PATH_TO_PARTITIONED_DATASET=$2
PATH_TO_MODEL=$3

if [[ -z "$PATH_TO_PARTITIONED_DATASET" ]]; then
    echo "Output path not set, setting to current directory."
    PATH_TO_PARTITIONED_DATASET="datasets"
fi

if [[ -n "$PATH_TO_MODEL" ]]; then
    echo "Model path set to $PATH_TO_MODEL"
    MODEL_PATH_ARGS="--model-path $PATH_TO_MODEL --pretrained-model-path $PATH_TO_MODEL"
else
    MODEL_PATH_ARGS=""
fi

if [[ -n "$PATH_TO_COCO_DATASET" ]]; then
    cd "${0%/*}" || exit

    echo "Creating and activating virtual environment"
    virtualenv -p python3 resnext_data_virtualenv
    source resnext_data_virtualenv/bin/activate    

    echo "Installing required python packages"
    pip3 install -r requirements.txt

    echo "Making copies of Coco dataset in directory $PATH_TO_PARTITIONED_DATASET"
    echo "Coco dataset: $PATH_TO_COCO_DATASET"
    for i in {1,2,8}
    do
        DATASET_PART_PATH="$PATH_TO_PARTITIONED_DATASET/dataset$i"
        if [[ -d "$DATASET_PART_PATH" ]]; then
            rm -r "$PATH_TO_PARTITIONED_DATASET/dataset$i"
        fi
        mkdir -p "$DATASET_PART_PATH"

        python3 setup_dataset.py --data-dir "$PATH_TO_COCO_DATASET" \
                                     --copies "$i" \
                                     --output "$PATH_TO_PARTITIONED_DATASET/dataset$i"
    done

    echo "Getting pretrained model"
    for i in {16, 32, 64}
    do
        python3 get_model.py --micro-batch-size "$i" $MODEL_PATH_ARGS
    done

    echo "Deactivating virtual environment"
    deactivate
else
    echo "Must provide the path to the coco dataset as the first argument"
    exit 22
fi
