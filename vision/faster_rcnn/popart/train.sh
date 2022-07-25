#!/bin/bash
if [ -z $1 ]
then 
    echo "Please provide the yaml that will be used for training"
    echo "Script should be called as ./train.sh path/to/yaml/file"
    exit 1 
else 
    echo "Training using yaml file: $1"
fi

python3 train.py --YAML $1 &&
python3 evaluation.py --YAML $1
