#!/bin/bash
if [ -z $1 ]
then 
    echo "Please provide the yaml config that will be used for evaluation."
    echo "Script should be called as ./evaluation.sh path/to/yaml/file path/to/downloaded/data"
    exit 1 
else 
    echo "Evaluation using yaml file: $1"
fi 

if [ -z $2 ]
then 
    echo "Please provide a path to the downloaded data"
    echo "Script should be called as ./evaluation.sh path/to/yaml/file path/to/downloaded/data"
    exit 1
else
    echo "Using data downloaded to: $2"
fi 

python3 evaluation.py --YAML $1 --DATA-DIR $2 &&
python3 evaluation.py --YAML $1 --DATA-DIR $2 --EVAL-MODEL-NAME best
