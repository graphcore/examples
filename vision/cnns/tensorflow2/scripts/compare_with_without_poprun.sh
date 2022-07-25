#!/bin/bash

SEED=$((1 + $RANDOM % 1000000))
CONFIG="resnet50_64ipus_8k_bn_pipeline"
TOTAL_IPUS=16
NUM_INSTANCES=4

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift; shift
      ;;
    --seed)
      SEED="$2"
      shift; shift
      ;;
    --total-ipus)
      TOTAL_IPUS="$2"
      shift; shift
      ;;
    --num-instances)
      NUM_INSTANCES="$2"
      shift; shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ ${#POSITIONAL_ARGS} -gt 0 ]; then echo "Error. Unknown arguments: $POSITIONAL_ARGS[@]"; exit 1; fi

# first train then validate each instance separately
distributed="bash scripts/run_experiment.sh --config ${CONFIG} --seed ${SEED} --num-instances $NUM_INSTANCES --total-ipus $TOTAL_IPUS --ckpt-all-instances"
echo $distributed
eval $distributed

# then train and evaluate without poprun
non_distributed="bash scripts/run_experiment.sh --config ${CONFIG} --seed ${SEED} --total-ipus $TOTAL_IPUS --poprun-off"
echo $non_distributed
eval $non_distributed
