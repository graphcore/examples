#!/bin/bash

# Install dependencies:
sudo -n apt -y install datamash

# Run SQUAD inference benchmarks. These use configs optimised for low latency.
# This script also uses the real-time scheduling option to increase determinism.
# Requires sudo to be runable in non-interactive mode for the real-time
# scheduling to take effect.

T_INFO_STRING="Mean Min Max (throughput in sequences/sec):"
L_INFO_STRING="Mean Min Max (latency in seconds):"
TCOL=10

DATA_DIR=$1
DATA_DIR_ARG=""

if ! [[ -n "$DATA_DIR" ]]; then
  echo "Data directory not set, using scripts default."
  DATA_DIR_ARG=""
  VOCAB_DIR_ARG=""
  EVAL_SCRIPT_ARG=""
else
  echo "Data directory set to $DATA_DIR"
  DATA_DIR_ARG="--input-files $DATA_DIR/squad/dev-v1.1.json"
  VOCAB_DIR_ARG="--vocab-file $DATA_DIR/ckpts/uncased_L-12_H-768_A-12/vocab.txt"
  EVAL_SCRIPT_ARG="--squad-evaluate-script $DATA_DIR/squad/evaluate-v1.1.py"
fi

# BASE Pipelined Inference
for d in 1 2 3 4
do
  FILE=tmp_squad_base_bs${d}_log.txt
  echo "Benchmarking BERT Base bs ${d} Inference.\nLogging to file: ${FILE}"

  python3 bert.py --config configs/squad_base_128_inference.json \
    --report-hw-cycle-count --realtime-scheduler --batch-size ${d} \
    $DATA_DIR_ARG $VOCAB_DIR_ARG $EVAL_SCRIPT_ARG > $FILE 2>&1

  echo $T_INFO_STRING
  cat $FILE | grep Iteration | datamash -W mean $TCOL min $TCOL max $TCOL
  echo $L_INFO_STRING
  cat $FILE | grep Iteration | datamash -W mean 14 min 15 max 16
  echo "Mean Latency base bs ${d}: " `cat $FILE | grep Iteration | datamash -W mean 14`
  echo ""
done

# LARGE Pipelined Inference
for d in 1 3
do
  FILE=tmp_squad_large_bs${d}_log.txt
  echo "Benchmarking BERT Large bs ${d} Inference.\nLogging to file: ${FILE}"
  python3 bert.py --config configs/squad_large_384_inference.json \
    --report-hw-cycle-count --realtime-scheduler --batch-size ${d} \
    $DATA_DIR_ARG $VOCAB_DIR_ARG $EVAL_SCRIPT_ARG > $FILE 2>&1

  echo $T_INFO_STRING
  cat $FILE | grep Iteration | datamash -W mean $TCOL min $TCOL max $TCOL
  echo $L_INFO_STRING
  cat $FILE | grep Iteration | datamash -W mean 14 min 15 max 16
  echo "Mean Latency large bs ${d}: " `cat $FILE | grep Iteration | datamash -W mean 14`
  echo ""
done
