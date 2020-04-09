#!/bin/sh

# Install dependencies:
sudo -n apt -y install datamash

# Run SQUAD inference benchmarks. These use configs optimised for low latency.
# This script also uses the real-time scheduling option to increase determinism.
# Requires sudo to be runable in non-interactive mode for the real-time
# scheduling to take effect.

T_INFO_STRING="Mean Min Max (throughput in sequences/sec):"
L_INFO_STRING="Mean Min Max (latency in seconds):"
TCOL=10

# BASE Pipelined Inference
for d in 1 2 3 4
do
  FILE=tmp_squad_base_bs${d}_log.txt
  echo "Benchmarking BERT Base bs ${d} Inference.\nLogging to file: ${FILE}"

  python bert.py --config configs/squad_base_inference.json \
    --report-hw-cycle-count --realtime-scheduler --batch-size ${d} > $FILE 2>&1

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
  python bert.py --config configs/squad_large_inference.json \
    --report-hw-cycle-count --realtime-scheduler --batch-size ${d} > $FILE 2>&1

  echo $T_INFO_STRING
  cat $FILE | grep Iteration | datamash -W mean $TCOL min $TCOL max $TCOL
  echo $L_INFO_STRING
  cat $FILE | grep Iteration | datamash -W mean 14 min 15 max 16
  echo "Mean Latency large bs ${d}: " `cat $FILE | grep Iteration | datamash -W mean 14`
  echo ""
done
