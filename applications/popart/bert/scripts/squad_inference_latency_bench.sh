#!/bin/sh

# Run SQUAD inference becnhmarks. These use configs optimised for low latency.
# They also run with real-time scheduling and numactl to increase determinism.
# The script requires sudo to runable in non-interactive mode for the real-time
# scheduling to take effect.

T_INFO_STRING="Mean Min Max (throughput in sequences/sec):"
L_INFO_STRING="Mean Min Max (latency in seconds):"
TCOL=10
NUMA="numactl --cpunodebind=0"

# BASE Pipelined Inference
FILE=tmp_squad_base_log.txt
echo "Benchmarking BERT Base Inference..."
$NUMA python bert.py --config configs/squad_base_inference.json > $FILE 2>&1 &
# On Ubuntu the following returns the python PID not the numctl PID:
PID=$!
# Have to change scheduling after starting python otherwise python will run in
# sudo's env which will not be setup correctly:
sudo -n chrt --fifo -p 99 $PID
# Use tail to wait for process to finish
tail --pid $PID -f /dev/null

echo $T_INFO_STRING
cat $FILE | grep Iteration | datamash -W mean $TCOL min $TCOL max $TCOL
echo $L_INFO_STRING
cat $FILE | grep Iteration | datamash -W mean 14 min 15 max 16
echo "Mean Latency base: " `cat $FILE | grep Iteration | datamash -W mean 14`

# LARGE Pipelined Inference
FILE=tmp_squad_large_log.txt
echo "Benchmarking BERT Large Inference..."
$NUMA python bert.py --config configs/squad_large_inference.json > $FILE 2>&1 &
# On Ubuntu the following returns the python PID not the numctl PID:
PID=$!
# Have to change scheduling after starting python otherwise python will run in
# sudo's env which will not be setup correctly:
sudo -n chrt --fifo -p 99 $PID
# Use tail to wait for process to finish
tail --pid $PID -f /dev/null

echo $T_INFO_STRING
cat $FILE | grep Iteration | datamash -W mean $TCOL min $TCOL max $TCOL
echo $L_INFO_STRING
cat $FILE | grep Iteration | datamash -W mean 14 min 15 max 16
echo "Mean Latency large: " `cat $FILE | grep Iteration | datamash -W mean 14`
