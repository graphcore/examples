#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

if [ $# -ne 1 ]; then
    echo "Usage: $0 RESULTS_DIR"
    exit -1
fi

DIR=$1

mkdir $DIR
if [ $? -ne 0 ]; then
  exit -1
fi

echo Writing results to: $DIR
cp $0 $DIR/

export PYTHONPATH=./

# Hyper params:
SEED=101
DENSITY1=0.01
DENSITY2=1
BS=16
STEPS=5
EPOCHS=10
SPARSE_DROP=0.1
OPT="Adam --optimizer-arg epsilon=1e-02"
DTYPE=fp16
PTYPE=fp32
PRUNE=0.8
REGROW="--regrow rigl"
PERMUTE=none
HIDDEN=320

# Run fully dense for reference:
REC=$DIR/mnist_block_sweep_dense
mkdir -p $REC
python mnist_rigl/sparse_mnist.py --batch-size=$BS --epochs=$EPOCHS --steps-per-epoch=$STEPS \
--densities 1 1 --log $REC/acc.log --seed $SEED --optimizer $OPT --data-type=$DTYPE --hidden-size=$HIDDEN 2>&1 | tee -a $REC/run.out &

# Run the experiments in parallel:
for BLOCKS in 1 4 8 16; do
echo Launching block size: $BLOCKS
  REC=$DIR/mnist_block_sweep_${BLOCKS}x${BLOCKS}
  mkdir -p $REC
  rm -rf $REC/*
  python3 mnist_rigl/sparse_mnist.py --block-size $BLOCKS --batch-size=$BS --epochs=$EPOCHS --steps-per-epoch=$STEPS \
  --densities $DENSITY1 $DENSITY2 --log $REC/acc.log --seed $SEED --optimizer $OPT \
  --data-type=$DTYPE --records-path $REC \
  $REGROW --droprate $SPARSE_DROP --prune-ratio $PRUNE \
  --partials-type=$PTYPE --permute-input=$PERMUTE --hidden-size=$HIDDEN 2>&1 | tee -a $REC/run.out &
done

wait

# Make the pretty pictures:
for BLOCKS in 1 4 8 16; do
  REC=$DIR/mnist_block_sweep_${BLOCKS}x${BLOCKS}
  python3 mnist_rigl/visualise_connectivity.py --records-path $REC
  python mnist_rigl/visualise_connectivity.py --records-path $REC --animate
  convert -delay 20 $REC/connectivity_*.png $REC/connection_evolution_${BLOCKS}x${BLOCKS}.gif
  rm $REC/connectivity_*.png
done
