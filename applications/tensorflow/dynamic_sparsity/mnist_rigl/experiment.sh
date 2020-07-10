#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

SEED=101
DENSITY1=0.01
DENSITY2=1
BS=20
STEPS=4
EPOCHS=10
SPARSE_DROP=0.1

export PYTHONPATH=./

echo "" > experiment.out

# Run 4 variations in parallel (requires one available IPU each):

python mnist_rigl/sparse_mnist.py --batch-size=$BS --epochs=$EPOCHS --steps-per-epoch=$STEPS \
--densities 1 1 --log dense_acc.log --seed $SEED 2>&1 | tee -a experiment.out &

mkdir -p static_records
rm static_records/*
python mnist_rigl/sparse_mnist.py --batch-size=$BS --epochs=$EPOCHS --steps-per-epoch=$STEPS \
--densities $DENSITY1 $DENSITY2 --log static_records/acc.log --seed $SEED --records-path static_records \
--disable-pruning --droprate $SPARSE_DROP 2>&1 | tee -a experiment.out &

mkdir -p rand_records
rm rand_records/*
python mnist_rigl/sparse_mnist.py --batch-size=$BS --epochs=$EPOCHS --steps-per-epoch=$STEPS \
--densities $DENSITY1 $DENSITY2 --log rand_records/acc.log --seed $SEED  --records-path rand_records \
--regrow random --droprate $SPARSE_DROP 2>&1 | tee -a experiment.out &

mkdir -p rigl_records
rm rigl_records/*
python mnist_rigl/sparse_mnist.py --batch-size=$BS --epochs=$EPOCHS --steps-per-epoch=$STEPS \
--densities $DENSITY1 $DENSITY2 --log rigl_records/acc.log --seed $SEED  --records-path rigl_records \
--regrow rigl --droprate $SPARSE_DROP 2>&1 | tee -a experiment.out &

wait

# Combine all the training accuracies into one plot:
paste -d ' ' dense_acc.log static_records/acc.log rand_records/acc.log rigl_records/acc.log | cut -d ' ' -f 1,2,4,6,8 > results.txt

python mnist_rigl/result_plotter.py --input results.txt \
--title "Sparse FC MNIST Training with Densities $DENSITY1 $DENSITY2" --xlabel "Batches Processed (batch-size=$BS)" \
--ylabel "Accuracy Over Last Step (steps-per-epoch=$STEPS)" --output rigl_results_${DENSITY1}_${DENSITY2}.png  --cmap cividis \
--headers Dense Sparse_Static Sparse_Grow_Random Sparse_Grow_RigL 2>&1 | tee -a experiment.out 

# Plot the connectivity to input pixels for initial and final networks:
python mnist_rigl/visualise_connectivity.py --records-path static_records
python mnist_rigl/visualise_connectivity.py --records-path rand_records
python mnist_rigl/visualise_connectivity.py --records-path rigl_records