#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

export PYTHONPATH=./

REGROW=random

for d1 in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 1
do

  for d2 in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01
  do
    python mnist_rigl/sparse_mnist.py --regrow $REGROW --seed 101 --densities $d1 $d2 2>&1 | tee sweep_experiment_${d1}_${d2}.out &
  done

wait

done

grep -H "Test acc" sweep_experiment* | cut -d ' ' -f 1,8,9,10 | awk '{print gensub(".out"," ","g",$0)}' | awk 'BEGIN { FS="[ _]" } ; { print $3,$4,$8 }' > mnist_density_sweep.txt
gnuplot mnist_rigl/plot_grid.gnuplot
