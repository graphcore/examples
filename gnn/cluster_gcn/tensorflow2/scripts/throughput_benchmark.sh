#!/bin/bash
# Copyright (c) 2022 Graphcore Ltd. All Rights Reserved.
set +eux
display_usage() {
	echo "Benchmark Cluster-GCN model throughput"
    echo "Usage: $0 <num seeds> <device>"
}
NUM_SEEDS=$1
DEVICE=$2
CONFIG_PATH="configs/train_ppi.json configs/train_reddit.json configs/train_arxiv.json\
             configs/train_mag.json configs/train_products.json"
PRECISION="fp32 fp16"
SPARSE_MODE="true false"
RANDOM=640
for config in $CONFIG_PATH; do
    for sparse in $SPARSE_MODE; do
        for precision in $PRECISION; do
            new_precision=$precision
            old_precision=0
            for i in $(seq 1 $NUM_SEEDS); do
                seed=$RANDOM
                if [ $precision == fp32 ] && [ $sparse == true ]; then
                    echo "Skip the float32 sparse mode, we only test float16 for sparse representation."
                else
                    if [ $new_precision != $old_precision ]; then
                        old_precision=$precision
                        log_file_name=log_${sparse}_${config:14:3}_${precision}_${seed}.txt
                        case_name="sparse_${sparse}_${config:14:3}_${precision}: "
                    else
                        log_file_name=log_${sparse}_${config:14:3}_${precision}_${seed}.txt
                        case_name=""
                    fi
                    python3 run_cluster_gcn.py $config \
                        --seed $seed \
                        --training.precision $precision \
                        --training.use-sparse-representation $sparse \
                        --training.epochs 20 \
                        --do-validation false  \
                        --do-test false \
                        --training.device $DEVICE 2>&1 | tee $log_file_name
                    grep 'Mean throughput:' $log_file_name \
                        | (echo $case_name; sed -e 's/^.*: //') | tee -a tput_benchmark.csv
                fi
            done
        done
    done
done
