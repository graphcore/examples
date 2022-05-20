# Benchmarking on IPUs

This README describes how to run the fastspeech2 model for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/tensorflow2/fastspeech2/README.md.

## Training

Command:
```console
        python3 train.py \
            --config config/fastspeech2.json \
            --train \
            --generated-data \
            --batch-size 2 \
            --replicas {replicas} \
            --gradient-accumulation-count {ga_count} \
            --epochs 2 \
            --available-memory-proportion 0.1
```

for: [replicas, ga_count] [2, 8], and [8, 4]


## Inference

Command:
```console
    python3 infer.py \
            --config config/fastspeech2.json \
            --generated-data \
            --batch-size {batch_size} \
            --replicas 1 \
            --epochs 10 \
            --steps-per-batch 100 \
            --precision {prec}
```

for: [batch_size, prec] [1, 16], and [1, 32]