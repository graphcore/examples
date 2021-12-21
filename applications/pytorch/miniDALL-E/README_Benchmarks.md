# Benchmarking on IPUs

This README describes how to run the MiniDALL-E model for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/pytorch/miniDALL-E/README.md.

## Training

Command:
```console
        python3 train.py \
            --config L16 \
            --input-folder $DATASETS_DIR/coco \
            --checkpoint-output-dir "" \
            --gradient-accumulation 512 \
            --epochs 2
```

for more replicase add --replication-factor x
