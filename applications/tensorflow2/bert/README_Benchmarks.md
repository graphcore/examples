# Benchmarking on IPUs

This README describes how to run BERT for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/tensorflow2/bert/README.md.

## Training

Command:
```console
        python train.py \
            --config mk2_resnet50_8k_bn_pipeline \
            --num-epochs 5 \
            --dataset-path $DATASETS_DIR/imagenet-data
```




