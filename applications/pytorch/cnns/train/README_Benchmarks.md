# Benchmarking on IPUs

This README describes how to run PyTorch CNN models for training throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/cnns.
Follow the instructions in applications/pytorch/cnns/datasets/README.md to create a directory containing the ImageNet dataset in the WebDataset format.
Set the DATASETS_DIR environment variable to the parent directory of the dataset.

Run the following command lines from inside the applications/pytorch/cnns/training directory.

The first value reported will always be lower than the subsequent values. When calculating an average throughput, remove the first reported result from the calculation in order to get a fair indication of the throughput of a full training run.

## Training

### ResNet-50 v1.5

#### 1 x IPU-M2000

Command:
```console
python train.py --config resnet50_mk2_pipelined --imagenet-data-path $DATASETS_DIR/imagenet-webdata-img --replicas 1 --gradient-accumulation 1024 --epoch 2 --validation-mode none --disable-metrics
```

#### 1 x IPU-POD16

Command:
```console
poprun --mpi-global-args="--allow-run-as-root --tag-output" --numa-aware 1 --ipus-per-replica 4 --num-instances 4 --num-replicas 4 python train.py --config resnet50_mk2_pipelined --imagenet-data-path $DATASETS_DIR/imagenet-webdata-img --epoch 2 --validation-mode none --disable-metrics
```

#### 1 x IPU-POD64

Command:
```console
./rn50_pod64.sh --wandb --imagenet-data-path $DATASETS_DIR/imagenet-webdata-img --checkpoint-path checkpoints/pod64_convergence
```
