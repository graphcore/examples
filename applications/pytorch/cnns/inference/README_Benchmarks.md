# Benchmarking on IPUs

This README describes how to run PyTorch CNN models for inference throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/cnns/README.md

Run the following command lines from inside the applications/pytorch/cnns/inference directory.

## Inference

In all cases, to get minimum latency performance, at the cost of some throughput, set
POPLAR_ENGINE_OPTIONS: '{"exchange.enablePrefetch": "false"}'

### ResNet-50 v1.5

#### 1 x IPU-M2000 

Command:
```console 
    poprun \
      --mpi-global-args="--allow-run-as-root --tag-output" \
      --mpi-local-args="-x POPLAR_ENGINE_OPTIONS" \
      --num-instances=4 \
      --numa-aware=yes \
      --num-replicas=4 \
      --ipus-per-replica=1 \
    python run_benchmark.py \
      --config resnet50-mk2 \
      --data generated \
      --batch-size {batchsize} \
      --iterations 20 \
      --num-io-tiles 64
```
Set --batch-size to 1, ... 80

### EfficientNet-B0 - Standard Group Dim 1

#### 1 x IPU-M2000

Command:
```console
    poprun \
      --mpi-global-args="--allow-run-as-root --tag-output" \
      --mpi-local-args="-x POPLAR_RUNTIME_OPTIONS" \
      --num-instances=4 \
      --numa-aware=yes \
      --num-replicas=4 \
      --ipus-per-replica=1 \
    python run_benchmark.py \
      --config efficientnet-b0-mk2 \
      --data generated \
      --batch-size {batchsize} \
      --iterations 20 \
      --num-io-tiles 64
```

Set --batch-size to 1, ... 48. 

### EfficientNet-B4 - Standard Group Dim 1

#### 1 x IPU-M2000

Command:
```console
    poprun \
      --mpi-global-args="--allow-run-as-root --tag-output" \
      --mpi-local-args="-x POPLAR_RUNTIME_OPTIONS" \
      --num-instances=4 \
      --numa-aware=yes \
      --num-replicas=4 \
      --ipus-per-replica=1 \
    python run_benchmark.py \
      --config efficientnet-b4-mk2 \
      --data generated \
      --batch-size {batchsize} \
      --iterations 20 \
      --num-io-tiles 0
```

Set --batch-size to 1, ... 10. 

### ResNeXt-101

#### 1 x IPU-M2000

Command:
```console
    poprun \
      --mpi-global-args="--allow-run-as-root --tag-output" \
      --mpi-local-args="-x POPLAR_RUNTIME_OPTIONS" \
      --num-instances=4 \
      --numa-aware=yes \
      --num-replicas=4 \
      --ipus-per-replica=1 \
    python run_benchmark.py \
      --data generated \
      --batch-size {batchsize} \
      --model resnext101 \
      --device-iterations 128 \
      --norm-type batch \
      --precision 16.16 \
      --half-partial \
      --eight-bit-io \
      --normalization-location ipu \
      --iterations 20 \
      --num-io-tiles 32
```

Set --batch-size to 1, ... 16.
