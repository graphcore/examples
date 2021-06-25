# Benchmarking on IPUs

This README describes how to run PyTorch CNN models for inference throughput benchmarking on the Mk2 IPU.

## Preparation

Follow the installation instructions in applications/pytorch/cnns/README.md

Run the following command lines from inside the applications/pytorch/cnns/inference directory.

## Inference

### ResNet-50 v1.5

#### 1 x IPU-M2000

Command:
```console
poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=4 --numa-aware=yes --num-replicas=4 --ipus-per-replica=1 python run_benchmark.py --config resnet50-mk2 --data generated --batch-size 1 --iterations 20
```

Set --batch-size to 1, 4, 16, 32, 64 or 90. 

### EfficientNet-B0 - Standard Group Dim 1

#### 1 x IPU-M2000

Command:
```console
poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=4 --numa-aware=yes --num-replicas=4 --ipus-per-replica=1 python run_benchmark.py --config efficientnet-b0-mk2 --data generated --batch-size 1 --iterations 20 --device-iterations 128
```

Set --batch-size to 1, 8, 16, 32, 36 or 40. Set --device-iterations to 128, 64 or 32. 

### EfficientNet-B4

#### 1 x IPU-M2000

Command:
```console
poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=4 --numa-aware=yes --num-replicas=4 --ipus-per-replica=1 python run_benchmark.py --data generated --batch-size 1 --model efficientnet-b4 --device-iterations 256 --norm-type batch --precision 16.16 --half-partial --eight-bit-io --normalization-location ipu --iterations 20
```

Set --batch-size to 1, 2, 4, 5 or 8. 

### ResNeXt-101

#### 1 x IPU-M2000

Command:
```console
poprun --mpi-global-args="--allow-run-as-root --tag-output" --num-instances=4 --numa-aware=yes --num-replicas=4 --ipus-per-replica=1 python run_benchmark.py --data generated --batch-size 1 --model resnext101 --device-iterations 128 --norm-type batch --precision 16.16 --half-partial --eight-bit-io --normalization-location ipu --iterations 20
```

Set --batch-size to 1, 2, 4, 8 or 16.
