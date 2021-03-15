# CNN Benchmarking on IPUs

This README describes how to run PyTorch CNN models for inference throughput benchmarking on the Mk2 IPU.

### PyTorch CNNs Inference

Follow the installation instructions in applications/pytorch/cnns/README.md

Run the following command lines from inside the applications/pytorch/cnns/inference directory.

#### ResNet-50 v1.5

1 x IPU

```
python run_benchmark.py --data synthetic --batch-size 1 --model resnet50 --device-iterations 128 --norm-type batch --precision 16.16 --half-partial --iterations 20
```

Change the --batch-size argument to be one of 1, 4, 16, 32, 64, 90

#### EfficientNet-B0 - Standard (Group Dim 1)

1 x IPU

```
python run_benchmark.py --data synthetic --batch-size 1 --model efficientnet-b0 --device-iterations 128 --norm-type batch --precision 16.16 --half-partial --iterations 20
```

Change the --batch-size argument to be one of 1, 8, 16, 32, 36, 40
