# PyTorch CNN application

This directory contains a PyTorch Lightning port of the [PyTorch CNNs training application](../pytorch/train).

It can be used to verify that the same performance is obtained with PyTorch Lightning as we do when running directly in PopTorch.

### How to use this demo

Go to the examples/vision/cnns/pytorch directory and follow the instructions for creating a virtual environment and installing dependencies.
Return to the pytorch-lightning directory

Enable the Poplar SDK (pick desired SDK version):

```console
source <path-to-poplar-sdk>/popart-ubuntu_...../enable.sh
source <path-to-poplar-sdk>/poplar-ubuntu_...../enable.sh
```

See the README.md in examples/applications/pytorch/cnns for full documentation, including command line options
To get started:

Using generated data
```console
python3 train_lightning.py --data generated --config resnet50
```

Using imagenet data
```console
python3 train_lightning.py --config resnet50 --imagenet-data-path <path-to/imagenet>
```

see ../pytorch/train/config.yml for available configs


## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```console
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```console
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).
