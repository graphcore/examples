# Small tutorial examples

This directory contains some small code examples showing a series of concepts in a minimal amount of code.

To set up the script, first go to the examples/vision/cnns/pytorch directory and follow the instructions for creating a virtual environment and installing dependencies.
Return to the pytorch-lightning directory

Now enable the Poplar SDK:

```console
source <path-to-poplar-sdk>/popart-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
source <path-to-poplar-sdk>/poplar-ubuntu_18_04-2.4.0+2529-969064e2df/enable.sh
```

Each example can be run as follows:

```console
python3 NAME.py
```

The comments in each one describe what it is doing/showing.

There is a simple model which we use to compose the examples.
* `simple_torch_model.py` A simple model for us to use in the examples.

The examples show:
* `simple_lightning_ipu.py` Shows how to run a simple Lightning model on IPU.
* `ipu_strategy_and_dataloading.py` Shows how to pass PopTorch options. Includes tutorial on replication and dataloading.
* `pipelined_ipu.py` Shows how to pipeline a model across IPUs.
* `custom_losses.py` Shows how to add a custom loss on the IPU in a simple regression model.
