<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Distributed Training using PopDist Example

PopDist (Poplar Distributed Configuration) provides a set of APIs which can be used to write a distributed application. The application can then be launched on multiple instances using PopRun.

This example contains a PopTorch CNN with PopDist support, which can be launched on multiple instances using a PopRun command line.

You can learn more about PopDist and PopRun in the [PopDist and PopRun User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/en/3.3.0/index.html).

## File structure

* `popdist_training.py` Example training script with PopDist support.
* `tests` Integration tests for this example.
* `README.md` This file.

## How to use this example

1. Prepare the PopTorch environment. Install the Poplar SDK following the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU system. Make sure to source the `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.
2. Install additional Python packages specified in requirements.txt
```:bash
python -m pip install -U pip
python3 -m pip install -r requirements.txt
```
3. Launch the script using PopRun. The number of instances and replicas are provided as command-line arguments
Example:
```
poprun --num-instances=2 --num-replicas=4 python3 popdist_training.py --epochs=10
```
