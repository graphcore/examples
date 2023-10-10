<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# TensorFlow 2 Distributed Training and Inference Example

PopDist (Poplar Distributed Configuration) provides a set of APIs which can be used to
write a distributed application. The application can then be launched on multiple instances
through PopRun, our command line utility.

This example contains a TensorFlow CNN with PopDist support, which can be launched on
multiple instances using a PopRun command line.

You can learn more about PopDist and PopRun in the
[PopDist and PopRun User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/).

## File structure

* `popdist_training.py` Example training script with PopDist support.
* `popdist_inference.py` Example inference script with PopDist support.
* `tests` Integration tests for this example.
* `README.md` This file.

## How to use this example

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started
   guide for your IPU system. Make sure to source the `enable.sh` script for
   Poplar and activate a Python virtualenv with a TensorFlow 2 wheel from
   the Poplar SDK installed (use the version appropriate to your operating
   system and processor).

2) Run the script using PopRun. The number of instances and replicas are
   provided as command-line arguments. Example:

   ```
   poprun --num-instances=2 --num-replicas=4 python3 popdist_training.py
   ```

3) After training, run the inference script using PopRun. For inference, it is
   recommended that you do not use multiple replicas per instance. This is
   because you will likely become CPU bound by the data preparation done on the
   host (as each instance will only use a single CPU core). Example:

   ```
   poprun --num-instances=4 --num-replicas=4 python3 popdist_inference.py
   ```
