# Graphcore

---
## PopART Phased Execution demo

This example runs a network in inference mode over two IPUs by splitting it in several execution phases, using PopART.
This feature allows to keep the weights in Streaming Memory, loading them in In-Processor Memory only when needed. When one IPU is computing a part of the model the other IPU in parallel can communicate with the host (sending/receiving to/from Streaming Memory), then in the following phase they can switch roles, where the IPU that before was computing now communicates with the host, and vice versa; as in the previous phase, the compute and communication can overlap. This feature also allow two IPUs to perform computation at the same time, in a data-parallel fashion.

### File structure

* `phased_execution.py` The main PopART program.
* `README.md` This file.
* `requirements.txt` Requirements needed to run this example.
* `test_phased_execution.py` Test script.
* `conftest.py` Pytest conftest file holds test fixtures.

### How to use this demo

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU System.
   Make sure to source the `enable.sh` scripts for poplar, gc_drivers and popart.

   The PopART Python API only supports Python 3. It is recommended to use a virtualenv.

2) Run the model.

       python3 phased_execution.py

3) (Optional) To run the tests

       python3 -m pytest

#### Options
The program has a few command line options.

`-h`                  Show usage information.

`--batch-size`        Sets the batch size.

`--dsize`             Size of all the square matrices involved in the computations.

`--num-layers`        Number of layers that constitute the model.

`--batches-per-step`  Number of mini-batches to perform on the device before returning to the host.

`--iters`             Number of iterations to run.

`--dtype`             Data type for the model. Can be `float16` (default) or `float32`.

`--profile`           Profile the execution and generate a report.

`--profile-dir`       Directory where to save the report files (ignored if `--profile` not set).

`--sharded-execution` Run the model by just sharding it over the two devices, without making use of the phased execution.


### Note on efficiency

With this mode of execution, when compute and I/O are overlapped, the time needed to complete each phase corresponds to the longest between the I/O time and the compute time.
Therefore it is desirable that for each phase these two parts take a similar time to complete. Heuristically, this means that each phase needs to account from 3750 to 7500 FLOPs of compute per byte loaded from Streaming Memory.
To see what this means concretely, consider the model in this demo, where each execution phase consists of one matrix multiplication. For readability of the calculations that follows, let's use `b` for `batch-size` and `d` for `dsize`. In each phase the inputs have dimensions: `bxDxD` and `DxD`, so the output has dimensions: `bxDxD`. Therefore each phase will perform `bxDxDx2xD = 2xbxD^3` FLOPs, and when loading the weights from Streaming Memory, supposing we're using `float16` data type, it will transfer `DxDx2` bytes.
Using the heuristic mentioned above, say we aim for about: FLOPs / bytes = 7500. In this case then: `(2xbxD^3) / (2xD^2) = bxD = 7500`. With `b = 32` we need `D ~= 235` to satisfy the required condition; let's take `D = 250` rounding up. In fact, these are the default values for these parameters, with which in each phase the compute and the I/O should take a similar time to complete, so to ensure that this mode of execution achieve a good utilisation of the resources.
