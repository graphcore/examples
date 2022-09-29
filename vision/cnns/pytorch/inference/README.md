Graphcore
---

## Image classification inference on IPU using PyTorch

The following models are supported using this inference harness.

1. ResNet18-34-50-101-152
2. MobileNet
3. EfficientNet-B0, B4
4. ResNext50-101

### File structure

* `run_benchmark.py` Driver script for running inference.
* `run_benchmark_with_triton_server.py` Driver script for running inference with Triton Server.
* `README.md` This file.
* `configs.yml` Contains the common inference configurations.

### How to use this demo

1) Install and activate the PopTorch environment as per cnns folder README.md, if you haven't done already.

2) Download the images using `get_images.sh`. The download script should be run from within the data directory:
    
```console
cd ../datasets/
./get_images.sh
cd ../inference/
```

  This will create and populate the `images/` directory in the `data` folder with sample test images.

3) Run the graph.

```console
python3 run_benchmark.py --data real --replicas 1  --micro-batch-size 1 --model resnet18 --device-iteration 1
```

  Run benchmarks with Triton Server:

```console
python3 run_benchmark_with_triton_server.py --benchmark_only=true ../tests_serial/tritonserver/
```

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

### Options

The `run_benchmark.py` program has a few command-line options:

`-h`                            Show usage information.

`--config`                      Apply the selected configuration.

`--micro-batch-size`            Sets the micro batch size for inference.

`--model`                       Select the model (from a list of supported models) for inference.

`--data`                        Choose the data mode between `real`, `synthetic` and `generated`. In synthetic data mode (only for benchmarking throughput) there is no host-device I/O and random data is created on the device.

`--pipeline-splits`             List of layers to create stages of the pipeline. Each stage runs on different IPUs. Example: `layer0 layer1/conv layer2/block3/bn` The stages are space separated.

`--replicas`                    Number of IPU replicas.

`--device-iterations`           Sets the device iteration: the number of inference steps before program control is returned to the host.

`--precision`                   Precision of Ops(weights/activations/gradients) and Master data types: `32.32` or `16.16`.

`--iterations`                  Sets the number of program iterations to execute.

`--half-partial`                Flag for accumulating matrix multiplication partials in half precision.

`--norm-type`                   Select the used normlayer from the following list: `batch`, `group`, `none`.

`--norm-num-groups`             If group normalization is used, the number of groups can be set here.

`--enable-fast-groupnorm`       There are two implementations of the group norm layer. If the fast implementation enabled, it couldn't load checkpoints, which didn't train with this flag. The default implementation can use any checkpoint.

`--efficientnet-expand-ratio`   Expand ratio of the blocks in EfficientNet.

`--efficientnet-group-dim`      Group dimensionality of depthwise convolution in EfficientNet.

`--profile`                     Generate PopVision Graph Analyser report

`--eight-bit-io`                Image transfer from host to IPU in 8-bit format, requires normalisation on the IPU

`--normalization-location`      Location of the data normalization. Options: `host`, `ipu`, `none`

`--model-cache-path`            If path is given the compiled model is cached to the provided folder.

`--random-weights`              When true, weights of the model are initialized randomly.

The `run_benchmark_with_triton_server.py` is a thin wrapper that allows to run `pytest` tests from python without explicit library request (`python -m pytest ...`). Script is accepting all `pytest` options with additional command-line options:

`--model-repository`            Path to directory which consists model configuration files for Triton Server and model exported to popef file. Default: `../test_serial/tritonserver/models`

`--grpc-port`                   Port on which Triton Server will be listening for commands.

`--benchmark_only`              Run benchmarks sending requests with exactly 1 batch size of selected model.

### Model configurations

Inference on a single IPU.

|Model|Command|
|-----|------|
|ResNet50|`python3 run_benchmark.py --config resnet50`|
|EfficientNet-B0|`python3 run_benchmark.py --config efficientnet-b0`|
|EfficientNet-B4|`python3 run_benchmark.py --config efficientnet-b4`|
|EfficientNet-B0 (Group Norm, Group Conv)|`python3 run_benchmark.py --config efficientnet-b0-g16-gn`|
|EfficientNet-B4 (Group Norm, Group Conv)|`python3 run_benchmark.py --config efficientnet-b4-g16-gn`|
|MobileNet v3 small|`python3 run_benchmark.py --config mobilenet-v3-small`|
|MobileNet v3 large|`python3 run_benchmark.py --config mobilenet-v3-large`|

Inference with Triton Server, requires at least POD4.

|Model|Command|
|-----|------|
|ResNet50|`python3 run_benchmark_with_triton_server.py -s -k test_single_model[resnet50-resnet50 --benchmark_only=true ../tests_serial/tritonserver/`|
|EfficientNet-B0|`python3 run_benchmark_with_triton_server.py -s -k test_single_model[efficientnet-b0-efficientnet-b0 --benchmark_only=true ../tests_serial/tritonserver/`|
|EfficientNet-B4|`python3 run_benchmark_with_triton_server.py -s -k test_single_model[efficientnet-b4-efficientnet-b4 --benchmark_only=true ../tests_serial/tritonserver/`|
