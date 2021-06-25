Graphcore
---

## Image classification inference on IPU using PyTorch

The following models are supported using this inference harness.

1. ResNet18-34-50-101-152
2. MobileNet
3. EfficientNet-B0..7
4. ResNext50-101

### File structure

* `run_benchmark.py` Driver script for running inference.
* `README.md` This file.
* `test_inference.py` Test cases for inference.
* `configs.yml` Contains the common inference configurations.

### Benchmarking

To reproduce the published Mk2 throughput benchmarks, please follow the setup instructions in this README, and then follow the instructions in [README_Benchmarks.md](README_Benchmarks.md) 

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
python3 run_benchmark.py --data real --replicas 1  --batch-size 1 --model resnet18 --device-iteration 1
```

### Options

The program has a few command-line options:

`-h`                            Show usage information.

`--config`                      Apply the selected configuration.

`--batch-size`                  Sets the batch size for inference.

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

`--full-precision-norm`         Calculate the norm layers in full precision.

`--enable-fast-groupnorm`       There are two implementations of the group norm layer. If the fast implementation enabled, it couldn't load checkpoints, which didn't train with this flag. The default implementation can use any checkpoint.

`--efficientnet-expand-ratio`   Expand ratio of the blocks in EfficientNet.

`--efficientnet-group-dim`      Group dimensionality of depthwise convolution in EfficientNet.

`--profile`                     Generate PopVision Graph Analyser report

`--eight-bit-io`                Image transfer from host to IPU in 8-bit format, requires normalisation on the IPU

`--normalization-location`      Location of the data normalization. Options: `host`, `ipu`, `none`

`--model-cache-path`            If path is given the compiled model is cached to the provided folder.

### Model configurations

Inference on single IPU.

|Model   | MK1 IPU | MK2 IPU |
|-------|----------|---------|
|ResNet50| `python3 run_benchmark.py --config resnet50-mk1` | `python3 run_benchmark.py --config resnet50-mk2`  |
|EfficientNet-B0| `python3 run_benchmark.py --config efficientnet-b0-mk1`  |`python3 run_benchmark.py --config efficientnet-b0-mk2`|
