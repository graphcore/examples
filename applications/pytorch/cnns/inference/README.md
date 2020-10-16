Graphcore

## Image classification inference on IPU using PyTorch

The following models are supported using this inference harness.

1. Resnet18
2. Resnet34
3. ResNet50
4. ResNet101
5. ResNet152
6. MobileNet
7. EfficientNet-B0..7
8. ResNext50
9. ResNext101


### File structure

* `run_benchmark.py` Driver script for running inference.
* `get_images.sh` Script to fetch sample test images.
* `README.md` This file.
* `data.py` Creates the DataLoader to load images.
* `test_inference.py` Test cases for inference.

### How to use this demo

1) Install and activate the PopTorch environment as per cnns folder README.md, if you haven't done already. 
   
2) Download the images.

       ./get_images.sh

  This will create and populate the `images/` directory with sample test images.

3) Run the graph.
```
       python3 run_benchmark.py --data real --replicas 1  --batch-size 1 --model resnet18 --device-iteration 1
```

#### Options
The program has a few command-line options:

`-h`                            Show usage information.

`--batch-size`                  Sets the batch size for inference.

`--model`                       Select the model (from a list of supported models) for inference.

`--data`                        Choose the data mode between real and synthetic. In synthetic data mode (only for benchmarking throughput) there is no host-device I/O and random data is generated on the device.

`--pipeline-splits`             List of layers to create stages of the pipeline. Each stage runs on different IPUs. Example: layer0 layer1/conv layer2/block3/bn

`--replicas`                    Number of IPU replicas.

`--device-iteration`            Sets the device iteration: the number of inference steps before program control is returned to the host.

`--precision`                   Sets the floating point precision: full or half.

`--iterations`                  Sets the number of program iterations to execute.

`--half-partial`                Flag for accumulating matrix multiplication partials in half precision

`--normlayer`                   Select the used normlayer from the following list: 'batch', 'group', 'none'

`--groupnorm-group-num`          If group normalization is used, the number of groups can be set here.

`--available-memory-proportion` Proportion of memory which is available for convolutions.


#### ResNet
The code supports the following models
##### ResNet18

Inference with synthetic data:

```
python3 run_benchmark.py --data synthetic --replicas 1 --batch-size 16 --model resnet18 --device-iteration 1000 --normlayer none --precision half
```

Inference with real data:

```
python3 run_benchmark.py --data real --replicas 1 --batch-size 24 --model resnet18 --device-iteration 64 --normlayer batch --precision half
```

##### ResNet34

Inference with real data:

```
python3 run_benchmark.py --data real --replicas 1 --batch-size 19 --model resnet34 --device-iteration 64 --normlayer batch --precision half
```

##### ResNet50

Inference with real data:

```
python3 run_benchmark.py --data real --replicas 1 --batch-size 8 --model resnet50 --device-iteration 64 --normlayer batch --precision half
```

##### EfficientNet-B0

Inference with real data:

```
python3 run_benchmark.py --data real --replicas 1 --batch-size 11 --model efficientnet-b0 --device-iteration 1000 --normlayer batch --precision half
```
