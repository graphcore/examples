Graphcore
---

## Image classification train on IPU using PyTorch

This README describes how to run CNN models for image recognition training on the IPU.

### File structure

* `train.py` Train the graph from scratch.
* `restore.py` Restore the training process from a given checkpoint file.
* `validate.py` Validate the given checkpoint(s).
* `README.md` This file.
* `lr_schedule.py` Collection of learning rate schedulers.
* `train_utils.py` Collection of functions which are not closely related to the training.
* `test_train.py` Test cases for training.
* `weight_avg.py` Create a weight averaged model from checkpints.
* `configs.yml` Contains the common train configurations.

### How to use this demo

1) Install and activate the PopTorch environment as per cnns folder README.md, if you haven't done already.

2) Download the datasets:
    * Raw ImageNet dataset (available at [http://www.image-net.org/](http://www.image-net.org/))
    * CIFAR10 dataset downloads automatically

3) Run the training:

```bash
       python3 train.py --data imagenet --imagenet-data-path <path-to/imagenet>
```

### Training examples

The following training configurations using 16 IPUs.

#### ImageNet - ResNet-50

MK1 IPUs:

```bash
python3 train.py --config resnet50-16ipu-mk1 --imagenet-data-path <path-to/imagenet>
```

MK2 IPUs:

```bash
python3 train.py --config resnet50-16ipu-mk2 --imagenet-data-path <path-to/imagenet>
```

#### ImageNet - EfficientNet-B0

MK1 IPUs:

```bash
python3 train.py --config efficientnet-b0-16ipu-mk1 --imagenet-data-path <path-to/imagenet>
```

MK2 IPUs:

```bash
python3 train.py --config efficientnet-b0-16ipu-mk2 --imagenet-data-path <path-to/imagenet>
```

## Options

The program has a few command line options:

`-h`                            Show usage information

`--config`                      Apply the selected configuration

`--batch-size`                  Sets the batch size for training

`--model`                       Select the model (from a list of supported models) for training

`--data`                        Choose the dataset between CIFAR10 and imagenet and synthetic. In synthetic data mode (only for benchmarking throughput) there is no host-device I/O and random data is generated on the device.

`--imagenet-data-path`          The path of the downloaded ImageNet dataset (only required if imagenet is selected as data)

`--pipeline-splits`             List of layers to create stages of the pipeline. Each stage runs on different IPUs. Example: layer0 layer1/conv layer2/block3/bn

`--replicas`                    Number of IPU replicas

`--device-iterations`           Sets the device iteration: the number of inference steps before program control is returned to the host

`--precision`                   Precision of Ops(weights/activations/gradients) and Master data types: `16.16`, `32.32` or `16.32`

`--half-partial`                Flag for accumulating matrix multiplication partials in half precision

`--available-memory-proportion` Proportion of memory which is available for convolutions

`--gradient-accumulation`       Number of batches to accumulate before a gradient update

`--lr`                          Initial learning rate

`--epoch`                       Number of training epochs

`--norm-type`                   Select the used normlayer from the following list: `batch`, `group`, `none`

`--norm-num-groups`             If group normalization is used, the number of groups can be set here

`--no-validation`               Skip validation

`--disable-metrics`             Do not calculate metrics during training, useful to measure peak throughput

`--enable-pipeline-recompute`   Enable the recomputation of network activations during backward pass instead of caching them during forward pass

`--lr-schedule`                 Select learning rate schedule from [`step`, `cosine`, `exponential`] options

`--lr-decay`                    Learning rate decay (required with step schedule). At the predefined epoch, the learning rate is multiplied with this number

`--lr-epoch-decay`              List of epochs, when learning rate is modified

`--warmup-epoch`                Number of learning rate warmup epochs

`--checkpoint-path`             The checkpoint folder. In the given folder a checkpoint is created after every epoch

`--optimizer`                   Define the optimizer: `sgd`, `adamw`, `rmsprop`

`--momentum`                    Momentum factor

`--loss-scaling`                Loss scaling factor

`--enable-stochastic-rounding`  Enable Stochastic Rounding

`--weight-decay`                Weight decay factor

`--wandb`                       Use Weights and Biases to log the training

`--logs-per-epoch`              Number of logging steps in each epoch

`--label-smoothing`             The label smoothing factor: 0.0 means no label smoothing (this is the default)

`--lr-scheduler-freq`           Number of learning rate update in each epoch. In case of 0 used, it is updated after every batch

`--reduction`                   Applied reduction for loss and gradients: `sum` or `mean`

`--weight-avg-strategy`         Weight average strategy

`--weight-avg-exp-decay`        The exponential decay constant for weight averaging. Applied if exponential weight average strategy is chosen

`--rmsprop-decay`               RMSprop smoothing constant

### How to use the checkpoints

A given checkpoint file can be used to restore the training and continue from there, with the following command:

```bash
       python3 restore.py --checkpoint-path <File path>
```

Validation is also possible with the following command:

```bash
       python3 validate.py --checkpoint-path <path>
```

Weight average is available with the following command:

```bash
       python3 weight_avg.py --checkpoint-path <path>
```

If the provided path is a file: The validation accuracy is calculated for the given file.
If the path is a folder, then the validation accuracy is calculated for every single checkpoint in the folder.
