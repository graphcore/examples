Image Classification Training Example using PyTorch
---

### File structure

* `train.py` Train the graph from scratch.
* `restore.py` Restore the training process from a given checkpoint file.
* `validate.py` Validate the given checkpoint(s).
* `README.md` This file.
* `lr_schedule.py` Collection of learning rate schedulers.
* `train_utils.py` Collection of functions which parse the training configuration.
* `test_train.py` Test cases for training.
* `weight_avg.py` Create a weight averaged model from checkpints.
* `configs.yml` Contains the common train configurations.

### Benchmarking

To reproduce the published Mk2 throughput benchmarks, please follow the setup instructions in this README, and then follow the instructions in [README_Benchmarks.md](README_Benchmarks.md) 

### How to use this demo

1) Install and activate the PopTorch environment as per cnns folder README.md, if you haven't done already.

2) Download the datasets:
    * ImageNet dataset (available at [http://www.image-net.org/](http://www.image-net.org/))
    * CIFAR10 dataset downloads automatically

The ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes, can be downloaded from [http://www.image-net.org/download](the ImageNet website). It is approximately 150GB for the training and validation sets. Please note you need to register and request permission to download this dataset on the Imagenet website. You cannot download the dataset until ImageNet confirms your registration and sends you a confirmation email. If you do not get the confirmation email within a couple of days, contact [support@imagenet.org](ImageNet support) to see why your registration has not been confirmed. Once your registration is confirmed, go to the download site. The dataset is available for non-commercial use only. Full terms and conditions and more information are available on the [http://www.image-net.org/download-faq](ImageNet download FAQ)

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

`--seed`                        Provide a seed for random number generation

`--batch-size`                  Sets the batch size for training

`--model`                       Select the model (from a list of supported models) for training

`--data`                        Choose the dataset between `cifar10`, `imagenet`, `generated` and `synthetic`. In synthetic data mode (only for benchmarking throughput) there is no host-device I/O and random data is generated on the device. In generated mode random data is created on host side.

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

`--full-precision-norm`         Calculate the norm layers in full precision.

`--no-validation`               Skip validation

`--disable-metrics`             Do not calculate metrics during training, useful to measure peak throughput

`--enable-recompute`            Enable the recomputation of network activations during backward pass instead of caching them during forward pass. This option turns on the recomputation for single-stage models. If the model is multi-stage (pipelined) the recomputation is always enabled.

`--recompute-checkpoints`       List of recomputation checkpoint rules: [conv:store convolution activations|norm: store normlayer activations]

`--offload-optimizer`           Store the optimizer status off-chip

`--lr-schedule`                 Select learning rate schedule from [`step`, `cosine`, `exponential`] options

`--lr-decay`                    Learning rate decay (required with step schedule). At the predefined epoch, the learning rate is multiplied with this number

`--lr-epoch-decay`              List of epochs, when learning rate is modified

`--warmup-epoch`                Number of learning rate warmup epochs

`--checkpoint-path`             The checkpoint folder. In the given folder a checkpoint is created after every epoch

`--optimizer`                   Define the optimizer: `sgd`, `adamw`, `rmsprop`

`--momentum`                    Momentum factor

`--loss-scaling`                Loss scaling factor. This value is reached by the end of the training.

`--initial-loss-scaling`        Initial loss scaling factor. The loss scaling interpolates between this and loss-scaling value. The loss scaling value multiplies by 2 during the training until the given loss scaling value is not reached. If not determined the `--loss-scaling` is used during the training. Example: 100 epoch, initial loss scaling 16, loss scaling 128: Epoch 1-25 ls=16;Epoch 26-50 ls=32;Epoch 51-75 ls=64;Epoch 76-100 ls=128

`--enable-stochastic-rounding`  Enable Stochastic Rounding

`--weight-decay`                Weight decay factor

`--wandb`                       Use Weights and Biases to log the training

`--logs-per-epoch`              Number of logging steps in each epoch

`--label-smoothing`             The label smoothing factor: 0.0 means no label smoothing (this is the default)

`--lr-scheduler-freq`           Number of learning rate update in each epoch. In case of 0 used, it is updated after every batch

`--weight-avg-strategy`         Weight average strategy

`--weight-avg-exp-decay`        The exponential decay constant for weight averaging. Applied if exponential weight average strategy is chosen

`--weight-avg-N`                Weight average applied on last N checkpoint, -1 means all checkpoints

`--rmsprop-decay`               RMSprop smoothing constant

`--efficientnet-expand-ratio`   Expand ratio of the blocks in EfficientNet

`--efficientnet-group-dim`      Group dimensionality of depthwise convolution in EfficientNet

`--profile`                     Generate PopVision Graph Analyser report

`--loss-velocity-scaling-ratio` Only for SGD optimizer: Loss Velocity / Velocity scaling ratio. In case of large number of replicas >1.0 can increase numerical stability

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

### How to use poprun to make train distributed

```
poprun --num-instances=<number of instances> --numa-aware=yes --num-replicas=<Total number of repicas> ...
```
