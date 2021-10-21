Image Classification Training Example using PyTorch
---

### File structure

* `train.py` Train the graph from scratch.
* `restore.py` Restore the training process from a given checkpoint file.
* `validate.py` Validate the given checkpoint(s).
* `README.md` This file.
* `lr_schedule.py` Collection of learning rate schedulers.
* `train_utils.py` Collection of functions which parse the training configuration.
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

```console
       python3 train.py --data imagenet --imagenet-data-path <path-to/imagenet>
```

#### Supported ImageNet formats:
* Raw ImageNet: without any modifications
* WebDataset(recommended): you can use the given script to generate it (see datasets folder).
* TFRecord format, to generate see the TensorFlow applications


### Training examples

NOTE: It is suggested to generate the webdataset format of the ImageNet dataset to avoid host side bottlenecks.

#### ImageNet

|IPU configuration|Model  | Config name| Note |
|-------|----------|---------|---------|
|Mk2 IPU-POD16|ResNet50| `resnet50_mk2`| single IPU, 16 replicas |
|Mk2 IPU-POD64|ResNet50| `resnet50_mk2_pod64`| single IPU, 64 replicas |
|Mk2 IPU-POD16|EfficientNet-B0 (Group Norm, Group Conv)| `efficientnet-b0-g16-gn-16ipu-mk2`| 2 IPUs, 8 replicas |
|Mk2 IPU-POD16|EfficientNet-B4 (Group Norm, Group Conv)| `efficientnet-b4-g16-gn-16ipu-mk2`| 4 IPUs, 4 replicas |
|Mk2 IPU-POD4|MobileNet v3 small | `mobilenet-v3-small-pod4`| single IPU, 4 replicas |
|Mk2 IPU-POD16|MobileNet v3 large | `mobilenet-v3-large-pod16`| single IPU, 16 replicas |


```console
python3 train.py --config <config name> --imagenet-data-path <path-to/imagenet>
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

`--norm-eps`                    Set normalization layers epsilon

`--norm-num-groups`             If group normalization is used, the number of groups can be set here

`--enable-fast-groupnorm`       There are two implementations of the group norm layer. If the fast implementation enabled, it couldn't load checkpoints, which didn't train with this flag. The default implementation can use any checkpoint.

`--disable-stable-batchnorm`    There are two implementations of the batch norm layer. The default version is numerically more stable. The less stable is faster.

`--validation-mode`             The model validation mode. Possible values are `none` (no validation) `during` (validate after every n epochs) and `after` (validate after the training).

`--validation-frequency`        How many training epochs to run between validation steps.

`--disable-metrics`             Do not calculate metrics during training, useful to measure peak throughput

`--enable-recompute`            Enable the recomputation of network activations during backward pass instead of caching them during forward pass. This option turns on the recomputation for single-stage models. If the model is multi-stage (pipelined) the recomputation is always enabled.

`--recompute-checkpoints`       List of recomputation checkpoints. List of regex rules for the layer names. (Example: Select convolutional layers: `.*conv.*`)

`--offload-optimizer`           Store the optimizer status off-chip

`--lr-schedule`                 Select learning rate schedule from [`step`, `cosine`, `exponential`] options

`--lr-decay`                    Learning rate decay (required with step schedule). At the predefined epoch, the learning rate is multiplied with this number

`--lr-epoch-decay`              List of epochs, when learning rate is modified

`--warmup-epoch`                Number of learning rate warmup epochs

`--checkpoint-path`             The checkpoint folder. In the given folder a checkpoint is created after every epoch

`--optimizer`                   Define the optimizer: `sgd`, `sgd_combined`, `adamw`, `rmsprop`, `rmsprop_tf`

`--momentum`                    Momentum factor

`--optimizer-eps`               Small constant added to the updater term denominator for numerical stability.

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

`--loss-velocity-scaling-ratio` Only for sgd_combined optimizer: Loss Velocity / Velocity scaling ratio. In case of large number of replicas >1.0 can increase numerical stability

`--use-bbox-info`               Images may contain bounding box information for the target object. If this flag is set, during the augmentation process make sure the augmented image overlaps with the target object.

`--eight-bit-io`                Image transfer from host to IPU in 8-bit format, requires normalisation on the IPU

`--normalization-location`      Location of the data normalization. Options: `host`, `ipu`, `none`

`--dataloader-rebatch-size`     Batch size of the dataloader worker. The final batch is created from the smaller batches. Lower value results in less host-side memory. A higher value can reduce the overhead of rebatching. This setting can be useful for reducing memory pressure with a large global batch size.

`--iterations`                  Number of program iterations for generated and synthetic data. This helps to modify the length of these datasets.

`--model-cache-path`            If path is given the compiled model is cached to the provided folder.

`--mixup-alpha`                 The first shape parameter of the beta distribution used to sample mixup coefficients. The second shape parameter is the same as the first one. Value of 0.0 means mixup is disabled.

`--cutmix-lambda-low`           Lower bound for the cutmix lambda coefficient (lambda is sampled uniformly from [low, high)). If both bounds are set to 0.0 or 1.0, cutmix is disabled. If both bounds are equal, lambda always equals that value.

`--cutmix-lambda-high`          Higher bound for the cutmix lambda coefficient (lambda is sampled uniformly from [low, high)). If both bounds are set to 0.0 or 1.0, cutmix is disabled. If both bounds are equal, lambda always equals that value.

`--cutmix-disable-prob`         Probability that cutmix is disabled for a particular batch.

`--webdataset-memory-cache-ratio` Determines the portion of the webdataset, which is cached in memory.

`--input-image-padding`        Pad input images to be 4 channel images. This could speed up the model.

`--exchange-memory-target`     Exchange memory optimisation target: balanced/cycles/memory. In case of cycles it uses more memory, but runs faster.


### How to use the checkpoints

A given checkpoint file can be used to restore the training and continue from there, with the following command:

```console
       python3 restore.py --checkpoint-path <File path>
```

Validation is also possible with the following command:

```console
       python3 validate.py --checkpoint-path <path>
```

Weight average is available with the following command:

```console
       python3 weight_avg.py --checkpoint-path <path>
```

If the provided path is a file: The validation accuracy is calculated for the given file.
If the path is a folder, then the validation accuracy is calculated for every single checkpoint in the folder.

### How to use poprun to make training distributed

```
poprun --num-instances=<number of instances> --numa-aware=yes --num-replicas=<Total number of repicas> ...
```

### Reference distributed settings

These settings use poprun, to provide the maximal throughput.
The following scripts support the previously defined arguments too.

ResNet50 IPU-POD16 reference

```
./rn50_pod16.sh --checkpoint-path <path> --imagenet-data-path <path-to/imagenet> 
```

ResNet50 IPU-POD64 reference

```
./rn50_pod64.sh --checkpoint-path <path> --imagenet-data-path <path-to/imagenet> 
```

In case the script is unable to select the right partition or VIPU server, you can set them by running the following lines.

```
export VIPU_SERVER=<vipu server IP address>
export PARTITION=<name of the selected partition>
```
