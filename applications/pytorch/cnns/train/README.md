# Image classification train on IPU using PyTorch

This README describes how to run CNN models for image recognition training on the IPU.


## File structure

* `train.py` Train the graph from scretch
* `restore.py` Restore the training process from a given checkpoint file. 
* `validate.py` Validate the given checkpoint(s)
* `README.md` This file.
* `data.py` Creates the DataLoader to load images.
* `lr_schedule.py` Collection of learning rate schedulers.
* `dataloader.py` Provides high performance dataloaders.
* `utils.py` Collection of functions which are not closely related to the training.
* `test_train.py` Test cases for training

## How to use this demo

1) Install and activate the PopTorch environment as per cnns folder README.md, if you haven't done already. 
   
2) Download the raw imagenet dataset (available at http://www.image-net.org/). CIFAR10 dataset downloads automatically.

3) Run the training.
```
       python3 train.py --data imagenet --imagenet-data-path path-to/imagenet
```


## Training examples
### ImageNet - ResNet-50

Training over many IPUs can utilise both model and data parallelism. An example using 16 IPUs with four replicas of a
four stage pipeline model is:
```
python3 train.py --model resnet50  --data cifar10 --normlayer group --groupnorm-group-num 32 --replicas 4 --batch-size 4 --device-iteration 2 --pipeline-splits layer1/2 layer2/3 layer3/3 --enable-pipeline-recompute --no-validation --precision half --gradient-accumulation 64
```

### ImageNet - EfficientNet-B0
```
python3 train.py --model efficientnet-b0 --precision half --pipeline-splits _blocks/1/_project_conv _blocks/4/_bn1 _blocks/10 --device-iteration 1 --available-memory-proportion 0.1 --enable-pipeline-recompute --no-validation --batch-size 4 --gradient-accumulation 128 --epoch 10 --replicas 4 
```


## Options
The program has a few command-line options:

`-h`                            Show usage information.

`--batch-size`                  Sets the batch size for training.

`--model`                       Select the model (from a list of supported models) for training.

`--data`                        Choose the dataset between CIFAR10 and imagenet and synthetic. In synthetic data mode (only for benchmarking throughput) there is no host-device I/O and random data is generated on the device.

`--imagenet-data-path`          The path of the downloaded imagenet dataset (only required, if imagenet is selected as data)

`--pipeline-splits`             List of layers to create stages of the pipeline. Each stage runs on different IPUs. Example: layer0 layer1/conv layer2/block3/bn

`--replicas`                    Number of IPU replicas.

`--device-iteration`            Sets the device iteration: the number of inference steps before program control is returned to the host.

`--precision`                   Sets the floating point precision: full or half.

`--half-partial`                Flag for accumulating matrix multiplication partials in half precision

`--available-memory-proportion` Proportion of memory which is available for convolutions.

`--gradient-accumulation`       Number of batches to accumulate before a gradient update

`--lr`                          Initial learning rate 

`--epoch`                       Number of training epochs

`--normlayer`                   Select the used normlayer from the following list: 'batch', 'group', 'none'

`--groupnorm-group-num`         If group normalization is used, the number of groups can be set here.

`--no-validation`               Skip validation

`--disable-metrics`             Do not calculate metrics during training, useful to measure peak throughput

`--enable-pipeline-recompute`   Enable the recomputation of network activations during backward pass instead of caching them during forward pass

`--lr-schedule`                 Select learning rate schedule from [no, step] options

`--lr-decay`                    Learning rate decay (required with step schedule). At the predefined epoch, the learning rate is multiplied with this number

`--lr-epoch-decay`              List of epochs, when learning rate is modified.

`--warmup-epoch`                Number of learning rate warmup epochs

`--checkpoint-path`             The checkpoint folder. In the given folder a checkpoint is created after every epoch

`--optimizer`                   Define the optimizer

`--momentum`                    Momentum factor

`--loss-scaling`                Loss scaling factor

`--enable-stochastic-rounding`  Enable Stochastic Rounding

### How to use the checkpoints
A given checkpoint file can be used to restore the training and continue from there, with the following command:
```
       python3 restore.py --checkpoint-path <File path>
```

Validation is also possible with the following command:
```
       python3 validate.py --checkpoint-path <path>
```
If the provided path is a file: The validation accuracy is calculated for the given file.
If the path is a folder, then the validation accuracy is calculated for every single checkpoint in the folder.
