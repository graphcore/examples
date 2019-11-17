# Graphcore

## ResNet Training

This directory contains the code to run ResNet training end-to-end.

### File structure

* `resnet_main.py` Main training and validation loop.
* `resnet_data.py` Data pipeline.
* `resnet_dataloader.py` A modified pytorch data loader
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

  Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
  scripts for poplar, gc_drivers (if running on hardware) and popart.

2) Download the data.

  For Cifar10/100, the data will automatically be downloaded by the program if not present at
  the location defined by the '--data-dir' option

  For ImageNet follow the instruction on how to download and prepare the data. 
  <https://github.com/pytorch/examples/tree/master/imagenet>

3) Install pytorch

  This example uses the pytorch to prepare and load the data. To install with pip execute:
  
  pip install torch torchvision

  Fully instructions can be found on https://pytorch.org

4) Install requirements

  pip install -r requirements.txt

  Note: You need torch version 1.1.0. The dataloader is not compatiable with 1.0.0

5) Run the training program. Use the `--data-dir` option to specify a path to
   the data.

    python resnet_main.py --data-dir=./ [--help]

6) To run run parallel data training on two IPUS

    python resnet_main.py --data-dir=./ --num-ipus=2 --batch-size=8

### Extra information

#### Options


Use `--help` to show the available options.

`--dataset` specifies which data set to use. Current options are `CIFAR-10` 
or `IMAGENET`. The default is `CIFAR-10` 

`--data-dir` specified the folder with the training and validation data. For 
cifar-10 this done automatically. For imagenet a folder with train and val 
folders

`--base-learning-rate` specifies the exponent of the base learning rate.
The learning rate is set by `lr = blr * batch-size`.
See <https://arxiv.org/abs/1804.07612> for more details.

`--batch-size` the batch size. Must be a multiple of num-ipus

`--num-ipus` the number of ipus to use. The graph will be replicated on each
ipu and executed in parallel. The number of samples processed on each ipu will be
(batch-size / num-ipus).

`--no-prng` disables stochastic rounding.

