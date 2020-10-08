# Graphcore

## ResNet and ResNeXt Training

This directory contains the code to run ResNet and ResNeXt training end-to-end.

You can train ResNets of all sizes, or a ResNeXt with 50 layers.
To train a ResNext50 as opposed to a ResNet 50, pass in `x50` to the `--size` argument. 

### File structure

* `resnet_main.py` Main training and validation loop.
* `resnet_data.py` Data pipeline.
* `resnet_dataloader.py` A modified pytorch data loader
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for poplar and popart.

2) Download the data.

  For Cifar10/100, the data will automatically be downloaded by the program if not present at
  the location defined by the '--data-dir' option

  For ImageNet follow the instruction on how to download and prepare the data. 
  <https://github.com/pytorch/examples/tree/master/imagenet>

3) Setup a virtual environment

  virtualenv venv -p python3.6

  source venv/bin/activate

4) Install requirements

  pip install -r requirements.txt

5) Run the training program. Use the `--data-dir` option to specify a path to
   the data.

    python resnet_main.py --data-dir=./ [--help]


### Extra information

#### Options


Use `--help` to show the available options. Here are a few common options:

`--size` (string representation of integer) specifies the size of the Resnet model. Prefix with `x` to instead a
 ResNeXt model of that size, where one exists, e.g. `x50`.

`--dataset` specifies which data set to use. Current options are `CIFAR-10` 
or `IMAGENET`. The default is `CIFAR-10` 

`--data-dir` specified the folder with the training and validation data. For 
cifar-10 this done automatically. For imagenet a folder with train and val 
folders

`--base-learning-rate` specifies the exponent of the base learning rate.
The learning rate is set by `lr = blr * batch-size`.
See <https://arxiv.org/abs/1804.07612> for more details.

`--batch-size` the batch size. Must be a multiple of num-ipus

`--replication-factor` specifies the number of replicas to execute in parallel. The number of samples processed on each ipu will be
(batch-size / replication-factor).  Note that the number of ipus must be a multiple of the replication factor.

`--gradient-accumulation-factor` specifies the number of gradients to accumulate before doing a weight update.

`--recompute` will recompute activations rather than storing them. This allows you to work with much bigger effective batch sizes, where the effective batch size is the product of the batch size and the gradient accumulation factor, to increase throughput.

`--num-ipus` the number of ipus to use. 

`--no-prng` disables stochastic rounding.

`--pipeline` runs the model over multiple IPUs using a pipelining technique for efficiency.
