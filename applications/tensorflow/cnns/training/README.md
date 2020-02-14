# CNN Training on IPUs

This readme describes how to run CNN models such as ResNet for image recognition training on the IPU.

## Overview

Deep CNN residual learning models such as ResNet are used for image recognition and classification. The training 
examples given below use a ResNet-50 model implemented in TensorFlow, optimised for Graphcore's IPU.

## Graphcore ResNet-50 model

The ResNet model provided has been written to best utilize Graphcore's IPU processors. As the whole of the model is 
always in memory on the IPU, smaller batch sizes become more efficient than on other hardware. The IPU's built-in 
stochastic rounding support improves accuracy when using half-precision which allows greater throughput. The model
uses loss scaling to maintain accuracy at low precision, and has techniques such as cosine learning rates and label
smoothing to improve final verification accuracy. Both model and data parallelism can be used to scale training
efficiently over many IPUs.

## Quick start guide

1. Prepare the TensorFlow environment. Install the poplar-sdk following the README provided. Make sure to run the 
   `enable.sh` scripts and activate a Python virtualenv with gc_tensorflow installed.
2. Download the data. See below for details on obtaining the datasets. 
3. Run the training program. For example: 
   `python3 train.py --dataset imagenet --data-dir path-to/imagenet`

### Datasets

The dataset used for the ResNet-50 training is the ImageNet LSVRC 2012 dataset. This contains about 1.28 million images 
in 1000 classes. It can be downloaded from here http://image-net.org/download and is approximately 150GB for the
training and validation sets.

The CIFAR-10 dataset can be downloaded here https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz, and the 
CIFAR-100 dataset is here https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz.

## File structure

|            |                           |
|------------|---------------------------|
| `train.py`      | The main training program |
| `validation.py` | Contains the validation code used in `train.py` but also can be run to perform validation on previously generated checkpoints. The options should be set to be the same as those used for training, with the `--restore-path` option pointing to the log directory of the previously generated checkpoints |
| `restore.py`    | Used for restoring a training run from a previously saved checkpoint. For example: `python restore.py --restore-path logs/RN20_bs32_BN_16.16_v0.9.115_3S9/` |
| `ipu_utils.py`  | IPU specific utilities |
| `log.py`        | Module containing functions for logging results |
| `Datasets/`     | Code for using different datasets. Currently CIFAR-10, CIFAR-100 and ImageNet are supported |
| `Models/`       | Code for neural networks<br/>- `resnet.py`: A ResNet description based on code from Graphcore's Customer Engineering team and well optimised for the IPU.<br/>- `resnext.py`: Definition for ResNeXt model.<br/>- `squeezenet.py`: Definition for SqueezeNet model.
| `LR_Schedules/` | Different LR schedules<br/> - `stepped.py`: A stepped learning rate schedule with optional warmup<br/>- `cosine.py`: A cosine learning rate schedule with optional warmup
| `test/`         | Test files |


## Training examples

The default values for each of the supported data sets should give good results. See below for details on how best to
use the `--data-dir` and `--dataset` options.

If you set a `DATA_DIR` environment variable (`export DATA_DIR=path/to/data`) then that will be used be default if
`--data-dir` is omitted.

Note also that the `--synthetic-data` flag can be set which will transfer randomised instead of real data.


### ImageNet - ResNet-50

This is a larger network and may need to be split across two or more IPUs. The `--shards` option cuts the graph into
multiple parts that are run on separate IPUs.

    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 2 --shards 2

Validation top-1 accuracy 75.1% after 100 epochs

The model will fit on a single IPU with a batch size of 1. 

    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 1 --available-memory-proportion 0.1

There are a number of options that will help achieve a higher accuracy, many of which are detailed below. For example:

    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 2 --shards 4 --precision 16.32 \
    --lr-schedule cosine --epochs 120 --label-smoothing 0.1

Should achieve a top-1 validation accuracy of about 77.3%

Training over many IPUs can utilise both model and data parallelism. An example using 16 IPUs with four replicas of a 
four stage pipeline model is:

    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 4 --shards 4 \
    --pipeline-depth 64 --pipeline-splits b1/2/relu b2/3/relu b3/5/relu --xla-recompute --replicas 4 


# Training Options

`--model`: By default this is set to `resnet`, and this document is focused on ResNet 
training, but other examples such as `squeezenet` and `resnext` are available in the
`Models/` directory.

## ResNet model options

`--model-size` : The size of the model to use. Only certain sizes are supported depending on the model and input data 
size. For ImageNet data values of 18, 34 and 50 are typical, and for Cifar 14, 20 and 32. Check the code for the full 
ranges.

`--batch-norm` : Batch normalisation is recommended for medium and larger batch sizes that will typically be used 
training with Cifar data on the IPU. For ImageNet data smaller batch sizes are usually used and group normalisation is 
recommended.

`--group-norm` : Group normalisation can be used for small batch sizes (including 1) when batch normalisation would be 
unsuitable. Use the `--groups` option to specify the number of groups used (default 32).


## Major options

`--batch-size` : The batch size used for training. When training on IPUs this will typically be smaller than batch 
sizes used on GPUs. A batch size of four or less is often used, but in these cases using group normalisation is 
recommended instead of batch normalisation (see `--group-norm`).

`--base-learning-rate` : The base learning rate is scaled by the batch size to obtain the final learning rate. This 
means that a single `base-learning-rate` can be appropriate over a range of batch sizes.

`--epochs` \ `--iterations` : These options determine the length of training and only one should be specified.

`--data-dir` : The directory in which to search for the training and validation data. ImageNet must be in TFRecords 
format. CIFAR-10/100 must be in binary format. If you have set a `DATA_DIR` environment variable then this will be used 
if `--data-dir` is omitted.

`--dataset` : The dataset to use. Must be one of `imagenet`, `cifar-10` or `cifar-100`. This can often be inferred from 
the `--data-dir` option.

`--lr-schedule` : The learning rate schedule function to use. The default is `stepped` which is configured using 
`--learning-rate-decay` and `--learning-rate-schedule`. You can also select `cosine` for a cosine learning rate.

`--warmup-epochs` : Both the `stepped` and `cosine` learning rate schedules can have a warmup length which linearly 
increases the learning rate over the first few epochs (default 5). This can help prevent early divergences in the 
network when weights have been set randomly. To turn off warmup set the value to 0.


## IPU options

`--shards` : The number of IPUs to split the model over (default `1`). If `shards > 1` then the first part of the model 
will be run on one IPU with later parts run on other IPUs with data passed between them. This is essential if the model 
is too large to fit on a single IPU, but can also be used to increase the possible batch size. As an advanced usage,
the sharding algorithm can be influenced using the `--sharding-exclude-filter` and `--sharding-include-filter` options.
These specify sub-strings of edge names that can be used when looking for a cutting point in the graph. They will be 
ignored when using pipelining which uses `--pipeline-splits` instead. See also `--pipeline-depth`.

`--pipeline-depth` : When a model is run over multiple shards (see `--shards`) pipelining the data flow can improve
throughput by utilising more than one IPU at a time. At present the splitting points for the pipelined model must
be specified, with one less split than the number of shards used. Use `--pipeline-splits` to specifiy the splits - 
if omitted then the list of available splits will be output.

`--precision` : Specifies the data types to use for calculations. The default, `16.16` uses half-precision 
floating-point arithmetic throughout. This lowers the required memory which allows larger models to train, or larger 
batch sizes to be used. It is however more prone to numerical instability and will have a small but significant 
negative effect on final trained model accuracy. The `16.32` option uses half-precision for most operations but keeps 
master weights in full-precision - this should improve final accuracy at the cost of extra memory usage. The `32.32` 
option uses full-precision floating-point throughout.

`--no-stochastic-rounding` : By default stochastic rounding on the IPU is turned on. This gives extra precision and is 
especially important when using half-precision numbers. Using stochastic rounding on the IPU doesn't impact the 
performance of arithmetic operations and so it is generally recommended that it is left on.

`--batches-per-step` : The number of batches that are performed on the device before returning to the host. Setting 
this to 1 will make only a single batch be processed on the IPU(s) before returning data to the host. This can be 
useful when debugging but the extra data transfers will significantly slow training. The default value of 1000 is a 
reasonable compromise for most situations.

`--select-ipus` : The choice of which IPU to run the training and/or validation graphs on is usually automatic, based 
on availability. This option can be used to override that selection by choosing a training and optionally also 
validation IPU ID. The IPU configurations can be listed using the `gc-info` tool (type `gc-info -l` on the command 
line).

`--fp-exceptions` : Turns on floating point exceptions.

`--available-memory-proportion` : The approximate proportion of memory which is available for convolutions. If the
 value is 0 then just optimise for memory. Default is 0.6.


## Validation options

`--no-validation` : Turns off validation.

`--valid-batch-size` : The batch size to use for validation.

Note that the `validation.py` script can be run to validate previously generated checkpoints. Use the `--restore-path` 
option to point to the checkpoints and set up the model the same way as in training. See also the `--max-ckpts-to-keep` 
option when training.


## Other options

`--synthetic-data` : Uses a synthetic dataset filled with random data. If running with this option turned on is 
significantly faster than running with real data then the training speed is likely CPU bound.

`--gradients-to-accumulate` : The number of gradients to accumulate before doing a weight update. This allows the
effective mini-batch size to increase to sizes that would otherwise not fit into memory. Note that when using
`--pipeline-depth` this option cannot be used as gradients will instead be accumulated over the pipeline depth.

`--max-ckpts-to-keep` : The maximum number of checkpoints to keep when training. Defaults to 1, except when the 
validation mode is set to `end`.

`--replicas` : The number of replicas of the graph to use. Using `N` replicas increases the batch size by a factor of 
`N` (as well as the number of IPUs used for training)

`--optimiser` : Choice of optimiser. Default is `SGD` but `momentum` is also available, in which case you also should 
set the `--momentum` flag.

`--momentum` : Momentum coefficient to use when `--optimiser` is set to `momentum`. The default is `0.9`.

`--label-smoothing` : A label smoothing factor can help improve the model accuracy by reducing the polarity of the
labels in the final layer. A reasonable value might be 0.1 or 0.2. The default is 0, which implies no smoothing. 

`--weight-decay` : Value for weight decay bias. Setting to 0 removes weight decay.

`--loss-scaling` : When using mixed or half precision, loss scaling can be used to help preserve small gradient values. 
This scales the loss value calculated in the forward pass (and unscales it before the weight update) to reduce the 
chance of gradients becoming zero in half precision. The default value should be suitable in most cases.

`--seed` : Setting an integer as a seed will make training runs reproducible. At present this also limits the 
pre-processing pipeline to a single thread which has significant performance implications. 


# Resuming training runs

Training can be resumed from a checkpoint using the `restore.py` script. You must supply the `--restore-path` option 
with a valid checkpoint.


# Profiling

Logging information can be generated that can be used with the GC profiler tool at
https://github.com/graphcore/toolshed/

----

Use `--help` to show all available options.

