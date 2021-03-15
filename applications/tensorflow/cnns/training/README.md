# CNN Training on IPUs

This readme describes how to run CNN models such as ResNet and EfficientNet for image recognition training on the IPU.

## Overview

Deep CNN residual learning models such as ResNet and EfficientNet are used for image recognition and classification.
The training examples given below use models implemented in TensorFlow, optimised for Graphcore's IPU.

## Benchmarking

To reproduce the published Mk2 throughput benchmarks, please follow the setup instructions in this README, and then follow the instructions in [README_Benchmarks.md](README_Benchmarks.md) 

## Graphcore ResNet-50 and EfficientNet models

The models provided have been written to best utilize Graphcore's IPU processors. As the whole of the model is
always in memory on the IPU, smaller micro batch sizes become more efficient than on other hardware. The micro batch
size is the number of samples processed in one full forward/backward pass of the algorithm. This contrasts with
the global batch size which is the total number of samples processed in a weight update, and is defined as the product
of micro batch size, number of accumulated gradients and total number of replicas. The IPU's built-in
stochastic rounding support improves accuracy when using half-precision which allows greater throughput. The model
uses loss scaling to maintain accuracy at low precision, and has techniques such as cosine learning rates and label
smoothing to improve final verification accuracy. Both model and data parallelism can be used to scale training
efficiently over many IPUs.

ResNeXt and SqueezeNet models are also available.

## Quick start guide

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to run the `enable.sh` script for Poplar and activate a Python 3 virtualenv with
   the TensorFlow 1 wheel from the Poplar SDK installed.
2. Download the data. See below for details on obtaining the datasets.
3. Install the packages required by this application using (`pip install -r requirements.txt`)
4. Run the training script. For example:
   `python3 train.py --dataset imagenet --data-dir path-to/imagenet`

### Datasets

You can download the ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes,
from http://image-net.org/download. It is approximately 150GB for the training and validation sets.

The CIFAR-10 dataset is available here https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz, and the
CIFAR-100 dataset is available here https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz.

## File structure

| File / Subdirectory | Description               |
|---------------------|---------------------------|
| `train.py`          | The main training program |
| `validation.py`     | Contains the validation code used in `train.py` but also can be run to perform validation on     previously generated checkpoints. The options should be set to be the same as those used for training, with the `--restore-path` option pointing to the log directory of the previously generated checkpoints |
| `restore.py`        | Used for restoring a training run from a previously saved checkpoint. For example: `python restore.py --restore-path logs/RN20_bs32_BN_16.16_v0.9.115_3S9/` |
| `ipu_optimizer.py`  | Custom optimizer |
| `ipu_utils.py`      | IPU specific utilities |
| `log.py`            | Module containing functions for logging results |
| `Datasets/`         | Code for using different datasets. Currently CIFAR-10, CIFAR-100 and ImageNet are supported |
| `Models/`           | Code for neural networks<br/>- `resnet.py`: Definition for ResNet model.<br/>- `resnext.py`: Definition for ResNeXt model.<br/>- `squeezenet.py`: Definition for SqueezeNet model.<br/>- `efficientnet.py`: Definition for EfficientNet models.
| `LR_Schedules/`     | Different LR schedules<br/> - `stepped.py`: A stepped learning rate schedule with optional warmup<br/>- `cosine.py`: A cosine learning rate schedule with optional warmup <br/>- `exponential.py`: An exponential learning rate schedule with optional warmup <br/>- `polynomial_decay_lr.py`: A polynomial learning rate schedule with optional warmup
| `requirements.txt`  | Required packages for the tests |
| `weight_avg.py`     | Code for performing weight averaging of multiple checkpoints. |
| `configurations.py` | Code for parsing configuration files. |
| `configs.yml`       | File where configurations are defined. |
| `test/`             | Test files - run using `python3 -m pytest` after installing the required packages. |


## Training examples

The default values for each of the supported data sets should give good results. See below for details on how best to
use the `--data-dir` and `--dataset` options.

If you set a `DATA_DIR` environment variable (`export DATA_DIR=path/to/data`) then that will be used by default if
`--data-dir` is omitted.

Note also that the `--generated-data` option can be set which will transfer randomised instead of real data.


### ImageNet - ResNet-50

In this application you can find support for the ResNet family of neural networks of a number of model sizes, including
ResNet50.

This model fits on a single MK2 IPU with a batch size of up to 8:

    python train.py --model resnet --model-size 50 --dataset imagenet --data-dir .../imagenet-data/ --batch-size 8

Training over many IPUs can use both model and data parallelism. An example using 8 IPUs with four replicas of a
two stage pipeline model is:

    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 2 --shards 2 \
    --pipeline --gradient-accumulation-count 128 --pipeline-splits b3/0/relu --pipeline-schedule Grouped \
    --available-memory-proportion 0.4 --xla-recompute --replicas 4

Note that some combinations of options might not work. For example, it will not be possible to add replication for
a model that is a tight fit on a single IPU because replication introduces additional control code and buffers for
copying updated gradients between replicas.

There are a number of options that will help achieve a higher accuracy, many of which are detailed below. For example,
the following configuration trains ResNet50 using 16 Mk2 IPUs. Each IPU runs a single data-parallel replica of the
model with a micro-batch size of 8. We use a gradient accumulation count of 8, and 16 replicas in total for a
global batch size of 1024 (8 * 8 * 16). Activations for the forwards pass are re-calculated during the backwards pass. Partials
saved to tile memory within the convolutions are set to half-precision to maximise throughput on the IPU.
This example uses the SGD-M optimizer with group normalisation, cosine learning rate and label smoothing to train to >75.90%
validation accuracy in 65 epochs.

    python train.py --model resnet --model-size 50 --dataset imagenet --data-dir .../imagenet-data/ \
    --replicas 16 --batch-size 8 --gradient-accumulation-count 8 --epochs 65 \
    --xla-recompute --optimiser momentum --momentum 0.90 --ckpts-per-epoch 1 \
    --internal-exchange-optimisation-target balanced --normalise-input --stable-norm \
    --enable-half-partials --lr-schedule cosine --label-smoothing 0.1 --eight-bit-io


On MK1 IPUs, the model will fit on a single IPU with a batch size of 1.

    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 1 --available-memory-proportion 0.1

The following configuration achieves >75.9% accuracy for 16 MK1 IPUs. The MK1 IPU has less on-chip memory so the ResNet50 model
is pipelined over 4 IPUs and the available memory proportion option is set to adjust how much memory is used for convolutions
and matrix multiplications. Each IPU executes its corresponding pipeline stage with a micro-batch size of 4. We use a gradient accumulation
count of 64, and 4 replicas in total for a global batch size of 1024 (4 * 64 * 4). Activations for the forwards pass are re-calculated
during the backwards pass. Partials saved to tile memory within the convolutions are set to half-precision to maximise throughput on
the IPU. This example uses the SGD-M optimizer with group normalisation, cosine learning rate and label smoothing to train to >75.90%
validation accuracy in 65 epochs.

    python train.py --model resnet --model-size 50 --dataset imagenet --data-dir .../imagenet-data \
    --shards 4 --replicas 4 --batch-size 4 --gradient-accumulation-count 64 --epochs 65 \
    --pipeline --pipeline-splits b1/2/relu b2/3/relu b3/5/relu --pipeline-schedule Grouped \
    --xla-recompute --optimiser momentum --momentum 0.90 --ckpts-per-epoch 1 \
    --max-cross-replica-buffer-size 100000000 --available-memory-proportion 0.6 0.6 0.6 0.6 0.6 0.6 0.16 0.2 \
    --internal-exchange-optimisation-target balanced --normalise-input --stable-norm \
    --enable-half-partials --lr-schedule cosine --label-smoothing 0.1 --eight-bit-io


### ImageNet - ResNeXt

ResNeXt is a variant of ResNet that adds multiple paths to the ResBlocks.

The following configuration will train a ResNeXt-101 model to 78.8% validation accuracy in 120 epochs on 16 Mk2 IPUs (the number of epochs has not been tuned):
   
    python train.py --model resnext --model-size 101 --dataset imagenet --data-dir .../imagenet-data \
    --shards 2 --replicas 8 --batch-size 6 --gradient-accumulation-count 16 --epoch 120 \
    --pipeline  --pipeline-splits b3/3/relu --pipeline-schedule Grouped \
    --xla-recompute --optimiser momentum --momentum 0.9 --ckpts-per-epoch 1 \
    --internal-exchange-optimisation-target balanced --disable-variable-offloading \
    --normalise-input --stable-norm --base-learning-rate -11 --no-validation \
    --enable-half-partials --lr-schedule cosine --label-smoothing 0.1 --eight-bit-io


### ImageNet - EfficientNet

We recommend that you split the model across 2 or 4 IPUs for the best throughput. For example this will
achieve good B0 model convergence pipelined over 4 IPUs, with the RMSprop optimiser and 32 bit precision
for weight updates:

    python train.py --model efficientnet --model-size B0 --data-dir .../imagenet-data --shards 4 \
    --batch-size 4 --pipeline --gradient-accumulation-count 128 --pipeline-splits block2b block4a block6a --xla-recompute \
    --pipeline-schedule Grouped --optimiser RMSprop --precision 16.32

Note that larger models (B1, B2, etc.) will automatically increase the image size used. You can
override this behaviour with the `--image-size` option if needed.

Changing the dimension of the group convolutions can make the model more efficient on the IPU. To keep the
number of parameters approximately the same, you can reduce the expansion ratio. For example a modified
EfficientNet-B0 model, with a similar number of trainable parameters can be trained using:

    python train.py --model efficientnet --model-size B0 --data-dir .../imagenet-data --batch-size 8 \
    --shards 4 --pipeline --gradient-accumulation-count 64 --pipeline-splits block2a/SE block4a block5c --xla-recompute \
    --pipeline-schedule Grouped --optimiser RMSprop --precision 16.32 --group-dim 16 --expand-ratio 4 --groups 4

This should give a similar validation accuracy as the standard model but with improved training throughput.

### ImageNet - EfficientNet - Inference

The training harness can also be used to demonstrate inference performance using the `validation.py` script.
For example, to check inference for EfficientNet use:

    python validation.py --model efficientnet --model-size B0 --dataset imagenet --batch-size 8 \
    --generated-data --repeat 10 --batch-norm

## Configurations

The training script supports a large number of program arguments, allowing you to make the most of the IPU
performance for a given model. In order to facilitate the handling of a large number of parameters, you can
define an ML configuration, in the more human-readable YAML format, in the `configs.yml` file. After the configuration
is defined, you can run the training script with this option:

    python3 train.py --config my_config

From the command line you can override any option previously defined in a configuration. For example, you can
change the number of training epochs of a configuration with:

    python3 train.py --config my_config --epochs 50

We provide reference configurations for the models described above.

# Distributed training

The following section explains how you can train models at scale using distributed training.


# PopDist and PopRun - distributed training on IPU-PODs

To get the most performance from our IPU-PODs, this application example now supports PopDist, Graphcore Poplar 
distributed configuration library. For more information about PopDist and PopRun, see the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/).
Beyond the benefit of enabling scaling an ML workload to an IPU-POD128 and above, the CNN application benefits from distributed
training even on an IPU-POD16 or IPU-POD64, since the additional launched instances increase the number of input data feeds,
therefore increasing the throughput of the ML workload.

For example the ResNet50 Group Norm configuration for 16 IPUs is defined in `configs.yml`. You can distribute this configuration on an IPU-POD16
with:

    poprun -v --numa-aware 1 --num-instances 8 --num-replicas 16 --ipus-per-replica 1 --mpi-local-args="--tag-output" \
    python3 train.py --config mk2_resnet50_gn_16ipus --data-dir your_dataset_dir_path --no-validation

As mentioned above, each instance sets an independent input data feed to the devices. The maximum number of instances is limited by
the number of replicas, so we could in theory, define `--num-instances 16` and have one input feed for each replica. However, we are
finding that given the NUMA configuration currently in use on IPU-PODs, the optimal maximum number of instances is 8. So while in total
there are 16 replicas, each instance is managing 2 local replicas. A side effect from this is that when executing distributed workloads,
the program option `--replicas value` is ignored and overwritten by the number of local replicas. However, all of this is done
programatically in `train.py` to make it more manageable.

Note that, there is currently a limitation with distributed training that prevents execution of validation after training in the same
process, therefore we need to pass the option `--no-validation`. Then, after training is complete, you can run validation with:

    poprun -v --numa-aware 1 --num-instances 8 --num-replicas 16 --ipus-per-replica 1 --mpi-local-args="--tag-output" \
    python3 validation.py --config mk2_resnet50_gn_16ipus --data-dir your_dataset_dir_path --restore-path generated_checkpoints_dir_path

When running pipelined models, such as the ResNet50 Batch Norm 16 IPU configuration, where each model is pipelined across 4 IPUs
during training, you need to change the distributed training command line accordingly:

    poprun -v --numa-aware 1 --num-instances 4 --num-replicas 4 --ipus-per-replica --mpi-local-args="--tag-output" \
    python3 train.py --config mk2_resnet50_bn_16ipus --data-dir your_dataset_path --no-validation

Note that we reduced the number of instances from 8 to 4 since we are only running 4 replicas.
During validation the model does not need to be pipelined as it fits in a single IPU. So distributed validation can be executed with:

    poprun -v --numa-aware 1 --num-instances 8 --num-replicas 16 --ipus-per-replica 1 --mpi-local-args="--tag-output" \
    python3 validation.py --config mk2_resnet50_bn_16ipus --shards 1 --data-dir your_dataset_dir_path  \
    --restore-path generated_checkpoints_dir_path

### Scaling from IPU-POD16 to IPU-POD64 and IPU-POD128

Beyond being equipped with 64 IPUs, a Graphcore IPU-POD64 can be equipped with up to four host servers. This allows another degree of scalability,
since we can now run instances across all 4 servers. To distribute a training job across multiple hosts, it is first assumed that the
filesystem structure is identical across those hosts. This is best achieved with a network shared filesystem, however if this is not
available the user needs to make sure that independent copies of the Poplar SDK, the examples repository and the datasets are located similarly
across the filesystems of the different hosts.

After the setup mentioned above is in place, the Poplar SDK only needs to be enabled on the host the user is connected to. The command line is then
extended with system information to make sure the other hosts execute the program with a similar development environment (here we assume that `$WORKSPACE` has been set appropriately):

    poprun -v --host 10.1.3.101,10.1.3.102,10.1.3.103,10.1.3.104 --numa-aware=yes --vipu-server-host=10.1.3.101 --vipu-partition=p64 \
    --reset-partition=no --update-partition=no --mpi-global-args="--tag-output --mca btl_tcp_if_include 10.1.3.0/24" \
    --mpi-local-args="-x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x TF_CPP_VMODULE=poplar_compiler=1 -x IPUOF_VIPU_API_TIMEOUT=300 \
    -x TF_POPLAR_FLAGS=--executable_cache_path=$WORKSPACE/exec_cache" --num-replicas=16 --num-instances=16 \
    --ipus-per-replica 4 python3 $WORKSPACE/examples/applications/tensorflow/cnns/training/train.py --config mk2_resnet50_bn_64ipus \
    --no-validation --data-dir your_dataset_path

Note that configuration `resnet50_bn_64_ipu` just changes the number of accumulated gradients in order to maintain the same global batch size while 
using more IPUs.
After training is complete, you can execute validation by adapting the instruction to run each model on 1 IPU, similarly to what was done above:

    poprun -v --host 10.1.3.101,10.1.3.102,10.1.3.103,10.1.3.104 --numa-aware=yes --vipu-server-host=10.1.3.101 --vipu-partition=p64 \
    --reset-partition=no --update-partition=no --mpi-global-args="--tag-output --mca btl_tcp_if_include 10.1.3.0/24" \
    --mpi-local-args="-x LD_LIBRARY_PATH -x PYTHONPATH -x TF_CPP_VMODULE=poplar_compiler=1 -x IPUOF_VIPU_API_TIMEOUT=300 \
    -x TF_POPLAR_FLAGS=--executable_cache_path=$WORKSPACE/exec_cache" --num-replicas=64 --num-instances=32 \
    --ipus-per-replica 1 python3 $WORKSPACE/examples/applications/tensorflow/cnns/training/train.py --config mk2_resnet50_bn_64ipus \
    --shards 1 --data-dir your_dataset_path --restore-path generated_checkpoints_dir_path

You can also scale your workload to multiple IPU-POD64s. To do so, the same assumptions are made regarding the identical filesystem structure across
the different host servers. As an example, to train a ResNet50 using an IPU-POD128 (which consists of two IPU-POD64s), using all 128 IPUs and 8 host servers
you can use the following instruction:


    POPLAR_ENGINE_OPTIONS='{"target.gatewayMode": "true", "target.syncReplicasIndependently": "true", "target.hostSyncTimeout": "900", \
     "target.maxStreamCallbackThreadsPerNumaNode": "auto"}' poprun -v --host \
      10.10.19.150,10.10.19.151,10.10.19.152,10.10.19.153,10.10.20.150,10.10.20.151,10.10.20.152,10.10.20.153 \
      --numa-aware=yes --vipu-server-host=10.3.19.150 --vipu-server-timeout=300 --vipu-partition=gcl128 --vipu-cluster=gcl --reset-partition=no \
      --update-partition=no --num-ilds=2 --mpi-global-args="--tag-output --allow-run-as-root --mca oob_tcp_if_include 10.10.0.0/16 \ 
      --mca btl_tcp_if_include 10.10.0.0/16" --mpi-local-args="-x LD_LIBRARY_PATH -x TF_CPP_VMODULE=poplar_compiler=1 -x PATH -x PYTHONPATH \ 
      -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_ENGINE_OPTIONS" --ipus-per-replica 4 --num-replicas=32 --num-instances=32 \
      python3 --config mk2_resnet50_bn_128ipus --no-validation --data-dir your_dataset_path

Note that the configuration `mk2_resnet50_bn_128ipus` only changes the number of replicas and gradient accumulation count to maintain the
same global batch size.


# Distributed training for IPU Servers equipped with C2 cards

Please note that this section is presented here for legacy reasons, for users of systems with equipped with C2 cards.
If you are using an IPU-POD, and are looking to distribute your workloads, please refer to the section above.

Training can also be distributed over multiple machines ("workers"). To do this, pass the argument
`--distributed` and set the environment variable `TF_CONFIG` with information about the distributed cluster.
The gradients are streamed to the host of each worker, then averaged across the workers over the network, and
finally streamed back to the IPUs where the weight updates are performed. To ensure that the workers perform
identical weight updates, you should turn off stochastic rounding. Distributed training gives a total batch
size of `num_workers * num_replicas * gradient_accumulation_count * batch_size`.

For example, to distribute the training over two machines, with hostnames worker0 and worker1, run this on worker0:

    export TF_CONFIG='{"cluster":{"worker":["worker0:3636","worker1:3636"]},"task":{"type":"worker","index":0}}'
    python train.py --dataset imagenet --data-dir .../imagenet-data --model-size 50 --batch-size 4 \
    --batches-per-step 1 --shards 4 --pipeline --gradient-accumulation-count 64 --pipeline-splits b1/2/relu b2/3/relu b3/5/relu \
    --xla-recompute --replicas 4 --distributed --no-stochastic-rounding --base-learning-rate -11

Run exactly the same commands on worker1, with the exception of changing the worker index in `TF_CONFIG` from 0 to 1:

    export TF_CONFIG='{"cluster":{"worker":["worker0:3636","worker1:3636"]},"task":{"type":"worker","index":1}}'

Alternatively, the `mpirun` program can be used to start the training over the cluster, and the worker index will
be picked up from the `OMPI_COMM_WORLD_RANK` environment variable:

    mpirun --tag-output -H worker0,worker1 -bind-to none -map-by slot -x LD_LIBRARY_PATH -x PATH \
    -x TF_CONFIG='{"cluster":{"worker":["worker0:3636","worker1:3636"]},"task":{"type":"worker"}}' \
    python train.py [...]

Note: Depending on the size of the model, you might have to limit stream copy merging to avoid overflowing the host
exchange outbound page table by setting, for example, `export POPLAR_ENGINE_OPTIONS='{"opt.maxCopyMergeSize": 8388608}'`.


# Training Options

Use `--help` to show all available options.

`--model`: By default this is set to `resnet` but other examples such as `efficientnet`, `squeezenet` and `resnext`
are available in the `Models/` directory. Consult the source code for these models to find the corresponding default options.

## ResNet model options

`--model-size` : The size of the model to use. Only certain sizes are supported depending on the model and input data
size. For ImageNet data values of 18, 34 and 50 are typical, and for Cifar 14, 20 and 32. Check the code for the full
ranges.

`--batch-norm` : Batch normalisation is recommended for medium and larger batch sizes that will typically be used
training with Cifar data on the IPU (and is the default for Cifar).
For ImageNet data smaller batch sizes are usually used and group normalisation is recommended (and is the default for ImageNet).

`--group-norm` : Group normalisation can be used for small batch sizes (including 1) when batch normalisation would be
unsuitable. Use the `--groups` option to specify the number of groups used (default 32).

## EfficientNet model options

`--model-size` : The model to use, default 'B0' for EfficientNet-B0. Also supported are 'B1' to 'B7', plus
an unofficial 'cifar' size which can be used with CIFAR sized datasets.

`--group-dim` : The dimension used for group convolutions, default 1. Using a higher group dimension can improve
performance on the IPU but will increase the number of parameters.

`--expand-ratio` : The EfficientNet expansion ratio, default 6. When using a higher group dimension this can be
reduced to keep the number of parameters approximately the same as the official models.

Use `python train.py --model efficientnet --help` to see other model options.


## Major options

`--batch-size` : The batch size used for training. When training on IPUs the batch size will typically be smaller than batch
sizes used on GPUs. A batch size of four or less is often used, but in these cases using group normalisation is
recommended instead of batch normalisation (see `--group-norm`).

`--base-learning-rate` : The base learning rate is scaled by the batch size to obtain the final learning rate. This
means that a single `base-learning-rate` can be appropriate over a range of batch sizes.

`--epochs` \ `--iterations` : These options determine the length of training and only one should be specified.

`--data-dir` : The directory in which to search for the training and validation data. ImageNet must be in TFRecord
format. CIFAR-10/100 must be in binary format. If you have set a `DATA_DIR` environment variable then this will be used
if `--data-dir` is omitted.

`--dataset` : The dataset to use. Must be one of `imagenet`, `cifar-10` or `cifar-100`. This can often be inferred from
the `--data-dir` option.

`--lr-schedule` : The learning rate schedule function to use. The default is `stepped` which is configured using
`--learning-rate-decay` and `--learning-rate-schedule`. You can also select `cosine` for a cosine learning rate.

`--warmup-epochs` : Both the `stepped` and `cosine` learning rate schedules can have a warmup length which linearly
increases the learning rate over the first few epochs (default 5). This can help prevent early divergences in the
network when weights have been set randomly. To turn off warmup set the value to 0.

`--gradient-accumulation-count` : The number of gradients to accumulate before doing a weight update. This allows the
effective mini-batch size to increase to sizes that would otherwise not fit into memory.
Note that when using `--pipeline` this is the number of times each pipeline stage will be executed.

`--eight-bit-io` : Transfer images to the IPUs in 8 bit format.

## IPU options

`--shards` : The number of IPUs to split the model over (default `1`). If `shards > 1` then the first part of the model
will be run on one IPU with later parts run on other IPUs with data passed between them. This is essential if the model
is too large to fit on a single IPU, but can also be used to increase the possible batch size. It is recommended to
use pipelining to improve throughput (see `--pipeline`).
It may be necessary to influence the automatic sharding algorithm using the `--sharding-exclude-filter`
and `--sharding-include-filter` options if the memory use across the IPUs is not well balanced,
particularly for large models.
These options specify sub-strings of edge names that can be used when looking for a cutting point in the graph.
They will be ignored when using pipelining which uses `--pipeline-splits` instead.

`--pipeline` : When a model is run over multiple IPUs (see `--shards`) pipelining the data flow can improve
throughput by utilising more than one IPU at a time. The splitting points for the pipelined model must
be specified, with one less split than the number of IPUs used. Use `--pipeline-splits` to specifiy the splits -
if omitted then the list of available splits will be output. The splits should be chosen to balance
the memory use across the IPUs. The weights will be updated after each pipeline stage is executed
the number of times specified by the `--gradient-accumulation-count` option.

`--precision` : Specifies the data types to use for calculations. The default, `16.16` uses half-precision
floating-point arithmetic throughout. This lowers the required memory which allows larger models to train, or larger
batch sizes to be used. It is however more prone to numerical instability and will have a small but significant
negative effect on final trained model accuracy. The `16.32` option uses half-precision for most operations but keeps
master weights in full-precision - this should improve final accuracy at the cost of extra memory usage. The `32.32`
option uses full-precision floating-point throughout (and will use more memory).

`--no-stochastic-rounding` : By default stochastic rounding on the IPU is turned on. This gives extra precision and is
especially important when using half-precision numbers. Using stochastic rounding on the IPU doesn't impact the
performance of arithmetic operations and so it is generally recommended that it is left on.

`--batches-per-step` : The number of batches that are performed on the device before returning to the host. Setting
this to 1 will make only a single batch be processed on the IPU(s) before returning data to the host. This can be
useful when debugging (especially when generating execution profiles) but the extra data transfers
will significantly slow training. The default value of 1000 is a reasonable compromise for most situations.
When using the `--distributed` option, `--batches-per-step` must be set to 1.

`--select-ipus` : The choice of which IPU to run the training and/or validation graphs on is usually automatic, based
on availability. This option can be used to override that selection by choosing a training and optionally also
validation IPU ID. The IPU configurations can be listed using the `gc-info` tool (type `gc-info -l` on the command
line).

`--fp-exceptions` : Turns on floating point exceptions.

`--no-hostside-norm` : Moves the image normalisation from the CPU to the IPU. This can help improve throughput if
the bottleneck is on the host CPU, but marginally increases the workload and code size on the IPU.

`--gc-profile` : Generates profiling files for use with the PopVision Graph Analyser.
Python must be run with the gc-profile command line tool:

    gc-profile -d profile_dir -- python train.py --gc-profile

The type of execution profile can be specified (default: IPU_PROFILE):

* NO_PROFILE: indicates that there should be no execution profiling.
* DEVICE_PROFILE: indicates that the execution profile should contain only device wide events.
* IPU_PROFILE: indicates that the profile should contain IPU level execution events.
* TILE_PROFILE: indicates that the profile should contain Tile level execution events.

NOTE: If using multiple iterations, the profile only considers the first iteration.

`--available-memory-proportion` : The approximate proportion of memory which is available for convolutions.
It may need to be adjusted (e.g. to 0.1) if an Out of Memory error is raised. A reasonable range is [0.05, 0.6].
Multiple values may be specified when using pipelining. In this case two values should be given for each pipeline stage
(the first is used for convolutions and the second for matmuls).



## Validation options

`--no-validation` : Turns off validation.

`--valid-batch-size` : The batch size to use for validation.

Note that the `validation.py` script can be run to validate previously generated checkpoints. Use the `--restore-path`
option to point to the checkpoints and set up the model the same way as in training. See also the `--max-ckpts-to-keep`
option when training.


## Other options

`--generated-data` : Uses a generated random dataset filled with random data. If running with this option turned on is
significantly faster than running with real data then the training speed is likely CPU bound.

`--max-ckpts-to-keep` : The maximum number of checkpoints to keep when training. Defaults to 1, except when the
validation mode is set to `end`.

`--replicas` : The number of replicas of the graph to use. Using `N` replicas increases the batch size by a factor of
`N` (as well as the number of IPUs used for training)

`--optimiser` : Choice of optimiser. Default is `SGD` but `momentum` and `RMSProp` are also available, and which
have additional options.

`--momentum` : Momentum coefficient to use when `--optimiser` is set to `momentum`. The default is `0.9`.

`--label-smoothing` : A label smoothing factor can help improve the model accuracy by reducing the polarity of the
labels in the final layer. A reasonable value might be 0.1 or 0.2. The default is 0, which implies no smoothing.

`--weight-decay` : Value for weight decay bias. Setting to 0 removes weight decay.

`--loss-scaling` : When using mixed or half precision, loss scaling can be used to help preserve small gradient values.
This scales the loss value calculated in the forward pass (and unscales it before the weight update) to reduce the
chance of gradients becoming zero in half precision. The default value should be suitable in most cases.

`--seed` : Setting an integer as a seed will make training runs reproducible. Note that this limits the
pre-processing pipeline on the CPU to a single thread which will significantly increase the training time.

`--standard-imagenet` : By default the ImageNet preprocessing pipeline uses optimisations to split the dataset in
order to maximise CPU throughput. This option allow you to revert to the standard ImageNet preprocessing pipeline.

`--no-dataset-cache` : Don't keep a cache of the ImageNet dataset in host RAM. Using a cache gives a speed boost
after the first epoch is complete but does use a lot of memory. It can be useful to turn off the cache if multiple
training runs are happening on a single host machine.


# Resuming training runs

Training can be resumed from a checkpoint using the `restore.py` script. You must supply the `--restore-path` option
with a valid checkpoint.


# Profiling

You can generate profiling information that can be used with the PopVision Graph Analyser tool. See `--gc-profile`, above.
