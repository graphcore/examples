# CNN Training on IPUs

This readme describes how to run CNN models such as ResNet and EfficientNet for image recognition training on the IPU.

## Overview

Deep CNN residual learning models such as ResNet and EfficientNet are used for image recognition and classification.

The training examples given below use models implemented in TensorFlow, optimised for Graphcore's IPU.

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

A ResNeXt model is also available.

## Quick start guide

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to run the `enable.sh` script for Poplar and activate a Python 3 virtualenv with
   the TensorFlow 1 wheel from the Poplar SDK installed.
2. Download the data. See below for details on obtaining the datasets.
3. Install the packages required by this application using `pip install -r requirements.txt`
4. Run the training script. For example:
   `python3 train.py --dataset imagenet --data-dir path-to/imagenet`

### Datasets

You can download the ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes,
from http://image-net.org/download. It is approximately 150GB for the training and validation sets.

The CIFAR-10 dataset is available from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz, and the
CIFAR-100 dataset is available from https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz.

## File structure

| File / Subdirectory | Description               |
|---------------------|---------------------------|
| `train.py`          | The main training program |
| `validation.py`     | Contains the validation code used in `train.py` but also can be run to perform validation on previously generated checkpoints. The options available in validation should be set to be the same as in training - note that not all options used in training are available in validation (e.g. learning rate or pipelining related options are ignored when parsed through a config file or throw an error when parsed from the terminal). If training was performed using `--config`, the same configuration can be given and relevant parameters will be parsed. The `--restore-path` option can either point to the log directory of the previously generated checkpoints (if arguments.json is present, `global_batch_size` is extracted from the file), or a specific checkpoint (then `global_batch_size` becomes `--micro-batch-size`) |
| `restore.py`        | Used for restoring a training run from a previously saved checkpoint. For example: `python restore.py --restore-path logs/RN20_bs32_BN_16.16_v0.9.115_3S9/` |
| `ipu_optimizer.py`  | Custom optimizer |
| `ipu_utils.py`      | IPU specific utilities |
| `log.py`            | Module containing functions for logging results |
| `Datasets/`         | Code for using different datasets. Currently CIFAR-10, CIFAR-100 and ImageNet are supported |
| `Models/`           | Code for neural networks<br/>- `resnet.py`: Definition for ResNet model.<br/>- `resnext.py`: Definition for ResNeXt model.<br/>- `efficientnet.py`: Definition for EfficientNet models.
| `LR_Schedules/`     | Different LR schedules<br/> - `stepped.py`: A stepped learning rate schedule with optional warmup<br/>- `cosine.py`: A cosine learning rate schedule with optional warmup <br/>- `exponential.py`: An exponential learning rate schedule with optional warmup <br/>- `polynomial_decay_lr.py`: A polynomial learning rate schedule with optional warmup
| `requirements.txt`  | Required packages for the tests |
| `weight_avg.py`     | Code for performing weight averaging of multiple checkpoints. |
| `configurations.py` | Code for parsing configuration files. |
| `configs.yml`       | File where configurations are defined. |
| `test/`             | Test files - run using `python3 -m pytest` after installing the required packages. |

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

## Configurations

The training script supports a large number of program arguments, allowing you to make the most of the IPU
performance for a given model. In order to facilitate the handling of a large number of parameters, you can
define an ML configuration, in the more human-readable YAML format, in the `configs.yml` file. After the configuration
is defined, you can run the training script with this option:

    python3 train.py --config my_config

From the command line you can override any option previously defined in a configuration. For example, you can
change the number of training epochs of a configuration with:

    python3 train.py --config my_config --epochs 50

We provide reference configurations for the models described below.

## Selecting checkpoints for validation

With `validation.py` either a single or all generated checkpoints can be evaluated. To evaluate a single checkout specify its name `--restore-path`:

    python3 validation.py --config my_config --data-dir <path-to-dataset> --restore-path logs/<training-dir>/<ckpt-id>

To evaluate all generated checkpoints specify the directory that contains them:

    python3 validation.py --config my_config --data-dir <path-to-dataset> --restore-path logs/<training-dir>


## PopDist and PopRun - distributed training on IPU-PODs

To get the most performance from our IPU-PODs, this application example now supports PopDist, Graphcore Poplar
distributed configuration library. For more information about PopDist and PopRun, see the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/).
The CNN application benefits from distributed training on IPU-POD16 or IPU-POD64, since the additional launched instances increase the number of input data feeds,
therefore increasing the throughput of the ML workload.

For example the ResNet50 Group Norm configuration for 16 IPUs is defined in `configs.yml`. You can distribute this configuration on an IPU-POD16
with:

    poprun -v --num-instances 8 --num-replicas 16 \
    python3 train.py --config resnet50_gn_16ipus --data-dir your_dataset_dir_path --no-validation

As mentioned above, each instance sets an independent input data feed to the devices. The maximum number of instances is limited by
the number of replicas, so we could in theory, define `--num-instances 16` and have one input feed for each replica. However, we are
finding that given the NUMA configuration currently in use on IPU-PODs, the optimal maximum number of instances is 8. So while in total
there are 16 replicas, each instance is managing 2 local replicas. A side effect from this is that when executing distributed workloads,
the program option `--replicas value` is ignored and overwritten by the number of local replicas. However, all of this is done
programatically in `train.py` to make it more manageable.

Note that, there is currently a limitation with distributed training that prevents execution of validation after training in the same
process, therefore we need to pass the option `--no-validation`. Then, after training is complete, you can run validation with:

    poprun -v --num-instances 8 --num-replicas 16 \
    python3 validation.py --config resnet50_gn_16ipus --data-dir your_dataset_dir_path --restore-path generated_checkpoints_dir_path

When running pipelined models, such as the ResNet50 Batch Norm 16 IPU configuration, where each model is pipelined across 4 IPUs
during training, you need to change the distributed training command line accordingly:

    poprun -v --num-instances 4 --num-replicas 4 --ipus-per-replica 4 \
    python3 train.py --config resnet50_bn_16ipus --data-dir your_dataset_path --no-validation

Note that we reduced the number of instances from 8 to 4 since we are only running 4 replicas.
During validation the model does not need to be pipelined as it fits in a single IPU. So distributed validation can be executed with:

    poprun -v --num-instances 8 --num-replicas 16 \
    python3 validation.py --config resnet50_bn_16ipus --shards 1 --data-dir your_dataset_dir_path  \
    --restore-path generated_checkpoints_dir_path

### Scaling from IPU-POD16 to IPU-POD64

Beyond being equipped with 64 IPUs, a Graphcore IPU-POD64 can be equipped with up to four host servers. This allows another degree of scalability,
since we can now run instances across all 4 servers. To distribute a training job across multiple hosts, it is first assumed that the
filesystem structure is identical across those hosts. This is best achieved with a network shared filesystem, however if this is not
available the user needs to make sure that independent copies of the Poplar SDK, the examples repository and the datasets are located similarly
across the filesystems of the different hosts.

After the setup mentioned above is in place, the Poplar SDK only needs to be enabled on the host the user is connected to. The command line is then
extended with system information to make sure the other hosts execute the program with a similar development environment. We assume that `$WORKSPACE` has been set appropriately. Replace with IP addresses as appropriate for the target hardware. `--mca btl_tcp_if_include xxx.xxx.xxx.0/xx` sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address. Replace `pod64_partition_name` with the name of your POD64 partition.

    poprun -v --host xxx.xxx.xxx.1,xxx.xxx.xxx.2,xxx.xxx.xxx.3,xxx.xxx.xxx.4 --vipu-server-host=xxx.xxx.xxx.xxx \ --vipu-partition=pod64_partition_name --reset-partition=yes --mpi-global-args=" \
    --mca btl_tcp_if_include xxx.xxx.xxx.0/24" --mpi-local-args="-x LD_LIBRARY_PATH -x PATH -x PYTHONPATH \
    -x TF_CPP_VMODULE=poplar_compiler=1 -x IPUOF_VIPU_API_TIMEOUT=300 -x TF_POPLAR_FLAGS=--executable_cache_path=$WORKSPACE/exec_cache" \ --num-replicas=16 --num-instances=16 --ipus-per-replica 4 python3 $WORKSPACE/examples/vision/cnns/tensorflow1/training/train.py --config resnet50_bn_64ipus --no-validation --data-dir your_dataset_path

Note that configuration `resnet50_bn_64_ipu` just changes the number of accumulated gradients in order to maintain the same global batch size while
using more IPUs.
After training is complete, you can execute validation by adapting the instruction to run each model on 1 IPU, similarly to what was done above:

    poprun -v --host xxx.xxx.xxx.1,xxx.xxx.xxx.2,xxx.xxx.xxx.3,xxx.xxx.xxx.4 --vipu-server-host=xxx.xxx.xxx.xxx \ --vipu-partition=pod64_partition_name --reset-partition=yes --mpi-global-args=" \
    --mca btl_tcp_if_include xxx.xxx.xxx.0/24 --mca oob_tcp_if_include xxx.xxx.xxx.0/24" \
    --mpi-local-args="-x LD_LIBRARY_PATH -x PYTHONPATH -x TF_CPP_VMODULE=poplar_compiler=1 -x IPUOF_VIPU_API_TIMEOUT=300 \
    -x TF_POPLAR_FLAGS=--executable_cache_path=$WORKSPACE/exec_cache" --num-replicas=64 --num-instances=32 \
    --ipus-per-replica 1 python3 $WORKSPACE/examples/vision/cnns/tensorflow1/training/validation.py \
    --config resnet50_bn_64ipus --shards 1 --data-dir your_dataset_path --restore-path generated_checkpoints_dir_path

## Convergence Optimized configurations

### ImageNet - ResNet-50

The following configuration trains ResNet50 using 16 IPUs. Each IPU runs a single data-parallel replica of the
model with a micro-batch size of 20. We use a gradient accumulation count of 6, and 16 replicas in total for a
global batch size of 1920 (20 * 6 * 16). Activations for the forwards pass are re-calculated during the backwards pass. Partials
saved to tile memory within the convolutions are set to half-precision to maximise throughput on the IPU. Batch norm statistics
computed for each batch of samples are distributed across groups of 2 IPUs to improve numerical stability and convergence,
This example uses the SGD-M optimizer, cosine learning rate and label smoothing to train to >75.90% validation accuracy in 45 epochs.
The example uses PopDist with 8 instances to maximize throughput.

    POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false"}' poprun -vv \
    --mpi-local-args='-x POPLAR_ENGINE_OPTIONS' \
    --num-instances 8 --num-replicas 16 python train.py --config resnet50_mlperf_pod16_bs20 --epochs-per-sync 20 \
    --data-dir your_dataset_path --no-validation

After training is complete, you can validate the previously saved checkpoints. As above, each IPU runs one replica of the model and
the model is replicated over the 16 IPUs. To make sure there are no validation samples discarded when sharding the validation dataset across 8 instances, a batch size of 25 is used.

    POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false"}' poprun -vv \
    --mpi-local-args='-x POPLAR_ENGINE_OPTIONS' \
    --num-instances 8 --num-replicas 16 python validation.py --config resnet50_mlperf_pod16_bs20 --no-stochastic-rounding \
    --micro-batch-size 25 --available-memory-proportion 0.6 --data-dir your_dataset_path --restore-path generated_checkpoints_dir_path

For POD64 systems, a similar configuration is used. Again each IPU runs a single replica of the model with a micro-batch size of 20. To maintain a similar global batch size we use a gradient accumulation count of 2 and 64 replicas for a global batch size of 2560. We also use 32 instances to maximize throughput. The optimiser, partials, activations recomputation and distributed batch norm setting are the same as for POD16 systems.

    POPLAR_ENGINE_OPTIONS='{"opt.enableMultiAccessCopies":"false"}' POPLAR_TARGET_OPTIONS='{"gatewayMode":"false"}' \
    poprun -vv --host xxx.xxx.1.1,xxx.xxx.1.2,xxx.xxx.1.3,xxx.xxx.1.4 \
    --mpi-global-args='--mca oob_tcp_if_include xxx.xxx.xxx.0/24 \
    --mca btl_tcp_if_include xxx.xxx.xxx.0/24' --mpi-local-args=' -x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH \
    -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_ENGINE_OPTIONS' \
    --reset-partition=yes --vipu-server-timeout 300 \
    --only-output-from-instance 0 --vipu-server-host xxx.xxx.xxx.xxx --vipu-partition=pod64_partition_name \
    --num-instances 32 --num-replicas 64 python train.py --config resnet50_mlperf_pod64_bs20 --epochs-per-sync 20 \
    --data-dir your_dataset_path --no-validation

To run validation without discarding samples when distributing data across batches and instances, the above validation instruction for the POD16 is used.



### ImageNet - ResNeXt

ResNeXt is a variant of ResNet that adds multiple paths to the Residual Blocks.

The following configuration will train a ResNeXt-101 model to 78.8% validation accuracy in 120 epochs on 16 IPUs. The model is pipelined over 2 IPUs with a micro batch size 6. We use a gradient accumulation count of 16 and 8 replicas for a global batch of 2048 (6 * 16 * 8).

    poprun --num-replicas 8 --ipus-per-replica 2 \
    --num-instances 8 python3 train.py --config resnext101_16ipus --data-dir your_dataset_path --no-validation

As above, you can run validation after training:

    poprun --num-replicas 16 \
    --num-instances 16 python3 validation.py --config resnext101_16ipus --shards 1 --data-dir your_dataset_path \
    --restore-path generated_checkpoints_dir_path

### ImageNet - EfficientNet

The following configuration trains EfficientNet-B4 to ~82% using 16 IPUs. Each model is pipelined across 4 IPUs with a micro-batch size of 3. We use a gradient accumulation count of 64, and 4 replicas in total for a global batch size of 768 (3 * 64 * 4).

    poprun --num-replicas 4 --num-instances 4 --ipus-per-replica 4 \ python3 train.py --config efficientnet_b4_g1_16ipus --data-dir your_dataset_path --no-validation

As above, you can run validation after training:

    poprun -v --num-instances 8 --num-replicas 16 \
    python3 validation.py --config efficientnet_b4_g1_16ipus --shards 1 --data-dir your_dataset_dir_path  \
    --restore-path generated_checkpoints_dir_path

Changing the dimension of the group convolutions can make the model more efficient on the IPU. To keep the
number of parameters approximately the same, you can reduce the expansion ratio. For example a modified
EfficientNet-B4 model, with a similar number of trainable parameters can be trained using:

    poprun --num-replicas 8 --num-instances 8 --ipus-per-replica 2 \
    python3 train.py --config efficientnet_b4_g16_16ipus --data-dir your_dataset_path --no-validation --identical-replica-seeding

This configuration trains EfficientNet-B4 to ~82.6% validation accuracy with improved training throughput, achieved by using half-precision arithmetic throughput and by pipelining across just 2 IPUs. The global batch size is 6144, enabled by using the LARS optimizer and polynomial decay learning rate, in addition to other hyperparameter tuning. This makes the configuration appropriate for a range of different sized systems. For example, to train over 64 IPUs, simply change the `--config` argument in the above argument to `efficientnet_b4_g16_64ipus` and `--num-replicas 32`.

The original EfficientNet-B4 described by [Tan and Le](https://arxiv.org/abs/1905.11946) fits on a single IPU and uses distributed batch norm, in which batches are split across replicas and the mean and variance statistics are calculated across several replicas.
In order to train as expected, a minimum of 8 IPUs are required to train 8 replicas concurrently.
The `efficientnet_b4_g1_with_batch_norm_16ipus` configuration replicates the model over 16 IPUs and calculates batch norm statistics across 8 replicas. The micro-batch size is 5 so the statistics are calculated across 40 examples (8 * 5).
This configuration achieves ~82.6% ImageNet Top-1 accuracy in 350 epochs. It uses 16-bit floating point arithmetic for maximum throughput. It can be executed with the following command:

```sh
python train.py --config efficientnet_b4_g1_with_batch_norm_16ipus
```

A similar config adapted to run on 64 IPUs is also available. It can be run with `poprun` to achieve much shorter time to train
given additional hardware. That command which uses 64 IPUs is available as part of the benchmarking suite under `tf1_efficientnet_b4_batchnorm_train_real_pod64_conv`. This command uses multiple CPU hosts to perform the data loading across the 64 replicas. For this configuration to work additional environment variables must be set:

- `DATASET_DIR`: the path to a directory containing a folder called `imagenet-data` containing the ImageNet dataset;
- `HOSTS`: must a comma separated list of host names or IPs for `poprun` to connect to;
- `TCP_IF_INCLUDE`: must be an IP subnet which all hosts have access to (not the rdma network), and not a subnet in which their are interfaces with multiple IP addresses (not the management network);
- V-IPU host and partition will be collected from the standard `IPUOF_VIPU_API_HOST` and `IPUOF_VIPU_API_PARTITION_ID`  environment variables.

Once these variables are set in your shell, run training followed by validation using this command:

```sh
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark\
    tf1_efficientnet_b4_batchnorm_train_real_pod64_conv \
    tf1_efficientnet_b4_batchnorm_infer_real_pod16
```

### ImageNet - EfficientNet - Inference

The training harness can also be used to demonstrate inference performance using the `validation.py` script.
For example, to check inference for EfficientNet use:

    python validation.py --model efficientnet --model-size B0 --dataset imagenet --micro-batch-size 8 \
    --generated-data --repeat 10 --batch-norm

There is also a possibility to run inference using the [embedded application runtime](https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/tensorflow/embedded_application_runtime.html#ipu-embedded-application-runtime) which allows us to save a
precompiled graph to a file and skip the compilation in the subsequent runs. It can be tested
using the `inference_embedded.py` script. Each time the script is executed it looks for the
precompiled graph in the working directory, then loads it and executes for the given number of
iterations. If the graph is not found, then it is constructed, compiled and saved to a file.
For example, to test the performance of EfficientNet inference use:

    python inference_embedded.py --model efficientnet --model-size B0 --dataset imagenet --micro-batch-size 1 \
    --iterations 1000 --device-iterations 100 --eight-bit-io --no-dataset-cache --generated-data


# View the results in Weights & Biases
Weights and Biases is a tool that helps you tracking different metrics of your machine learning job, for example the loss and accuracy but also the memory utilisation. For more information please see https://www.wandb.com/.
Installing the `requirements.txt` file will install a version of wandb.
You can login to wandb as you prefer and then simply activate it using the flag --wandb, eg.
```shell
python train.py --config resnet8_test --wandb
```

Near the start of the run you will see a link to your run appearing in your terminal.


# Training Options

Use `--help` to show all available options.

`--model`: By default this is set to `resnet` but other examples such as `efficientnet` and `resnext`
are available in the `Models/` directory. Consult the source code for these models to find the corresponding default options.

## ResNet model options

`--model-size` : The size of the model to use. Only certain sizes are supported depending on the model and input data
size. For ImageNet data values of 18, 34 and 50 are typical, and for Cifar 14, 20 and 32. Check the code for the full
ranges.

`--batch-norm` : Batch normalisation is recommended for medium and larger batch sizes that will typically be used
training with Cifar data on the IPU (and is the default for Cifar).
For ImageNet data smaller batch sizes are usually used and group normalisation is recommended (and is the default for ImageNet).
For a distributed batch norm across multiple replicas,
the `--BN-span` option can be used to specify the number of replicas.

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

`--micro-batch-size` : The micro batch size used for training. When training on IPUs the batch size will typically be smaller than batch
sizes used on GPUs. A micro batch size of four or less is often used, but in these cases using group normalisation is
recommended instead of batch normalisation (see `--group-norm`).

`--base-learning-rate-exponent` : The base learning rate exponent, N, is used to set the value of the base learning rate, which is 2<sup>N</sup>. The base learning rate is scaled by the batch size to obtain the final learning rate. This
means that a single base learning rate can be appropriate over a range of batch sizes.

`--epochs` \ `--iterations` : These options determine the length of training and only one should be specified.

`--data-dir` : The directory in which to search for the training and validation data. ImageNet must be in TFRecord
format. CIFAR-10/100 must be in binary format. If you have set a `DATA_DIR` environment variable then this will be used
if `--data-dir` is omitted.

`--dataset` : The dataset to use. Must be one of `imagenet`, `cifar-10` or `cifar-100`. This can often be inferred from
the `--data-dir` option.

`--ckpts-per-epoch` : How often a checkpoint would be saved in `logs/<run>/`. By default it's 1.

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
be specified, with one less split than the number of IPUs used. Use `--pipeline-splits` to specify the splits -
if omitted then the list of available splits will be output. The splits should be chosen to balance
the memory use across the IPUs. The weights will be updated after each pipeline stage is executed
the number of times specified by the `--gradient-accumulation-count` option.
It is also possible to pipeline the model on a single IPU in order to make use of recomputation.
You can use `--pipeline --shards 1 --pipeline-schedule Sequential --enable-recompute`
and the respective `--pipeline-splits`
to define the recomputation points.

`--pipeline-schedule`: There are three options.
In the `Grouped` configuration (default), forward passes are grouped together
and the backward passes are grouped together.
This makes the pipeline more balanced, especially when the forward passes have similar processing duration
and the backward passes.
Otherwise, the pipeline has to wait for the slowest processing step.
In the `Interleaved` scheme, backward and forward passes are interleaved.
The `Sequential` option is mainly used for debugging
but is required when pipelining a model on a single IPU (in order to make use of recomputation).
It distributes the processing over multiple IPUs but processes samples sequentially, one after the other.

`--precision` : Specifies the data types to use for calculations. The default, `16.16` uses half-precision
floating-point arithmetic throughout. This lowers the required memory which allows larger models to train, or larger
batch sizes to be used. It is however more prone to numerical instability and will have a small but significant
negative effect on final trained model accuracy. The `16.32` option uses half-precision for most operations but keeps
master weights in full-precision - this should improve final accuracy at the cost of extra memory usage. The `32.32`
option uses full-precision floating-point throughout (and will use more memory).

`--no-stochastic-rounding` : By default stochastic rounding on the IPU is turned on. This gives extra precision and is
especially important when using half-precision numbers. Using stochastic rounding on the IPU doesn't impact the
performance of arithmetic operations and so it is generally recommended that it is left on.

`--device-iterations` : The number of iterations that are performed on the device before returning to the host. Setting
this to 1 will make only a single batch be processed on the IPU(s) before returning data to the host. This can be
useful when debugging (especially when generating execution profiles) but the extra data transfers
will significantly slow training. The default value of 1000 is a reasonable compromise for most situations.
When using the `--distributed` option, `--device-iterations` must be set to 1.

`--select-ipus` : The choice of which IPU to run the training and/or validation graphs on is usually automatic, based
on availability. This option can be used to override that selection by choosing a training and optionally also
validation IPU ID. The IPU configurations can be listed using the `gc-info` tool (type `gc-info -l` on the command
line).

`--fp-exceptions` : Turns on floating point exceptions.

`--no-hostside-norm` : Moves the image normalisation from the CPU to the IPU. This can help improve throughput if
the bottleneck is on the host CPU, but marginally increases the workload and code size on the IPU.

`--available-memory-proportion` : The approximate proportion of memory which is available for convolutions.
It may need to be adjusted (e.g. to 0.1) if an Out of Memory error is raised. A reasonable range is [0.05, 0.6].
Multiple values may be specified when using pipelining. In this case two values should be given for each pipeline stage
(the first is used for convolutions and the second for matmuls).



## Validation options

`--no-validation` : Turns off validation.

Note that the `validation.py` script can be run to validate previously generated checkpoints: Use the `--restore-path`
option to point to the checkpoints, and either select the same configuration file as used in training with `--config` or pass `--model`, `--model-size`, `--dataset` matching those used in the training. If `arguments.json` is present the directory given in `--restore-path`, `global_batch_size` is extracted from that file, but it can be overwritten with `--global-batch-size` argument. If neither the argument nor the file are available, then `global_batch_size` becomes `--micro-batch-size`).

## Other options

`--generated-data` : Uses a generated random dataset filled with random data. If running with this option turned on is
significantly faster than running with real data then the training speed is likely CPU bound.

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
If using ImageNet then you should also set the `--standard-imagenet` option when setting a seed in order to
have a reproducible data pipeline.

`--standard-imagenet` : By default the ImageNet preprocessing pipeline uses optimisations to split the dataset in
order to maximise CPU throughput. This option allow you to revert to the standard ImageNet preprocessing pipeline.

`--no-dataset-cache` : Don't keep a cache of the ImageNet dataset in host RAM. Using a cache gives a speed boost
after the first epoch is complete but does use a lot of memory. It can be useful to turn off the cache if multiple
training runs are happening on a single host machine.


# Resuming training runs

Training can be resumed from a checkpoint using the `restore.py` script. You must supply the `--restore-path` option
with a valid checkpoint.
