# CNN Training on IPUs

This readme describes how to run CNN models such as ResNet for image recognition training on the IPU.

## Overview

Deep CNN residual learning models such as ResNet are used for image recognition and classification.
The training examples given below use models implemented in TensorFlow 2, optimised for Graphcore's IPU.

## Quick start guide

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to run the `enable.sh` script for Poplar and activate a Python 3 virtualenv with
   the TensorFlow 2 and IPU TensorFlow Addons wheels from the Poplar SDK installed.
2. Download the data. See below for details on obtaining the datasets.
3. Install the packages required by this application using (`pip install -r requirements.txt`)
4. Run the training script. You can check that the code runs with the simplest example:
   `python3 train.py`. See below for optimized instructions and options descriptions.

### Datasets

You can download the ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes,
from http://image-net.org/download or https://www.kaggle.com/c/imagenet-object-localization-challenge/data.
It is approximately 150GB for the training and validation sets.

For many other datasets, such as CIFAR-10, CIFAR-100 or MNIST datasets, we use TFDS (TensorFlow Datasets)
to load the data, and if it is not present on the disk, it will be automatically downloaded.

## File structure

| File / Subdirectory      | Description               |
|--------------------------|---------------------------|
| `callbacks/`             | Custom callback classes to pass to model.fit() or model.validate() |
| `configuration/`          | Code for parsing configuration from a file (e.g. configs.yml) or terminal arguments. |
| `data/`                  | Code for building and benchmarking dataset pipelines |
| `losses/`                | Loss functions and loss wrappers for e.g. scaling or logging |
| `metrics/`               | Code related to metrics used to evaluate the model |
| `model/`                 | Custom models and transformations applied to change them after the instantiation |
| `normalization/`         | Layers for feature normalization |
| `optimizers/`            | Optimizers and regularizers |
| `schedules/`             | Learning rate schedules and decorators |
| `scripts/`               | Auxiliary scripts |
| `test/`                  | Tests that can be safely executed in parallel |
| `tests_serial/`          | Tests that should be executed one at the time |
| `batch_config.py`        | Class that infers correct number of micro and global batches based on parsed arguments such as gradient accumulation |
| `configs.yml`            | File where configurations are defined |
| `custom_exceptions.py`   | Custom exceptions with more descriptive names |
| `eight_bit_transfer.py`  | Code related to 8 bit io |
| `ipu_config.py`          | Code related with the configuration of the IPUs |
| `precision.py`           | Handles application of floating point precision policies |
| `requirements.txt`       | Project third-party dependencies |
| `seed.py`                | Allows to set a seed for random number generator for reproducibility of results |
| `send_request.py`        | TensorFlow Serving executor.
| `test_common.py`         | Package that both test/ and tests_serial/ share |
| `time_to_train.py`       | Provides code for measuring TTT (Total-Time-to-Train) |
| `train.py`               | The main training program |
| `utilities.py`           | Shared methods used across the whole project |

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

## Executing training only or validation only

By default, the application will execute training and afterwards it will validate all model checkpoints produced during training. You can disable any of these two phases if necessary. To disable validation, just append `--validation False` to your instruction. This terminates the program immediately after training finishes. Similarly, if you want to disable training, just append `--training False` to your instruction. Note that by default, when training is disabled, validation will be done on a randomly initialized model. However, you can additionally specify a directory of checkpoints, with `--checkpoint-dir your_previously_generated_directory`, to run validation on the previously trained model without training the model any further.


## Configuration files

As can be observed from the above, the application provides a number of options to efficiently target execution on the IPUs. To facilitate building complex instructions, the user can instead write the instruction in a configuration file. The configuration is defined in YAML format. See `configs.yml` configuration file for examples of configuration instructions. After a configuration is specified you can use the configuration simply with:  
    `python3 train.py --config your_config --config-path your_config_file`

To facilitate experimentation, you can also overload specific parameters of a configuration directly from the command line:  
    `python3 train.py --config your_config --config-path your_config_file --some-param new_value`


## Periodic host events
While the bulk of the work is done by the IPUs, there are a number of operations we may want to do on the host CPU throughout the training run, such as logging or saving a checkpoint of the model. How often we want to execute these events has an impact on the throughput of the application. This is because the application is built in a way that the program to be executed on the IPUs is wrapped in a loop such that multiple iterations of the program can be performed on the IPUs before handing back control to the host CPU. By minimizing the handover between the IPUs and the host CPU we can reduce the total execution time of the program. The number of iterations executed on the IPUs can be adjusted to match the period of the host events.  

Note that depending on dataset size and batch configuration, not all frequency values are valid. More specifically, the switching from the IPU to the CPU must happen after a weight update has been completed. For example, a given dataset and batch configuration result in an odd number of weight updates per epoch, it is not possible to log the training metrics 2 times per epoch, because one of the events would happen in the middle of a weight update. To avoid these situations and to facilitate the user experience, the program automatically calculates the closest possible frequency that doesn't attempt to interrupt a weight update. However it is possible that the corresponding event is executed at a higher or lower frequency than originally requested.

Some examples of adjusting the frequency of logging the state of the training run over time:  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1`  
Sets steps per execution such that it executes one entire epoch on the device before returning to the host, while printing one log per epoch.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 2`  
It's not possible to execute half an epoch on the device because it would correspond to a partial weight update. It will execute 1 epoch on the device and print 1 log per epoch. It logs a warning to the user highlighting the logging frequency can't be executed and will be adjusted.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 10`  
It's not possible to execute 1/10 of an epoch on the device. The closest is 1/11. So the generated program executes 1/11 of an epoch before returning to the host and therefore 11 logs per epoch. A warning is logged.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1/2`  
The generated program is going to execute 2 epochs on the device before returning to the host. Therefore the program is only repeated for 3 times.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1/3`
The generated program is going to execute 3 epochs on the device before returning to the host. Therefore the program is only repeated for 2 times.  

`python3 train.py --model cifar_resnet8 --num-epochs 5 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1/2`  
An exception is raised: "ValueError: It is not possible to log 2.0 epochs a time for 5 epochs".  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 0`  
The entire training run is executed on the device and a warning message is logged so the user is notified.  


## Synthetic data

It is possible to execute the training program without a real dataset and instead using synthetic data. This is done by using the `--synthetic-data` option. When using synthetic data, the data can be generated on the host CPU and transferred to the IPU or the data can be generated directly on the IPU avoiding the need for any data transfer between the CPU and the IPU. To generate synthetic data on the CPU use `--synthetic-data cpu`, and to generate synthetic data on the IPU use `--synthetic-data ipu`.


## PopDist and PopRun - distributed training on IPU-PODs

To get the most performance from our IPU-PODs, this application example now supports PopDist, Graphcore Poplar
distributed configuration library. For more information about PopDist and PopRun, see the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/).

A very simple example of using distributed training can be done with:

    poprun --num-instances 16 --num-replicas 16 python3 train.py

Each instance sets an independent input data feed to the devices. The maximum number of instances is limited by
the number of replicas, in this case 16. A side effect from this is that when executing distributed workloads,
the program option `--num-replicas value` is ignored.

Note that the default behaviour when combining distributed training and pipelined model will not work. By default, the app takes a pipelined training model and assumes the model fits in a single IPU during training. This means that the number of replicas changes between training and validation. While this improves performance, it is incompatible with distributed training which expects a constant number of replicas and ipus per replica throughout the program execution. So avoid this limitation we provide a `--pipeline-validation-model` command line option which uses the same the pipeline splits in training and validation:

    poprun --num-instance X --num-replicas Y --ipus-per-replica Z python3 train.py --pipeline-splits split1 ... splitN --pipeline-validation-model

## Regularization
The application allows to apply the squared norm of the weights to the loss in two ways, which are `--weight-decay <lambda>` and `--l2-regularization <lambda>`. They are mathematically equivalent when applied to stateless optimisers (e.g. SGD without the momentum), however, for stateful optimisers they behave differently. `--l2-regularization` affects the optimiser state directly, whereas `--weight-decay` doesn't, i. e.:
- weight decay:  `delta_W = -lr*(O(dL/dW, s) + lambda*W)`
- L2 regularization:  `delta_W = -lr*(O(dL/dW + lambda*W, s)`

where `L` is the loss with respect to weights `W`, `O` is an optimizer with state `s` and `lambda` is the regularization coefficient.

## Convergence optimized configurations

### ImageNet - ResNet-50

The following configuration trains ResNet50 using 16 Mk2 IPUs. The model is split over 4 IPUs and executed according to a pipeline execution scheme. The model weights and computation are represented using 16 bit floating point numbers, and matmuls and convolutions partials are also computed in the same representation. With the above settings it is possible to fit on each IPU a micro batch size of 16 samples. Additionally the model is replicated 4 times (4 pipeline stages * 4 replicas = 16 IPUs), and gradients are accumulated 128 times, therefore each weight update corresponds to a global batch size of 8192 samples (16 micro-batch * 4 replicas * 128 gradient accumulation). The model is optimized for 100 epochs with SGD+Momentum, with a momentum value of 0.9, and L2 regularization of value 0.0001. A cosine learning rate is also used for entire duration of training, and the learnig rate is warmed-up for the first 5 epochs of training for additional training stability.

For ease of use, the entire instruction is implemented in the default configuration file (`configs.yml`), named `resnet50_16ipus_8k_bn_pipeline` and can be easily reproduced with the following command:    
    `python3 train.py --config resnet50_16ipus_8k_bn_pipeline`


## Optional arguments

### General configuration

  -h, --help            
                        Show this help message and exit  
  
  --config CONFIG_NAME  
                        Select from available configurations  

  --config_path CONFIG_PATH  
                        Path to the configuration file  
  
  --on-demand ON_DEMAND  
                        Defer connection to when IPUs are needed

  --seed SEED  
                        Global seed for the random number generators (default: None)

### Checkpoints

  --checkpoints CHECKPOINTS  
                        Save weights to files at each callback

  --checkpoint-dir CHECKPOINT_DIR  
                        Path to save checkpoints (default: /tmp/checkpoints_current_time/)

  --ckpt-all-instances CKPT_ALL_INSTANCES  
                        Wether to have all instances saving the model. By default only 1 instances per host generates checkpoints.  

  --clean-dir CLEAN_DIR  
                        Delete checkpoint directory after validation (default: True)

### Dataset & model

  --dataset DATASET  
                        Name of dataset to use (default: cifar10)    
                        
  --dataset-path DATASET_PATH  
                        Path to dataset (default: .)
  
  --model-name MODEL_NAME  
                        Name of model to use (default: toy_model)
  
  --eight-bit-transfer EIGHT_BIT_TRANSFER
                        Enable input transfer in 8 bit (default: False)  

  --synthetic-data SYNTHETIC_DATA  
                        Enable the use of synthetic data. Options are `host` or `ipu`. By default the feature is disabled.

  --pipeline-num-parallel PIPELINE_NUM_PARALLEL  
                        Number of images to process in parallel on the host side. (default: 48)
  
  ### Training parameters

  --training TRAINING  
                        Enables training (default: True)

  --micro-batch-size MICRO_BATCH_SIZE  
                        Micro batch size, in number of samples (default: 1)
  
  --num-epochs NUM_EPOCHS  
                        Number of training epochs (default: 1)  
  
  --logs-per-epoch --logs_per_epoch  
                        Logging frequency, per epoch (default: 1)  

  --weight-updates-per-epoch WEIGHT_UPDATES_PER_EPOCH  
                        number of weight updates per run on the device for one epoch
                        (default: dataset_size / global_batch_size)  
  
  --num-replicas NUM_REPLICAS  
                        Number of training replicas (default: 1)  
  
  --gradient-accumulation-count GRADIENT_ACCUMULATION_COUNT  
                        Number of gradients accumulated by each replica.   
                        When pipelining a model, `gradient_accumulation-count` has to be a multiple of twice the number of pipeline stages. 
                        (default: None)  

  --global-batch-size GLOBAL_BATCH_SIZE  
                        Global batch size, in number of samples.
                        Instead of defining the number of gradients to accumulate, alternatively the global batch size can be defined. The number of gradients to accumulate can be inferred from this information. This is useful for applying the same configuration to a different number of IPUs while maintaining the global batch size. (default: None)  
  
  --precision {16.16, 16.32, 32.32}  
                        (compute).(weight update) precision, both can be either 16 or 32 (default: 16.16)

  --pipeline-splits [PIPELINE_SPLITS]
                        Model layers that define the start of a new pipeline
                        stage. E.g. conv2d_1 conv2d_2 (default: [])  
  
  --device-mapping [DEVICE_MAPPING]  
                        List mapping pipeline stages to IPU numbers.  
                        E.g. 0 1 1 0 (default: None)  
  
  --pipeline-schedule {Grouped, Interleaved, Sequential}  
                        Pipelining schedule. Choose between 'Interleaved',
                        'Grouped' and 'Sequential'. (default: Grouped)

  --optimizer {sgd, lars}  
                        Name of optimizer to use (default: sgd)

  --optimizer-params OPTIMIZER_PARAMS  
                        Parameters for configuring the optimizer. To pass from the terminal use the following format: \'{"arg1": value1, "arg2": value2...}\'.
                        (default: \'{"momentum": 0}\')

  --loss-scaling LOSS_SCALING  
                        Value of static loss scaling. Loss scaling is not applied when == 0 (default: 0)

  --weight_decay WEIGHT_DECAY  
                        The optimizer weight decay value. (default: 1)

  --l2-regularization L2_REGULARIZATION
                        The optimizer L2 regularization value (default: 0) 

  --recomputation RECOMPUTATION  
                        Enable/disable recomputation of activations in the backward pass.   
                        Recomputation can be enabled only when `--pipeline-splits` and `--device-mapping` were set properly. (default: False)

  --accelerator-side-preprocess ACCELERATOR_SIDE_PREPROCESSING  
                        Moves some dataset preprocessing steps to the accelerator rather
                        than the host. Useful when the application is host/io bounded. (default: False)

  --accelerator-side-reduction ACCELERATOR_SIDE_REDUCTION  
                        Requires distributed training. When enabled, the reduction over replicas for logging is performed on the device rather than the host. (default: False)

  --stochastic-rounding STOCHASTIC_ROUNDING  
                        Enable stochastic rounding (default: False)


  --optimizer-state-offloading OPTIMIZER_STATE_OFFLOADING  
                        Enable offloading optimizer state to the IPU remote memory (default: True)

  --fp-exceptions FP_EXCEPTIONS  
                        Enable floating point exceptions (default: False)

  --lr-schedule {const, cosine, stepped, polynomial}  
                        Name of learning rate scheduler (default: 'const')
  
  --lr-warmup-params LR_WARMUP_PARAMS  
                        A dictionary of parameters to be used to configure the warm-up for learning rate scheduler. To pass this argument from the terminal use --lr-schedule-params \'{"warmup_mode": mode, "warmup_epochs": epochs}\' format. Modes available: {shift, mask}. (default: None)

  --lr-schedule-params LR_SCHEDULE_PARAMS  
                        A dictionary of parameters to be used to configure the learning rate scheduler. Different parameters are expected depending on the chosen lr scheduler in --lr-schedule. To pass from the terminal use the following format: \'{"arg1": value1, "arg2": value2...}\'. (default: \'{"initial_learning_rate": 0.0001}\')

  --lr-staircase LR_STAIRCASE  
                        Make learning rate values constant throughout the epoch. Applies to the chosen learning rate scheduler. (default: False)

  --dbn-replica-group-size DBN_REPLICA_GROUP_SIZE  
                        Distributed Batch Norm (DBN) option specifies how many replicas to aggregate the batch statistics across. DBN is disabled when ==1. It can be enabled only if model fits on a single ipu (num ipus per replica ==1), model is replicated (num replicas > 1) and replication factor is divisible by dbn replica group size. (default: 1)

  --label-smoothing LABEL_SMOOTHING  
                        Add smoothing factor to each zero label (default: None)

  --norm-layer NORM_LAYER  
                        Type of normalisation layer to use. When using group norm specify either num_groups or channels_per_group. '
                        When using custom batch norm specify momentum.

  --fused-preprocessing  FUSED_PREPROCESSING  
                        Use fused operations for preprocessing images on device.

### Poplar optimizations

  --half-partials HALF_PARTIALS  
                        Accumulate matmul and convolution partial results in half precision (default: False) 

  --internal-exchange-optimization-target {cycles, memory, balanced}  
                        Set poplar internal exchange optimization target (default: None)

  --max-cross-replica-buffer-size MAX_CROSS_REPLICA_BUFFER_SIZE  
                        The maximum number of bytes that can be waiting before a cross replica sum op is scheduled. 0 (default) means that they are scheduled immediately.

  --max-reduce-many-buffer-size MAX_REDUCE_MANY_BUFFER_SIZE  
                        The maximum size (in bytes) a cluster of reduce operations can reach before it is scheduled. These clusters are lowered to popops ReduceMany operations. (default: 0)

  --gather-conv-output GATHER_CONV_OUTPUT  
                        Reduce sync cost of small sized all-reduces. Useful when paired with distributed batch norm. (default: False)

  --stable-norm STABLE_NORM  
                        Enable/disable numerically more stable but less parallelizable normalization layers. (default: False)

  --available-memory-proportion [AVAILABLE_MEMORY_PROPORTION]  
                        The percentage of IPU memory dedicated to convolutions and matrix multiplications. (default: [])  

### Evaluation & logging

  --wandb WANDB  
                        Enable/disable logging to Weights & Biases (default: False)  

  --wandb-params WANDB_PARAMS
                        Parameters for configuring wandb. To pass from the terminal use the following format: \'{"arg1": value1, "arg2": value2...}\'.

  --validation VALIDATION  
                        Enables validation (default: True)

  --validation-micro-batch-size VALIDATION_MICRO_BATCH_SIZE  
                        Validation micro batch size, in number of samples. None (default) means that --micro-batch-size is used. 

  --validation-num-replicas VALIDATION_NUM_REPLICAS  
                        Number of validation replicas. None (default) means that the value will be inferred from the total number of IPUs used during training.

  --pipeline-validation-model PIPELINE_VALIDATION_MODEL  
                        Reuse the training pipeline splits for validation. (default: False)  


## Exporting trained model in SavedModel format for TensorFlow Serving

Following the training and validation runs, `export_for_serving.py` script can be used to export a model to SavedModel format, which can be subsequently deployed to TensorFlow Serving instances and thus made available for inference. The model should be defined using the same options or configuration file that have been provided to `train.py` for training.
The following command line creates a SavedModel containing Resnet50 with weights initialized from `checkpoint.h5` file:
    `python3 scripts/export_for_serving.py --config resnet50_16ipus_8k_bn_pipeline  --export-dir="./resnet50_for_serving/001" --micro-batch-size=1  --checkpoint-file=checkpoint.h5 --pipeline-serving-model=False --iterations=128`

Please keep in mind that the exported SavedModel can't be used to load the model back into a TensorFlow script, as it only contains the IPU runtime op and an opaque executable and no model state.

### Additional arguments for export_for_serving.py script
  --export-dir EXPORT_DIR  
                        Path to the directory where the SavedModel will be written. (default: None)
                        
  --iterations ITERATIONS  
                        Number of iterations for the model exported for TensorFlow Serving. (default: 1)
                        
  --pipeline-serving-model PIPELINE_SERVING_MODEL  
                        Reuse the training pipeline splits for inference in the model exported for TensorFlow Serving (default: False)
                        
  --checkpoint-file CHECKPOINT_FILE  
                        Path to a checkpoint file that will be loaded before exporting model for TensorFlow Serving. If not set, the model will use randomly initialized parameters. (default: None)


### TensorFlow Serving example

Example of TensorFlow Serving usage can be found in `send_request.py` file. Script exports selected model to SavedModel format, initializes serving server, and sends images to server for predictions. Execution ends after given number of prediction requests. The model should be defined using the same options or configuration file that have been provided to `train.py` for training.

Basic usage example for Resnet50 model export in batch-size 16 and serving in batch-size 8: 
    `python3 send_request.py --config resnet50_infer_bs16 --dataset-path $DATASETS_DIR/imagenet-data --batch-size 8 --port 8502 --num-threads 32`

### Additional arguments for send_request.py script

   --port PORT  
                        Serving service acces port
                        
   --batch-size BATCH_SIZE 
                        Size of data batch used in single prediction request. Might be smaller than global-batch-size of exported model
                        
   --num-threads NUM_THREADS  
                        Number of threads/processes used for prediction requests, optimal value depends from used model and host system specification
                        
   --num-images NUM_IMAGES  
                        Number of image prediction requests
                        
   --serving-bin-path SERVING_BIN_PATH  
                        Path to TensorFlow serving binary file
                        
   --use-async  
                        When enabled client will send next prediction requests without blocking/waitg for server response. Each request returns `Future Object`
                        
   --verbose  
                        Enables printing of each request execution time. Expect degradation in overall performace caused by printing
                        

## Licensing
The code in this directory is provided under the MIT License (see the license file at the top-level of this repository) with the exception of the following files that are derived work and licensed under the Apache License 2.0.

- `data/imagenet_preprocessing.py` from
[TensorFlow Models](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/imagenet_preprocessing.py) 

- `data/build_imagenet_data.py` from
[AWS Labs](https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/build_imagenet_data.py)

See the header in each of those files for copyright details and link to the Apache License 2.0.
