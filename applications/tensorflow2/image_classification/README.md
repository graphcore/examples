# CNN Training on IPUs

This readme describes how to run CNN models such as ResNet for image recognition training on the IPU.

## Overview

Deep CNN residual learning models such as ResNet are used for image recognition and classification.
The training examples given below use models implemented in TensorFlow 2, optimised for Graphcore's IPU.

## Quick start guide

1. Prepare the TensorFlow environment. Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to run the `enable.sh` script for Poplar and activate a Python 3 virtualenv with
   the TensorFlow 2 wheel from the Poplar SDK installed.
2. Download the data. See below for details on obtaining the datasets.
3. Install the packages required by this application using (`pip install -r requirements.txt`)
4. Run the training script. You can check that the code runs with the simplest example:
   `python3 train.py`. See below for optimized instructions and options descriptions.

### Datasets

You can download the ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes,
from http://image-net.org/download. It is approximately 150GB for the training and validation sets.

For many other datasets, such as CIFAR-10, CIFAR-100 or MNIST datasets, we use TFDS to load the data, and if it is not present
on the disk, it will be automatically downloaded.

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
| `test/`                  | Tests that can be safely executed in parallel |
| `tests_serial/`          | Tests that should be executed one at the time |
| `batch_config.py`        | Class that infers correct number of micro and global batches based on parsed arguments such as gradient accumulation |
| `configs.yml`            | File where configurations are defined |
| `custom_exceptions.py`   | Custom exceptions with more descriptive names |
| `eight_bit_transfer.py`  | Code related to 8 bit io |
| `precision.py`           | Handles application of floating point precision policies |
| `requirements.txt`       | Project third-party dependencies |
| `seed.py`                | Allows to set a seed for random number generator for reproducibility of results |
| `test_common.py`         | Package that both test/ and tests_serial/ share |
| `time_to_train.py`       | Provides code for measuring TTT (Total-Time-to-Train) |
| `train.py`               | The main training program |
| `utilities.py`           | Shared methods used across the whole project |
| `verify_dataset.py`      | Script allowing to assess whether locally available datasets are of the shape expected by the dataset factory |

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

  --stochastic-rounding STOCHASTIC_ROUNDING  
                        Enable stochastic rounding (default: False)


  --optimizer-state-offloading OPTIMIZER_STATE_OFFLOADING  
                        Enable offloading optimizer state to the IPU remote memory (default: True)

  --fp-exceptions FP_EXCEPTIONS  
                        Enable floating point exceptions (default: False)

  --lr-schedule {const, cosine, stepped, polynomial}  
                        Name of learning rate scheduler (default: 'const')
  
  --lr-warmup-params LR_WARMUP_PARAMS  
                        A dictionary of parameters to be used to configure the warm-up for learning rate scheduler. To pass this argument from the terminal use --lr-schedule-params \'{"warmup_mode": mode, "warmup_epochs": epochs}\' format. Modes available: {{shift, mask}. (default: None)

  --lr-schedule-params LR_SCHEDULE_PARAMS  
                        A dictionary of parameters to be used to configure the learning rate scheduler. Different parameters are expected depending on the chosen lr scheduler in --lr-schedule. To pass from the terminal use the following format: \'{"arg1": value1, "arg2": value2...}\'. (default: \'{"initial_learning_rate": 0.0001}\')

  --lr-staircase LR_STAIRCASE  
                        Make learning rate values constant throughout the epoch. Applies to the chosen learning rate scheduler. (default: False)

  --dbn-replica-group-size DBN_REPLICA_GROUP_SIZE  
                        Distributed Batch Norm (DBN) option specifies how many replicas to aggregate the batch statistics across. DBN is disabled when ==1. It can be enabled only if model fits on a single ipu (num ipus per replica ==1), model is replicated (num replicas > 1) and replication factor is divisible by dbn replica group size. (default: 1)

  --bn-momentum BN_MOMENTUM  
                        Batch norm moving statistics momentum (default: 0.97)

  --label-smoothing LABEL_SMOOTHING  
                        Add smoothing factor to each zero label (default: None)

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
                        Enable/disable logging to Weights & Biases (default:
                        False)  

  --validation VALIDATION  
                        Enables validation (default: True)

  --validation-micro-batch-size VALIDATION_MICRO_BATCH_SIZE  
                        Validation micro batch size, in number of samples. None (default) means that --micro-batch-size is used. 

  --validation-num-replicas VALIDATION_NUM_REPLICAS  
                        Number of validation replicas. None (default) means that the value will be inferred from the total number of IPUs used during training.


## Configuration files

As can be observed from the above, the application provides a number of options to efficiently target execution on the IPUs. To facilitate building complex instructions, the user can instead write the instruction in a configuration file. The configuration is defined in YAML format. See `configs.yml` configuration file for examples of configuration instructions. After a configuration is specified you can use the configuration simply with:  
    `python3 train.py --config your_config --config-path your_config_file`

To facilitate experimentation, you can also overload specific parameters of a configuration directly from the command line:  
    `python3 train.py --config your_config --config-path your_config_file --some-param new_value`


## Examples

From the command line, to run resnet8 with cifar10 dataset, precison 16 bits for compute and weight update, batch size 32 on 1 IPU:   
    `python3 train.py --model cifar_resnet8 --wandb --num-epochs 11 --precision 16.16  --micro-batch-size 32 --dataset-path /localdata/datasets/`  

Recomputation can be enabled only when `--pipeline-splits` and `--device-mapping` were set properly. `gradient_accumulation-count` has to be adjusted to equal double the number of stages in pipelining:  
    `python3 train.py --model-name cifar_resnet8 --pipeline-splits conv2_block1_post_addition_relu conv3_block1_post_addition_relu conv4_block1_post_addition_relu --device-mapping 0 1 2 3 --gradient-accumulation-count 8 --recomputation True`


### Periodic host events
While the bulk of the work is done by the IPUs, there are a number of operations we may want to do on the host CPU throughout the training run, such as logging or saving a checkpoint of the model. How often we want to execute these events has an impact on the throughput of the application. This is because the application is built in a way that the program to be executed on the IPUs is wrapped in a loop such that multiple iterations of the program can be performed on the IPUs before handing back control to the host CPU. By minimizing the handover between the IPUs and the host CPU we can reduce the total execution time of the program. The number of iterations executed on the IPUs can be adjusted to match the period of the host events.  

Note that depending on dataset size and batch configuration, not all frequency values are valid. More specifically, the switching from the IPU to the CPU must happen after a weight update has been completed. For example, a given dataset and batch configuration result in an odd number of weight updates per epoch, it is not possible to log the training metrics 2 times per epoch, because one of the events would happen in the middle of a weight update. To avoid these situations and to facilitate the user experience, the program automatically calculates the closest possible frequency that doesn't attempt to interrupt a weight update. However it is possible that the corresponding event is executed at a higher or lower frequency than originally requested.

Some examples of adjusting the frequency of logging the state of the training run over time:  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1`  
Sets steps per execution such that it executes one entire epoch on the device before returning to the host, while printing one log per epoch.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 2`  
It's not possible to execute half an epoch on the device because it would correspond to a partial weight update. It will execute 1 epoch on the device and print 1 log per epoch. It logs a warning to the user highlighting the logging frequency can't be executed and will be adjusted.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 10`  
It's not possible to execute 1/10 of an epoch on the device. The closest is 1/11. So the generated program executes 1/11 of an epoch before returning to the host and therefore 11 logs per epoch. A warning is logged.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 0.5`  
The generated program is going to execute 2 epochs on the device before returning to the host. Therefore the program is only repeated for 3 times.  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 0.3333333333333333333`
The generated program is going to execute 3 epochs on the device before returning to the host. Therefore the program is only repeated for 2 times.  

`python3 train.py --model cifar_resnet8 --num-epochs 5 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 0.5`  
An exception is raised: "ValueError: It is not possible to log 2.0 epochs a time for 5 epochs".  

`python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 0`  
The entire training run is executed on the device and a warning message is logged so the user is notified.  

cifar10 on device preprocess and eight bit transfer applied via model editor:  
    `python3 train.py --accelerator-side-preprocess True --eight-bit-transfer True`

gc-imagenet on host preprocess (for faster results it's recommended to change split='train' to split='train[:1%]'):  
    `python3 train.py --accelerator-side-preprocess False --dataset gc-imagenet --dataset-path /localdata/datasets/imagenet-data --gradient-accumulation-count 8 --micro-batch-size 8`

## Pipeline splits 
A first runs with N pipeline splits (N>0) give a rough estimate of locations where the layers give equal amount of loading memory and computation for each stage. At the moment the estimation is only based on the number of weights per stage and will be refined in the future.


## Convergence optimized configurations

### ImageNet - ResNet-50

The following configuration trains ResNet50 using 16 Mk2 IPUs. The model is split over 4 IPUs and executed according to a pipeline execution scheme. The model weights and computation are represented using 16 bit floating point numbers, and matmuls and convolutions partials are also computed in the same representation. With the above settings it is possible to fit on each IPU a micro batch size of 16 samples. Additionally the model is replicated 4 times (4 pipeline stages * 4 replicas = 16 IPUs), and gradients are accumulated 128 times, therefore each weight update corresponds to a global batch size of 8192 samples (16 micro-batch * 4 replicas * 128 gradient accumulation). The model is optimized for 100 epochs with SGD+Momentum, with a momentum value of 0.9, and L2 regularization of value 0.0001. A cosine learning rate is also used for entire duration of training, however the learnig rate is warmed-up for the first 5 epochs of training for additional training stability.

For ease of use, the entire instruction is implemented in the default configuration file (`configs.yml`), named `mk2_resnet50_8k_bn_pipeline` and can be easily reproduced with the following command:    
    `python3 --config mk2_resnet50_8k_bn_pipeline`

## Additional sources 

This directory includes derived work with Apache License, Version 2.0 for the following file:  
| `data/imagenet_preprocessing.py` | extracted from
[Tensorflow repository](https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/imagenet_preprocessing.py)  

| `data/build_imagenet_data.py`  | originally from Google, extracted from
[AWS Labs](https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/build_imagenet_data.py)



