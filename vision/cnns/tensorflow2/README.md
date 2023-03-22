# CNNs (TensorFlow2)
Deep CNN residual learning models for image recognition and classification, optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference |
|-----------|--------|-------|----------|-------|----------|-----------|
| TensorFlow 2 | Vision | CNNs | ImageNet LSVRC 2012, CIFAR-10/100, MNIST | Image recognition, Image classification | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required  | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required |

## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the ImageNet LSVRC 2012 dataset (See Dataset setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```


More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).

## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the TensorFlow 2 and IPU TensorFlow add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
For the CPU architecture you are running on

4. Navigate to this example's root directory

5. Install the Python requirements with:
```bash
pip3 install -r requirements.txt
```


More detailed instructions on setting up your TensorFlow 2 environment are available in the [TensorFlow 2 quick start guide](https://docs.graphcore.ai/projects/tensorflow2-quick-start).

## Dataset setup
### ImageNet LSVRC 2012
Download the ImageNet LSVRC 2012 dataset from [the source](http://image-net.org/download) or [via kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)

TODO: Add step on how to build into TFrecord format using scripts.

Disk space required: 144GB

```bash
.
├── labels.txt
├── train-0<xxxx>-of-01024
    .
    .
    .
├── validation-00<xxx>-of-00128
    .
    .
    .

0 directories, 1153 files
```

For other datasets, such as CIFAR-10, CIFAR-100 or MNIST, we use TFDS (TensorFlow Datasets) to load the data, and if it is not present on the disk, it will be automatically downloaded.


## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. The benchmarks are provided in the `benchmarks.yml` file in this example's root directory.

For example:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

## Custom training

### Executing training only or validation only

By default, the application will execute training and afterwards it will validate all model checkpoints produced during training. You can disable any of these two phases if necessary. To disable validation, just append `--validation False` to your instruction. This terminates the program immediately after training finishes. Similarly, if you want to disable training, just append `--training False` to your instruction. Note that by default, when training is disabled, validation will be done on a randomly initialized model. However, you can additionally specify a directory of checkpoints to input to validation, with `--checkpoint-input-dir your_previously_generated_directory`, to run validation on the previously trained model without training the model any further.

### PopDist and PopRun - distributed training on IPU-PODs

To get the most performance from our IPU-PODs, this application example now supports PopDist, Graphcore Poplar
distributed configuration library. For more information about PopDist and PopRun, see the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/).

A very simple example of using distributed training can be done with:
```bash
poprun --num-instances 16 --num-replicas 16 python3 train.py
```

Each instance sets an independent input data feed to the devices. The maximum number of instances is limited by
the number of replicas, in this case 16. A side effect from this is that when executing distributed workloads,
the program option `--num-replicas value` is ignored.

Note that the default behaviour when combining distributed training and pipelined model will not work. By default, the app takes a pipelined training model and assumes the model fits in a single IPU during training. This means that the number of replicas changes between training and validation. While this improves performance, it is incompatible with distributed training which expects a constant number of replicas and ipus per replica throughout the program execution. So avoid this limitation we provide a `--pipeline-validation-model` command line option which uses the same the pipeline splits in training and validation:
```bash
poprun --num-instance X --num-replicas Y --ipus-per-replica Z python3 train.py --pipeline-splits split1 ... splitN --pipeline-validation-model
```


## Other features

### Periodic host events
While the bulk of the work is done by the IPUs, there are a number of operations we may want to do on the host CPU throughout the training run, such as logging or saving a checkpoint of the model. How often we want to execute these events has an impact on the throughput of the application. This is because the application is built in a way that the program to be executed on the IPUs is wrapped in a loop such that multiple iterations of the program can be performed on the IPUs before handing back control to the host CPU. By minimizing the handover between the IPUs and the host CPU we can reduce the total execution time of the program. The number of iterations executed on the IPUs can be adjusted to match the period of the host events.

Note that depending on dataset size and batch configuration, not all frequency values are valid. More specifically, the switching from the IPU to the CPU must happen after a weight update has been completed. For example, a given dataset and batch configuration result in an odd number of weight updates per epoch, it is not possible to log the training metrics 2 times per epoch, because one of the events would happen in the middle of a weight update. To avoid these situations and to facilitate the user experience, the program automatically calculates the closest possible frequency that doesn't attempt to interrupt a weight update. However it is possible that the corresponding event is executed at a higher or lower frequency than originally requested.

Some examples of adjusting the frequency of logging the state of the training run over time:
```bash
python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1
```

Sets steps per execution such that it executes one entire epoch on the device before returning to the host, while printing one log per epoch.
```bash
python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 2
```

It's not possible to execute half an epoch on the device because it would correspond to a partial weight update. It will execute 1 epoch on the device and print 1 log per epoch. It logs a warning to the user highlighting the logging frequency can't be executed and will be adjusted.
```bash
python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 10
```

It's not possible to execute 1/10 of an epoch on the device. The closest is 1/11. So the generated program executes 1/11 of an epoch before returning to the host and therefore 11 logs per epoch. A warning is logged.
```bash
python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1/2
```

The generated program is going to execute 2 epochs on the device before returning to the host. Therefore the program is only repeated for 3 times.
```bash
python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1/3
```

The generated program is going to execute 3 epochs on the device before returning to the host. Therefore the program is only repeated for 2 times.
```bash
python3 train.py --model cifar_resnet8 --num-epochs 5 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 1/2
```

An exception is raised: "ValueError: It is not possible to log 2.0 epochs a time for 5 epochs".
```bash
python3 train.py --model cifar_resnet8 --num-epochs 6 --precision 16.16 --micro-batch-size 8 --validation False --half-partials True --gradient-accumulation 8 --logs-per-epoch 0
```

The entire training run is executed on the device and a warning message is logged so the user is notified.


### Synthetic data

It is possible to execute the training program without a real dataset and instead using synthetic data. This is done by using the `--synthetic-data` option. When using synthetic data, the data can be generated on the host CPU and transferred to the IPU or the data can be generated directly on the IPU avoiding the need for any data transfer between the CPU and the IPU. To generate synthetic data on the CPU use `--synthetic-data cpu`, and to generate synthetic data on the IPU use `--synthetic-data ipu`.


### Regularization
The application allows to apply the squared norm of the weights to the loss in two ways, which are `--weight-decay <lambda>` and `--l2-regularization <lambda>`. They are mathematically equivalent when applied to stateless optimisers (e.g. SGD without the momentum), however, for stateful optimisers they behave differently. `--l2-regularization` affects the optimiser state directly, whereas `--weight-decay` doesn't, i. e.:
- weight decay:  `delta_W = -lr*(O(dL/dW, s) + lambda*W)`
- L2 regularization:  `delta_W = -lr*(O(dL/dW + lambda*W, s)`

where `L` is the loss with respect to weights `W`, `O` is an optimizer with state `s` and `lambda` is the regularization coefficient.

### Automatic loss scaling (ALS)
ALS is a feature in the Poplar SDK which brings stability to training large models in half precision, especially when gradient accumulation and reduction across replicas also happen in half precision.

In the IPU version of Keras, ALS is an experimental feature supported for the SGD optimizer without momentum and constant learning rate. To enable ALS just add the flag:
```bash
python3 train.py --auto-loss-scaling
```

### Convergence optimized configurations

#### ImageNet - ResNet-50

The following configuration trains ResNet50 using 16 Mk2 IPUs. The model is split over 4 IPUs and executed according to a pipeline execution scheme. The model weights and computation are represented using 16 bit floating point numbers, and matmuls and convolutions partials are also computed in the same representation. With the above settings it is possible to fit on each IPU a micro batch size of 16 samples. Additionally the model is replicated 4 times (4 pipeline stages * 4 replicas = 16 IPUs), and gradients are accumulated 128 times, therefore each weight update corresponds to a global batch size of 8192 samples (16 micro-batch * 4 replicas * 128 gradient accumulation). The model is optimized for 100 epochs with SGD+Momentum, with a momentum value of 0.9, and L2 regularization of value 0.0001. A cosine learning rate is also used for entire duration of training, and the learnig rate is warmed-up for the first 5 epochs of training for additional training stability.

For ease of use, the entire instruction is implemented in the default configuration file (`configs.yml`), named `resnet50_16ipus_8k_bn_pipeline` and can be easily reproduced with the following command:
```bash
python3 train.py --config resnet50_16ipus_8k_bn_pipeline
```

### TensorFlow Serving example

Example of TensorFlow Serving usage can be found in `send_request.py` file. Script exports selected model to SavedModel format, initializes serving server, and sends images to server for predictions. Execution ends after given number of prediction requests. The model should be defined using the same options or configuration file that have been provided to `train.py` for training.

Basic usage example for Resnet50 model export in batch-size 16 and serving in batch-size 8:
```bash
python3 send_request.py --config resnet50_infer_bs16 --dataset-path $DATASETS_DIR/imagenet-data --batch-size 8 --port 8502 --num-threads 32
```

## License
The code in this directory is provided under the MIT License (see the license file at the top-level of this repository) with the exception of the following files that are derived work and licensed under the Apache License 2.0.

- `data/imagenet_preprocessing.py` from [TensorFlow Models](https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/imagenet_preprocessing.py)

- `data/build_imagenet_data.py` from [AWS Labs](https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/build_imagenet_data.py)

See the header in each of those files for copyright details and link to the Apache License 2.0.
