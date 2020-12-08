# Graphcore benchmarks: ResNeXt inference

This readme describes how to run a ResNeXt101 model for bulk inference on IPUs.

## Overview

ResNeXt is a simple, highly modularized network architecture for image classification. The network is constructed by repeating a building block that aggregates a set of transformations with the same topology.

## ResNeXt101 model

The ResNeXt101 model is taken from the research paper Aggregated Residual Transformations for Deep Neural Networks: https://arxiv.org/abs/1611.05431. The simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, cardinality, which is the size of the set of transformations. 

## Datasets

This benchmark uses the COCO dataset (COCOval2014) which can be downloaded from here: http://images.cocodataset.org/zips/val2014.zip)

## Running the model

This application runs resnext101_32d.onnx for inference over 8 IPUs. Each IPU runs a single PopART inference session, and receives one portion of the total dataset.

The following files are provided for running the ResNeXt benchmark. 

| File                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `data.py`                     | Define a dataset class. Instances are iterated over to feed the model                                                             |
| `dataloader.py`               | Prepare and load the data into host buffer                                                             |
| `get_model.py`                | Download the pre-trained ResNeXt model                       |
| `partition_dataset.py`        | Partition the COCO dataset into 8 directories for the 8 IPUs |
| `requirements.txt`            | Python requirements                                          |
| `resnext101.py`               | ResNeXt model definition                                     |
| `resnext_inference_launch.py` | Launch multiple inference sessions as child processes                                              |

## Quick start guide

### Prepare the environment

**1) Download the Poplar SDK**

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for Poplar and PopART.

**2) Python**

Create a virtualenv and install the required packages:

```
bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

### Download the data
Download the the COCO dataset (COCOval2014) from http://images.cocodataset.org/zips/val2014.zip and unzip.

### Prepare the dataset and model
The following script can be run to prepare the data and add the configured model to the directory `models/resnext101_32x4d/`.

```
(popart) $ ./model_prep_helper.sh path/to/coco/val2014 datasets
```
The first argument is the path to the COCO dataset, the second is the output directory for the processed datasets.

NOTE: If this script is run, the `Prepare the dataset` and `Prepare the model` sections below can be skipped.

#### Prepare the dataset

Assuming you are running on 8 IPUs, the COCO dataset should be partitioned into 8 distinct directories so each one can be read and streamed onto an IPU by a different process. To set the dataset partition up, make a directory to contain the data subsets, and then run `partition_dataset.py` as follows:

```
(popart) $ mkdir datasets
(popart) $ python partition_dataset.py --data-dir /localdata/datasets/coco/val2014 --partitions 8 --output datasets
```

Where `data-dir` is the location of the COCO dataset.

#### Prepare the model

This application can run over multiple IPUs. It does this by creating multiple PopART `inference Session`s, each of which uses one ONNX model and processes a fraction of the total batch of data. The size of the data that is used by one model is called the `micro batch size`. The total batch size used will therefore be `micro-batch-size * num-ipus`.

**1) Download the model**

To download the pretrained ResNext101 model from Cadene’s pretrained models repository (https://github.com/Cadene/pretrained-models.pytorch#resnext), convert it to ONNX format, and create a copy with chosen micro batch size (in this example a micro batch size of 6), run:

```
(popart) $ python get_model.py --micro-batch-size 6
```

This will add the configured model to the directory `models/resnext101_32x4d/`.

### Running the model

Now you have set up the data and created the model you can run the model on the partitioned data. The entry point is `resnext_inference_launch.py`. This takes the batch size, which is the sum of all micro batch sizes over all the IPUs you are using.


```
(popart) $ python resnext_inference_launch.py --batch_size 48 --num_ipus 8
```

Logging data, including throughput, is written per subprocess in `logs/` and aggregated over all subprocesses in stdout. Inference results are fetched but not written to file. This can be changed in 'resnext101.py', where outputs are saved as `results`.


## Options

```
python resnext_inference_launch.py -help

       USAGE: resnext_inference_launch.py [flags]
flags:

resnext_inference_launch.py:
  --batch_size: Overall size of batch (across all devices).
    (default: '48')
    (an integer)
  --batches_per_step: Number of batches to fetch on the host ready for streaming onto the device, reducing host IO
    (default: '1500')
    (an integer)
  --data_dir: Parent directory containing subdirectory dataset(s). The number of sub directories should equal num_ipus
    (default: 'datasets/')
  --iterations: Number of iterations to run if using synthetic data. Each iteration uses one `batches_per_step` x `batch_size` x `H` x `W` x `C` sized input tensor.
    (default: '1')
    (an integer)
  --model_name: model name. Used to locate ONNX protobuf in models/
    (default: 'resnext101_32x4d')
  --num_ipus: Number of IPUs to be used. One IPU runs one compute process and processes a fraction of the batch of samples.
    (default: '8')
    (an integer)
  --num_workers: Number of threads per dataloader. There is one dataloader per IPU.
    (default: '12')
    (an integer)
  --[no]synthetic: Use synthetic data created on the IPU.
    (default: 'false')
  --model-path: Directory containing the saved model required to run the model.
    (default: 'models/')
  --log-path: The directory where the logs will be saved.
    (default: 'logs/')
  --hide-output: If set the stdout of the subprocess that runs the model will be hidden.
    (default: 'false')
```
