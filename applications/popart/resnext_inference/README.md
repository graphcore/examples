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

**1) Download the Poplar SDK **

  Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
  scripts for poplar, gc_drivers (if running on hardware) and popART.

##### 2) Python

Create a virtualenv and install the required packages:

```
bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

### Prepare the dataset

**1) Download the dataset**

Download the the COCO dataset (COCOval2014) from here: http://images.cocodataset.org/zips/val2014.zip)

**2) Partition the dataset** 

Assuming you are running on 8 IPUs, the COCO dataset should be partitioned into 8 distinct directories so each one can be read and streamed onto an IPU by a different process. To set the dataset partition up, make a directory to contain the data subsets, and then run `partition_dataset.py` as follows:

```
(popart) $ mkdir datasets
(popart) $ python partition_dataset.py --data-dir /localdata/datasets/coco/val2014 --partitions 8 --output datasets
```

Where `data-dir` is the location of the COCO dataset.

### Running the model

**1) Download the model**

To download the pretrained ResNext101 model from Cadene’s pretrained models repository (https://github.com/Cadene/pretrained-models.pytorch#resnext), convert it to ONNX format, and create a copy with chosen batch size (in this example a batch size of 6), run:

```
(popart) $ python get_model.py --batch-size 6
```

This will populate `models/resnext101_32x4d/`. Rerun this script to create additional ONNX protobufs for every batch size you intend to use.

**2) Run the model**

Now you can run the model on the partitioned data.

```
(popart) $ python resnext_inference_launch.py --batch_size 6
```

Logging data, including throughput, is written per subprocess in `logs/` and aggregated over all subprocesses in stdout. Outputs are fetched but not written to file. This can be changed in 'resnext101.py', where outputs are saved as `results`.


## Options

```
python resnext_inference_launch.py -help

       USAGE: resnext_inference_launch.py [flags]
flags:

resnext_inference_launch.py:
  --batch_size: Batch size (per device)
    (default: '6')
    (an integer)
  --batches_per_step: Number of batches to fetch on the host ready for streaming onto the device, reducing host IO
    (default: '1500')
    (an integer)
  --data_dir: Parent directory containing subdirectory dataset(s). Number of subdirs should equal num_ipus
    (default: 'datasets/')
  --iterations: Number of iterations to run if using synthetic data. Each iteration uses one `batches_per_step` x `batch_size` x `H` x `W` x `C` sized input tensor.
    (default: '1')
    (an integer)
  --model_name: model name. Used to locate ONNX protobuf in models/
    (default: 'resnext101_32x4d')
  --num_ipus: Number of IPUs to be used. One IPU runs one compute process.
    (default: '8')
    (an integer)
  --num_workers: Number of threads per dataloader
    (default: '12')
    (an integer)
  --[no]synthetic: Use synthetic data created on the IPU for inference
    (default: 'false')
```