# Graphcore benchmarks: ResNeXt inference

This readme describes how to run a ResNeXt101 model for bulk inference on IPUs.

## Overview

ResNeXt is a simple, highly modularized network architecture for image classification. The network is constructed by repeating a building block that aggregates a set of transformations with the same topology.

## ResNeXt101 model

The ResNeXt101 model is taken from the research paper Aggregated Residual Transformations for Deep Neural Networks: https://arxiv.org/abs/1611.05431. The simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, cardinality, which is the size of the set of transformations. 

## Datasets

This benchmark uses the COCO dataset (COCOval2014) which can be downloaded from here: http://images.cocodataset.org/zips/val2014.zip)

## Running the model

This application runs resnext101_32d.onnx for inference over 4 IPUs. Each IPU runs a single PopART inference session as a separate Python subprocess. In order to provide enough data to run the model inference for multiple host-to-device iterations, we have set this demo up so that each inference session processes the entire dataset. Each of these iterations fills the host buffers with `device-iterations`-many batches, which are pulled onto the IPU batch by batch. In a realistic bulk inference workload, you would instead partition the dataset between inference sessions.

The following files are provided for running the ResNeXt benchmark. 

| File                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `data.py`                     | Define a dataset class. Instances are iterated over to feed the model                                                             |
| `dataloader.py`               | Prepare and load the data into host buffer                                                             |
| `get_model.py`                | Download the pre-trained ResNeXt model                       |
| `setup_dataset.py`        | Copy the COCO dataset into distinct copies per inference process |
| `requirements.txt`            | Python requirements                                          |
| `resnext101.py`               | ResNeXt model definition                                     |
| `resnext_inference_launch.py` | Launch multiple inference sessions as child processes                                              |


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

Assuming you are running on 4 IPUs, we will make 4 copies of the COCO dataset so each one can be read and streamed onto an IPU by a different process. To do this, run `setup_dataset.py` as follows:

```
(popart) $ python setup_dataset.py --data-dir /path/to/coco/val2014 --copies 4 --output datasets
```

Where `data-dir` is the location of the COCO dataset and `output` is the location where copies are to be written.

#### Prepare the model

This application can run over multiple IPUs. It does this by creating multiple PopART `inference Session`s, each of which uses one ONNX model. The size of the data that is used by one model is called the `micro batch size`. The total batch size used will therefore be `micro-batch-size * num-ipus`.

**1) Download the model**

To download the pretrained ResNext101 model from Cadene’s pretrained models repository (https://github.com/Cadene/pretrained-models.pytorch#resnext), convert it to ONNX format, and create a copy with chosen micro batch size (in this example a micro batch size of 6), run:

```
(popart) $ python get_model.py --micro-batch-size 64
```

This will add the configured model to the directory `models/resnext101_32x4d/`.

### Running the model

Now you have set up the data and created the model you can run the model on the partitioned data. The entry point is `resnext_inference_launch.py`. This takes the batch size, which is the sum of all micro batch sizes over all the IPUs you are using.


```
(popart) $ python resnext_inference_launch.py --batch_size 256 --num_ipus 4
```

Logging data, including throughput, is written per subprocess in `logs/` and aggregated over all subprocesses in stdout. Inference results are fetched but not written to file. This can be changed in 'resnext101.py', where outputs are saved as `results`.

### Run unit-tests for the Deep Voice model

To run unit-tests, simply do:

```
pytest test_resnext101.py
```

## Options

```
python resnext_inference_launch.py -help

       USAGE: resnext_inference_launch.py [flags]
flags:

resnext_inference_launch.py:
  --batch_size: Overall size of batch (across all devices).
    (default: '256')
    (an integer)
  --data_dir: Parent directory containing subdirectory dataset(s). The number of sub directories should equal num_ipus
    (default: 'datasets/')
  --device_iterations: Number of batches to fetch on the host ready for streaming onto the device, reducing host IO
    (default: '200')
    (an integer)
  --[no]hide_output: If set to true the subprocess that the model is run with will hide output.
    (default: 'true')
  --iterations: Number of iterations to run if using synthetic data.
    (default: '1')
    (an integer)
  --log_path: If set, the logs will be saved to this specfic path, instead of logs/
  --model_name: model name. Used to locate ONNX protobuf in models/
    (default: 'resnext101_32x4d')
  --model_path: If set, the model will be read from this specfic path, instead of models/
  --num_ipus: Number of IPUs to be used. One IPU runs one compute process and processes a fraction of the batch of samples.
    (default: '4')
    (an integer)
  --num_workers: Number of threads per dataloader. There is one dataloader per IPU.
    (default: '2')
    (an integer)
  --[no]report_hw_cycle_count: Report the number of cycles a 'run' takes.
    (default: 'false')
  --[no]synthetic: Use synthetic data created on the IPU.
    (default: 'false')
```
