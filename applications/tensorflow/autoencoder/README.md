# Graphcore benchmarks: autoencoder (training and inference)

This readme describes how to run a TensorFlow autoencoder model on an IPU for both training and inference and how to interpret the results. 
At least 10 GB free on disk for caching is required to run this example.

## Overview

Autoencoder models can be used to perform collaborative filtering in recommender systems in order to provide useful predictions, for example recommending films for Netflix users based on their (and other users) previous experiences. This autoencoder model shows significant improvement in results compared to previous models on a Netflix dataset. 
The model architecture is a deep autoencoder with 6 fully connected layers and a constrained decoder. Dense re-feeding is used for training to overcome the sparseness of data in collaborative filtering. It is implemented in TensorFlow with a model size of about 10 million parameters. This model is taken from the paper “Training Deep AutoEncoders for Collaborative Filtering” https://arxiv.org/pdf/1708.01715.pdf.

## Autoencoder model

Regularisation techniques such as dropout are used to prevent over-fitting. The model implementation also uses loss scaling and mixed precision. Stochastic rounding techniques on the IPU enable the use of low precision fp16.16 for optimisation. 

### Dataset

The autoencoder model uses the Netflix Prize training dataset, as shown below.
Please note that the dataset can be downloaded separately from this web page: [Netflix Prize Data Set](https://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a). 
If the path to training data file is not specified, random data generated on the host will be used by the scripts.

|      | Full | 3 months | 6 months | 1 year |
| ---- | ---- | -------- | -------- | ------ |
| **Training** | 12/99 - 11/05 | 09/05 - 11/05 | 06/05 - 11/05 | 06/04 - 05/05 |
| **Users** | 477,412 | 311,315 | 390,795 | 345,855 |
| **Ratings** | 98,074,901 | 13,675,402 | 29,179,009 | 41,451,832 |
|  |  |  |  |  |
| **Testing** | 12/05 | 12/05 | 12/05 | 06/05 |
| **Users** | 173,482 | 160,906 | 169,541 | 197,951 |
| **Rating** | 2,250,481 | 2,082,559 | 2,175,535 | 3,888,684 |

## File structure

| Autoencoder files           | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `README.md`                 | This file, describing how to run the model                   |
| `autoencoder_benchmark.py` | Script to perform a hyperparameter scan over learning rate values and benchmark training throughput across numerous IPU devices |
| `autoencoder_data.py`       | Autoencoder end-to-end data pipeline                         |
| `autoencoder_main.py`       | Main training and validation loop                            |
| `autoencoder_model.py`      | Autoencoder model definition                                 |
| `util.py`                   | Helper functions for running on the IPU                      |

## Quick start guide

1)	Prepare the TensorFlow environment. 
	Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
    Make sure to source the `enable.sh` scripts for Poplar.

2)	Run the training program on random data: 
```
python3 autoencoder_main.py
```

NOTE: If you have downloaded the dataset, you can use the --training-data-file and --validation-data-file options to specify a path to the data:
```
python3 autoencoder_main.py --training-data-file netflix_data/3m_train.txt --validation-data-file netflix_data/3m_valid.txt --size 128
```

## Examples of running the model

The autoencoder can be run with standard TensorFlow and Poplar on 1xIPU or 2xIPU. Different batch sizes and either mixed precision (fp16.32) or half precision (fp16.16) can be used to test the IPU. The original paper suggests using different network layer sizes for different datasets, as described below.

The following options are available:

`--size` defines neural network architecture options according to the original paper. Accepted values are 128, 256, or 512. Use 128 for the 3-month dataset, 256 for 6-month and 1-year, 512 for full. For size 256 and 1-year dataset specify `--batch-size 32 --validation-batch-size 32`. For size 512 also specify `--batch-size 8 --validation-batch-size 8`. 

`--batch-size` set the batch size of the training graph. 

**Note**: `--batch-size` and `--size` will be set automatically to the values recommended above if not specified manually, as long as the file names of the datasets obtained using `get_data.sh` are unchanged. Manually specifying these values overrides the automatic selection.

`--validation-batch-size` set the batch size of the validation graph. This is set equal to `--batch-size` if not specified manually.

`--precision` allows to select between full, mixed or half precision. Accepted values are `32.32`, `16.32`, or `16.16`.

`--multiprocessing` is recommended when running both training and validation. It will run the training and validation concurrently in separate processes.

`--base-learning-rate` specifies the exponent of the base learning rate. The learning rate is set by lr = blr * batch-size. See https://arxiv.org/abs/1804.07612 for more details.

`--no-prng` disables stochastic rounding.

`--help` shows the available options.

Some examples of running the model on real data:

```
python3 autoencoder_main.py --training-data-file netflix_data/3m_train.txt --validation-data-file netflix_data/3m_valid.txt --size 128
```
```
python3 autoencoder_main.py --training-data-file netflix_data/6m_train.txt --validation-data-file netflix_data/6m_valid.txt --size 256
```
```
python3 autoencoder_main.py --training-data-file netflix_data/1y_train.txt --validation-data-file netflix_data/1y_valid.txt --size 256 --batch-size 32 --validation-batch-size 32
```
```
python3 autoencoder_main.py --training-data-file netflix_data/full_train.txt --validation-data-file netflix_data/full_valid.txt --size 512 --batch-size 8 --validation-batch-size 8
```

### Benchmarking

The script `autoencoder_benchmark.py` allows for performing a hyperparameter search over different learning rate values across multiple IPU devices in parallel in order to optimise accuracy and loss, benchmarking the aggregate throughput (users/second).

Usage is similar to the main autoencoder script, with added options:

`--num-ipus` sets the number of IPU devices (default is 16, a full chassis).

`--base-learning-rates` is a string comprised of the base learning rates, noting that there must be as many learning rates as IPUs requested. This can be specified either as comma-separated values (e.g. '16,17,18') or as a range (e.g. '16..20'). The base learning rates are the moduli, due to parsing constraints, and the respective signs will be flipped.

For example to benchmark throughput on 1 IPU device over the 3-month Netflix training data, with a base learning rate of 2^-16 (batch and first layer sizes set automatically):

```
python3 autoencoder_benchmark.py --training-data-file netflix_data/3m_train.txt --num-ipus 1 --base-learning-rates '16'
```

To benchmark on 2 IPU devices with base learning rates 2^{-16, -17}:

```
python3 autoencoder_benchmark.py --training-data-file netflix_data/3m_train.txt --num-ipus 2 --base-learning-rates '16,17'
```

To benchmark across 16 devices with default learning rates of 2^{-12.5, ..., -20} (varying in steps of 0.5):

```
python3 autoencoder_benchmark.py --training-data-file netflix_data/3m_train.txt
```

