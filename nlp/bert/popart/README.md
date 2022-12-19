# Graphcore benchmarks: BERT training

This README describes how to run BERT models for NLP pre-training and training on IPUs.

## Overview

BERT (Bidirectional Encoder Representations for Transformers) is a deep learning model implemented in ONNX that is used for NLP. It requires pre-training with unsupervised learning on a large dataset such as Wikipedia. It is then trained on more specific tasks for fine tuning - Graphcore uses SQuAD (Stanford Question Answering Dataset), a Q&A dataset, for training BERT on IPUs.

## BERT models

There are two BERT models:

- BERT Base – 12 layers (transformer blocks), 110 million parameters
- BERT Large – 24 layers (transformer blocks), 340 million parameters

The JSON configuration files provided in the `configs` directory define how the layers
are distributed across IPUs for these BERT models for training and inference.

## Datasets

SQuAD is a large reading comprehension dataset which contains 100,000+ question-answer pairs on 500+ articles.

The Wikipedia dataset contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the Wikipedia dump file is updated all the time.

Instructions on how to download the Wikipedia and SQuAD datasets can be found in the `bert_data/README.md file`. At least 1TB of disk space will be required for full pre-training (two phases, phase 1 with sequence_length=128 and phase 2 with sequence_length=512) and the data should be stored on NVMe SSDs for maximum performance.

If full pre-training is required (with the two phases with different sequence lengths) then data will need to be generated separately for the two phases:

- once with --sequence-length 128 --mask-tokens 20 --duplication-factor 6
- once with --sequence-length 512 --mask-tokens 76 --duplication-factor 6

See the `bert_data/README.md file`  for more details on how to generate this data.

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

## Running the models


The following files are provided for running the BERT benchmarks.

| File            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `bert.py`       | Main training loop                                           |
| `bert_model.py` | BERT model definition                                        |
| `utils/`        | Utility functions                                            |
| `bert_data/`    | Directory containing the data pipeline and training data generation <br /><br />- `dataset.py` - Dataloader and preprocessing. Loads binary files into Numpy arrays to be passed `popart.PyStepIO`, with shapes to match the configuration <br /><br /> -`create_pretraining_data.py` - Script to generate binary files to be loaded from text data |
| `configs/`      | Directory containing JSON configuration files to be used by the `--config` argument. |
| `custom_ops/`   | Directory containing custom PopART operators. These are optimised parts of the graph that target Poplar/PopLibs operations directly. |


## Quick start guide

### Prepare the environment

##### 1) Install the Poplar SDK

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for Poplar and PopART.

##### 2) Python

Create a virtualenv and install the required packages:

```bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

### Generate pre-training data (small sample)

As an example we will create data from a small sample: `bert_data/sample_text.txt`, however the steps are the same for a large corpus of text. As described above, see `bert_data/README.md` for instructions on how to generate data for the Wikipedia and SQuAD datasets.

##### Download the vocab file

You can download a vocab from the pre-trained model checkpoints at https://github.com/google-research/bert. For this example we are using `Bert-Base, uncased`.

##### Creating the data

Create a directory to keep the data.

```bash
mkdir data
```

`bert_data/create_pretraining_data.py` has a few options that can be viewed by running with `-h/--help`.

Data for the sample text is created by running:

```bash
python3 bert_data/create_pretraining_data.py \
  --input-file bert_data/sample_text.txt \
  --output-file data/sample_text.bin \
  --vocab-file data/ckpts/uncased_L-12_H-768_A-12/vocab.txt \
  --do-lower-case \
  --sequence-length 128 \
  --mask-tokens 20 \
  --duplication-factor 6
```

**NOTE:** `--input-file/--output-file` can take multiple arguments if you want to split your dataset between files.

When creating data for your own dataset, make sure the text has been preprocessed as specified at https://github.com/google-research/bert. This means with one sentence per line and documents delimited by empty lines.

### Quick-Start SQuAD Data Setup

The supplied configs for SQuAD assume data has been set up in advance. In particular these are:

- `data/ckpts/uncased_L-12_H-768_A-12/vocab.txt`: The vocabularly used in pre-training (included in the Google Checkpoint)
- `data/squad/train-v1.1.json`: The training dataset for SQuAD v1.1
- `data/squad/dev-v1.1.json`: The evaluation dataset for SQuAD v1.1
- `data/squad/evaluate-v1.1.py`: The official SQuAD v1.1 evaluation script
- `data/squad/results`: The results output path. This is created automatically by the `bert.py` script.

A full guide for setting up the data is given in the [bert_data](bert_data) directory. The following serves as a quick-start guide to run the Bert Base SQuAD
fine-tuning configurations with a pre-trained checkpoint.

#### Pre-Trained Checkpoint

Download pre-trained Base checkpoint containing the vocabulary from https://github.com/google-research/bert

```bash
$ cd <examples_repo>/nlp/bert/popart/
$ curl --create-dirs -L https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -o data/ckpts/uncased_L-12_H-768_A-12.zip
$ unzip data/ckpts/uncased_L-12_H-768_A-12.zip -d data/ckpts/uncased_L-12_H-768_A-12
```

#### SQuAD 1.1 Dataset and Evaluation Script

Download the SQuAD dataset (from https://github.com/rajpurkar/SQuAD-explorer):

```bash
$ cd <examples_repo>/nlp/bert/popart/
$ curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/squad/dev-v1.1.json
$ curl --create-dirs -L https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -o data/squad/train-v1.1.json
$ curl -L https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py -o data/squad/evaluate-v1.1.py
```

### Run the training loop for pre-training (small sample)

For the sample text a configuration has been created -  `configs/demo.json`. It sets the following options:

```javascript
{
  # Two layers as our dataset does not need the capacity of the usual 12 Layer BERT Base
  "num_layers": 2,
  "popart_dtype": "FLOAT16",
  # The data generation should have created 64 samples. Therefore, we will do an epoch per session.run
  "device_iterations": 64,
  "training_steps": 500,
  # Here we specify the file we created in the previous step.
  "input_files": [
    "data/sample_text.bin"
  ],
  "no_validation": true
}
```

Run this config:

```bash
python3 bert.py --config configs/demo.json
```

This will compile the graph and run for 500 training steps. At end our model should have overfit to 100% training accuracy.

##### View the pre-training results in Tensorboard

`requirements.txt` will install a standalone version of tensorboard. The program will log all training runs to `--log-dir`(`logs` by default). View them by running:

```bash
tensorboard --logdir logs
```

### Run the training loop for pre-training (Wikipedia)

For BERT Base phase 1, use the following command:

`python3 bert.py --config configs/pretrain_base_128.json`

For BERT Base phase 2, use the following command:

`python3 bert.py --config configs/pretrain_base_512.json`

You will also need to specify the option `--checkpoint_input_dir <path-to-checkpoint>` to load the weights from a previous training phase. You will find the checkpoint path for a training phase logged just after the compilation has completed in a date-time stamped directory. The checkpoints will be of the form `{checkpoint-output-dir}/{timestamp}/model_{epoch}.onnx`.

### Run the training loop with training data (SQuAD 1.1)

How to get the SQuAD 1.1 training dataset is described in `bert_data/README`.

You can then extract the weights and launch SQuAD fine tuning using one of the preset configurations.

To run SQuAD with a BERT Base model and sequence length of 512:

`python3 bert.py --config configs/squad_base_512.json`

and for BERT Large:

`python3 bert.py --config configs/squad_large_512.json`

View the JSON files in configs for detailed parameters.

By default, SQuAD finetuning will use the pre-trained weights downloaded alongside the vocab, but you can also specify an onnx checkpoint using the option `--checkpoint-input-dir <path-to-checkpoint>`.

## Training options

`bert.py` has many different options. Run with `-h/--help` to view them. Any options used on the command line will overwrite those specified in the configuration file.

## Inference

Before running inference you should run fine tuning or acquire fine-tuned weights in order to obtain accurate results.  Without fine-tuned weights the inference performance will be poor.

How to get the SQuAD 1.1 files required for inference is described in `bert_data/README`.

To run SQuAD BERT Base inference with a sequence length of 128:

`python3 bert.py --config configs/squad_base_128_inf.json`

and for BERT Large with a sequence length of 512:

`python3 bert.py --config configs/squad_large_512_inf.json`

View the JSON files in configs for detailed parameters.
