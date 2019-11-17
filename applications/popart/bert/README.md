# Graphcore benchmarks: BERT training

This readme dscribes how to run BERT models for NLP pre-training and training on IPUs.

## Overview

BERT (Bidirectional Encoder Representations for Transformers) is a deep learning model implemented in ONNX that is used for NLP. It requires pre-training with unsupervised learning on a large dataset such as Wikipedia. It is then trained on more specific tasks for fine tuning - Graphcore uses SQuAD (Stanford Question Answering Dataset), a Q&A dataset, for training BERT on IPUs. 

There are two BERT models:

- BERT Base – 12 layers (transformer blocks), 110 million parameters
- BERT Large – 24 layers (transformer blocks), 340 million parameters

## BERT models

BERT Large requires 14 IPUs for pre-training (Wikipedia) and 13 IPUs for training on the SQuAD dataset. There are 2 layers of the model per IPU and the other IPUs are used for embedding and projection. 

Similarly BERT Base requires 8 IPUs for pre-training (Wikipedia) and 7 IPUs for training on the SQuAD dataset (2 layers of the model per IPU, other IPUs used for embedding and projection). 

**NOTE**: IPUs can only be acquired in powers of two (2, 4, 8, 16). Unused IPUs will be unavailable for other tasks.

The BERT Large model has up to 340 million parameters and therefore it is split across a number of IPUs (pipeline parallel). For example a 14 IPU pipeline for pre-training a 24 layer model could be arranged as follows:
IPU0: Tied Token Embedding/Projection
IPU1: Position Embedding/Losses
IPU2: Transformer layer 1 and 2
IPU3: Transformer layer 3 and 4
...
IPU13: Transformer layer 23 and 24

PopART/Poplar may use knowledge of the physical IPU connectivity to decide how the virtual IPUs map to physical devices for maximum efficiency, for example using a ring topology.

## Datasets

SQuAD is a large reading comprehension dataset which contains 100,000+ question-answer pairs on 500+ articles. 

The wikipedia dataset contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the wikipedia dump file is updated all the time.

Instructions on how to download the Wikipedia and SQuAD datasets can be found in the `bert_data/README.md file`. At least 1TB of disk space will be required for full pre-training (two phases, phase 1 with sequence_length=128 and phase 2 with sequence_length=384) and the data should be stored on NVMe SSDs for maximum performance. 

If full pre-training is required (with the two phases with different sequence lengths) then data will need to be generated separately for the two phases: 

-  once with --sequence-length 128 and --mask-tokens 20
- once with --sequence-length 384 --mask-tokens 60

See the `bert_data/README.md file`  for more details on how to generate this data. 

## Running the models


The following files are provided for running the BERT benchmarks. 

| File            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `bert.py`       | Main training loop                                           |
| `bert_model.py` | BERT model definition                                        |
| `utils.py`      | Utility functions                                            |
| `bert_data/`    | Directory containing the data pipeline and training data generation <br /><br />- `dataset.py` - Dataloader and preprocessing. Loads binary files into Numpy arrays to be passed `popart.PyStepIO`, with shapes based on training options,  `--batches-per-step` & `--pipeline` <br /><br /> -`create_pretraining_data.py` - Script to generate binary files to be loaded from text data |
| `configs/`      | Directory containing JSON configuration files to be used by the `--config` argument. |
| `custom_ops/`   | Directory containing custom PopART operators. These are optimised parts of the graph that target Poplar/Poplibs operations directly.<br />  - `attention.cpp` - This operation is the fwd and grad implementation for multi-headed self-attention.<br/>  - `detach.cpp` - This operation is an identity with no grad implementation. This allows for the embedding dictionary to only be updated by its use in the projection.<br/>  -`embeddingGather.cpp` - This operation is a modification on the PopART Gather to ensure correct layout of the weights. |


## Quick start guide

### Prepare the environment

##### 1) Download the Poplar SDK

  Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
  scripts for poplar, gc_drivers (if running on hardware) and popART.

##### 2) Python

Create a virtualenv and install the required packages:

```bash
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

##### 3) Compile `custom_ops`

Run:

```bash
make
```

This should create `custom_ops.so`.

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
python bert_data/create_pretraining_data.py \
  --input-file bert_data/sample_text.txt \
  --output-file data/sample_text.bin \
  --vocab-file uncased_L-12_H-768_A-12/vocab.txt \
  --do-lower-case \
  --sequence-length 128 \
  --mask-tokens 20 \
  --duplication-factor 10
```

**NOTE:** `--input-file/--output-file` can take multiple arguments if you want to split your dataset between files.

When creating data for your own dataset, make sure the text has been preprocessed as specified at https://github.com/google-research/bert. This means with one sentence per line and documents delimited by empty lines.

### Run the training loop for pre-training (small sample)

For the sample text a configuration has been created -  `configs/demo.json`. It sets the following options:

```javascript
{
  # Two layers as our dataset does not need the capacity of the usual 12 Layer BERT Base
  "num_layers": 2,
  "no_dropout": true,
  "popart_dtype": "FLOAT16",
  "loss_scaling": 1.0,
  "stochastic_rounding": true,
  "custom_ops": [
    "attention",
    "projection"
  ],
  # The data generation should have created 118 samples. Therefore, we will do an epoch per session.run
  "batches_per_step": 118,
  "epochs": 150,
  # Here we specify the file we created in the previous step.
  "input_files": [
    "data/sample_text.bin"
  ]
  "shuffle": true,
  "no_validation": true
}
```

Run this config:

```bash
python bert.py --config configs/demo.json
```

This will compile the graph and run for 150 epochs. At end our model should have overfit to 100% test accuracy.

#### Pre-training results (small sample)

```
[...]
Iteration: 149 Epoch: 149/149 Loss (MLM NSP): 0.019 0.029 Accuracy (MLM NSP): 0.993 1.000 Learning Rate: 0.00080 Duration: 0.1593 s Throughput:  740.8 samples/s
```

##### View the pre-training results in Tensorboard

`requirements.txt` will install a standalone version of tensorboard. The program will log all training runs to `--log-dir`(`logs` by default). View them by running:

```bash
tensorboard --logdir logs
```

### Run the training loop for pre-training (Wikipedia)

For BERT Base, use the following command:

`python bert.py --config configs/pretrain_base.json`

For BERT Large, use the following command:

`python bert.py --config configs/pretrain_large.json`

#### Pre-training results (Wikipedia)

The expected result from phase 1 of pretraining (`pretrain_base.json`) is:

```
[...]
Iteration:  12000 Epoch:   2/2 Loss (MLM NSP): 1.847 0.067 Accuracy (MLM NSP): 0.633 0.975 Learning Rate: 0.00012
```

### Run the training loop with training data (SQuAD 1.1)

How to get the SQuAD 1.1 training dataset is described in `bert_data/README`.

You can then extract the weights and launch SQuAD fine tuning using one of the preset configurations. 

To run SQuAD with a BERT Base model:

`python bert.py --config configs/squad_base.json`

and for BERT Large:

`python bert.py --config configs/squad_large.json`

View the JSON files in configs for detailed parameters.

## Training options

`bert.py` has many different options. Run with `-h/--help` to view them. Any options used on the command line will overwrite those specified in the configuration file.

