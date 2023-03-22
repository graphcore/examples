# GPS++: An Optimised Hybrid GNN/Transformer for Molecular Property Prediction

An optimised hybrid GNN/Transformer model for molecular property prediction using Graphcore IPUs, trained on the [PCQM4Mv2](https://arxiv.org/abs/2103.09430) dataset. The flexible hybrid model closely follows the [General, Powerful, Scalable (GPS) framework](https://arxiv.org/abs/2205.12454) and combines the benefits of both message passing and attention layers for graph-structured input data training.

## Running the model [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/3CGjC5E)

### Setup

This model is optimised for Graphcore IPUs and requires the Graphcore's Poplar SDK to run. You can access IPUs through [Paperspace](https://www.paperspace.com/graphcore), using the button above, or [G-Core](https://gcore.com/partners/graphcore).

Create a virtual environment and install the Poplar SDK, including the TensorFlow 2 wheels from inside the SDK directory:

```shell
virtualenv --python python3.6 .gps_venv
source .gps_venv/bin/activate
source <path to the Graphcore SDK>/enable
pip install <path to the TensorFlow-2 wheel from the Poplar SDK>
pip install --force-reinstall --no-deps <path to the Keras wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 2 from the Poplar SDK>
pip install -r requirements.txt
```

Our implementation includes a couple of things that need to be compiled prior to use, including an IPU optimised grouped gather-scatter custom op.

```shell
make -C data_utils/feature_generation
make -C static_ops
```

This project uses Weights & Biases to track experiments. If you don't have a Weights & Biases account, set the wandb mode to `offline`.

```shell
wandb offline
```

### Dataset

The [PCQM4Mv2](https://arxiv.org/abs/2103.09430) dataset is a recently published dataset for the OGB Large Scale Challenge built to aide the development of state-of-the-art machine learning models for molecular property prediction. The task is for the quantum chemistry task of predicting the [HOMO-LUMO energy gap](https://en.wikipedia.org/wiki/HOMO_and_LUMO) of a molecule.

The dataset consists of 3.7 million molecules defined by their SMILES strings which can simply be represented as a graph with nodes and edges.

The dataset includes four splits: train, valid, test-challenge and test-dev. The train and valid splits have true label and can be used for model development. The test-challenge split is used for the OGB-LSC PCQM4Mv2 challenge submission and test-dev for the [leaderboard](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2) submission.

At the start of training, the dataset will be downloaded and the additional features will be preprocessed automatically, including the 3D molecular features that are provided with the dataset.

We provide alternative dataset splits which will need to be downloaded and unpacked in this directory prior to running the application:

```bash
wget https://graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com/pcqm4mv2-cross_val_splits.tar.gz
tar xvzf pcqm4mv2-cross_val_splits.tar.gz
```

### Training and inference on OGB-LSC PCQM4Mv2

In order to begin training, select the configuration you wish to run from those in the `configs` directory.

Then run with the following command:

```shell
python3 run_training.py --config configs/<CONFIG_FILE>
```

After training has finished inference will follow and show the validation results on the validation dataset.

To run inference separately and on other dataset splits, use the following command, changing the `--inference_fold` flag:

```shell
python3 inference.py --config configs/<CONFIG_FILE> --checkpoint_path <CKPT_PATH> --inference_fold <DATA_SPLIT_NAME>
```

We also provide [training](notebook_training.ipynb) and [inference](notebook_inference.ipynb) notebooks.

## Performance

We have provided three configurations of our model of increasing size trained on OGB-LSC PCQM4Mv2 dataset:

| Model config | Parameters | No. layers | Train MAE | Valid MAE | Config file name | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPS++ 11M | 11M | 4 | ~0.075 | ~0.090 | GPS_PCQ_4gps_11M.yaml | [11M ckpt](https://graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com/GPS_PCQ_4gps_11M.tar.gz) |
| GPS++ 22M | 22M | 8 | ~0.056 | ~0.082 | GPS_PCQ_8gps_22M.yaml | [22M ckpt](https://graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com/GPS_PCQ_8gps_22M.tar.gz) |
| GPS++ | 44M |  16 | ~0.044 | ~0.077 | GPS_PCQ_16gps_44M.yaml | [gps++ ckpt1](https://graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com/GPS_PCQ_16gps_44M.tar.gz) |
| GPS++ trained on valid split | 44M |  16 | ~0.044 | NA | GPS_PCQ_16gps_44M.yaml | [gps++ ckpt2](https://graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com/GPS_PCQ_16gps_44M_inc_valid.tar.gz) |

## Our submission to OGB-LSC PCQM4Mv2

For the OGB-LSC PCQM4Mv2 challenge submission we trained an ensemble of the GPS++ (44M) model with six adjustments to the hyperparameters to form seven different model configurations.

Additionally, we trained the models on the training and validation data. The directory pcqm4mv2-cross_val_splits contains such split options and they can be used by modifying the flag `--split_mode`.

In total 112 models were ensembled and achieved an MAE of 0.0719 on the test-challenge set.

## Logging and visualisation in Weights & Biases

This project supports Weights & Biases, a platform to keep track of machine learning experiments. To enable this, use the `--wandb` flag.

The user will need to manually log in (see the quickstart guide [here](https://docs.wandb.ai/quickstart)) and configure these additional arguments:

- `--wandb_entity` (default `ogb-lsc-comp`)
- `--wandb_projecy` (default `PCQM4Mv2`)

For more information please see https://www.wandb.com/.

## Licensing

This application is licensed under the MIT license, see the LICENSE file at the base of the repository.

This directory includes derived work, these are documented in the NOTICE file at the base of the repository.

The checkpoints are licensed under the Creative Commons CC BY 4.0 license.
