# FastSpeech2 on IPUs using TensorFlow2

This directory provides a script and recipe to run FastSpeech2 models on Graphcore IPUs. The model is based on original paper [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) with few changes mentioned on [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/fastspeech2#whats-difference-).

## File Structure

| file/folder      | description                 |
| :--------------- | --------------------------- |
| config/         | Folder for model configs    |
| custom_op/             | Implement LengthRegular op          |
| preprocessor/    | Prepare datasets for training model     |
| tests/      | Unit tests.         |
| dataloader.py       | Loading and post-processing datasets |
| fastspeech2.py | FastSpeech2 model implementation     |
| optimizer.py          | Implement AdamW optimizer and warmup scheduler |
| options.py   | Commandline options for model configuration and IPU specific options|
| train.py          | Training process           |
| utils.py | Utils for callbacks, IPU configuration and other useful utils |
| ckpt_utils.py          | Utils for parsing and mapping saved `*.ckpt` or `*.h5`|

## Quick Guide

### 1. Configure Python virtual environment

#### 1) Download the Poplar SDK

[Download](https://downloads.graphcore.ai/) and install the Poplar SDK following the Getting Started guide for your IPU system. Source the `enable.sh` script for poplar.

#### 2) Configure Python virtual environment

Create a virtual environment and install the appropriate Graphcore TensorFlow 2 wheel from inside the SDK directory:

```shell
virtualenv --python python3 fs2_venv
source fs2_venv/bin/activate
pip install -r requirements.txt
pip install <path to the tensorflow-2 wheel from the Poplar SDK>
```

#### 3) Compile the custom op

We implement `LengthRegulator` custom op to avoid dynamic operations that we do not support. Make sure `MAX_TOTAL_SIZE` in `custom_op/length_regulator/poplar_code.cpp` is identical with `max_wave_length` in json file. Then run:

```shell
cd custom_op/length_regulator
make
```

Note that SDK needs to be sourced for the compilation to work.

### 2. Prepare dataset

#### 1) Download LJSpeech dataset

The data used for this example is LJSpeech, you can download the dataset at [here](https://keithito.com/LJ-Speech-Dataset/). The whole dataset tarball is about 2.6GB.

#### 2) Generate training dataset

- Unzip the datasets to specific folder.

```shell
mkdir -p /path/to/unzipped_dataset
tar -xvf /path/to/LJSpeech-1.1.tar.bz2 -C /path/to/unzipped_dataset
```

- Generate training and validation datasets

```shell
cd preprocessor
python3 audio.py \
--config ljspeech_preprocess.yaml \
--rootdir /path/to/unzipped_dataset \
--outdir /path/to/preproceed_dataset \
--n_cpus 4 \
--test_size 0.05 \
--seed 1234
```

It will take about 10 minutes to get the datasets. Increasing `n_cpus` to speed up the process. Then you should see the below files in the location set in `--outdir`:

| files/folders | description |
| :------------ | ----------- |
| vocab.txt | Vocabulary of LJSpeech dataset with characters, phonemes, punctuation per line. |
| ljspeech_mapper.json | Record the mapping between id and symbols. |
| length.json | Record maximium sequence length, maximium mel-spectrum length and vocab size of  the dataset |
| stats{,_f0, _energy}.npy | Contains the mean and std from the training split mel-spectrograms/f0/energy data. |
| {train,valid}_utt_ids.npy | Contains training/ validation utterances IDs respectively. |
| `train/` or `valid/` | <br>Contains pre-computed features of training/validation data.</br> <br>`ids/`: The ids of symbols in training data.</br> <br>`norm-feats/`: Mel-spectrum features with normalization. <br>`raw-feats/`: Mel-spectrum features. <br>`raw-f0/`: Pitch features.</br> <br>`raw-energies/`: Energy features.</br> <br>`wavs/`: Wave features.</br> |
The processed datasets is about 13GB.

#### 3) Duration dataset

You need to extract duration datasets for FastSpeech2. We used pre-trained duration alignment dataset provided by Anh Minh Nguyen Quan on [Google Drive](https://drive.google.com/drive/u/0/folders/1kaPXRdLg9gZrll9KtvH3-feOBMM8sn3_) instead of training from scratch.
However, the duration dataset was mismatched in  train/valid subsets. You need re-located them to corresponding folders according to utterance ids.

```shell
unzip /path/to/downloaded_duration -d /path/to/unzipped_dataset
cd /path/to/unzipped_dataset && mv train/* valid/* . && rm -r train/ valid/
python3 relocate_duration.py --root-path /path/to/preprocessed_dataset --duration-path /path/to/unzipped_dataset
```

After relocation, you will see `duration` folder under both `train/` and `valid/` directory.

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

### Train FastSpeech2 on IPU

Now that the data are ready we can start training our FastSpeech2 model on the IPU! Run this script:

```shell
python3 train.py --config config/fastspeech2.json \
--train \
--data-path /path/to/preprocessed_dataset \
--batch-size 2 \
--gradient-accumulation-count 8 \
--available-memory-proportion 0.1
```

All command line options can find in `options.py`. For example, you can specify `--wandb true` and `--wandb-name project_name` to use [**Weights&Biases**](https://github.com/wandb/client) to monitor training process. Using `--generated-data` to feed fake dataset if you don't have preprocessed dataset. Setting `--precision` to either `16` or `32` to train model with FP16 or FP32.
**[NOTE]**The option in configuration json file will be overided by command line options.

Then all profiles stay in `./profiles` folder and you can use our PopVision tool for deeper analysis.

### Inference FastSpeech2 on IPU

Once you have finished training and got the checkpoint, you can do inference by:

```shell
python3 infer.py --config config/fastspeech2.json \
--data-path /path/to/preprocessed_dataset \
--init-checkpoint /path/to/pretrained_checkpoint \
--epochs 100 \
--batch-size 1 \
--steps-per-epoch 10
```

If you don't have `init-checkpoint`, then the inference step will use random weights instead. Increasing the `steps-per-epoch` will give you better performance.

## Licensing

The code presented here is licensed under the Apache License Version 2.0, see the LICENSE file in this directory.

This directory includes derived work from the following:

TensorFlowTTS, https://github.com/TensorSpeech/TensorFlowTTS

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Reference

1. TensorFlowTTS: https://github.com/TensorSpeech/TensorFlowTTS
2. FastSpeech 2: https://arxiv.org/abs/2006.04558
