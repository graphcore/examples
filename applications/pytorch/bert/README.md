# PyTorch BERT

This directory contains an implementation of BERT models in PyTorch for the IPU, leveraging the HuggingFace Transformers library. There are two examples:

1. BERT for pretraining - `run_pretraining.py`
2. BERT for SQuAD - `run_squad.py`

## Environment setup

First, install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for Poplar and PopART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
make
```

## Run the pretraining application

Setup your environment as explained above and run the example with the configuration of your choice.

```console
python run_pretraining.py --config demo_tiny_128
```

## Configurations

To see the available configurations for both SQuAD and pretraining see the `configs.yml` file.

To see the available options available to use in the command line interface use the `--help` argument:

```console
python run_pretraining.py --help
# or
python run_squad.py --help
```

## Running pretraining with checkpointing

To enable the saving of model checkpoints on a run you need to add `--checkpoint-output-dir <path/to/checkpoint/dir>` to the command line. By default this will save a model checkpoint at the start and end of training.

Additionally, for more frequent outputting of checkpoints you can add `--checkpoint-steps <nsteps>` to save a model checkpoint after every `nsteps` training steps.

To load model weights from a checkpoint directory use the flag `--pretrained-checkpoint <path/to/checkpoint/step_N>`. (You can also pass the name of a model from HuggingFace model hub here too.) To also resume a training run from a checkpoint, also add the flag `--resume-training-from-checkpoint`.

## Run the SQuAD application

The question answering with SQuAD example is found in the `run_squad.py` script. Like with pretraining there are SQuAD configs defined in `configs.yml`. 

To run BERT-Base:
```console
python run_squad.py --config squad_base_384
```

For BERT-Large there is `squad_large_384`, which is a high performance large configuration that uses an 8 IPU pipeline, unlike the other configs that use 4.

You will also need to specify a pretrained checkpoint to fine-tune, which is specified with the `--pretrained-checkpoint <FILE-PATH/HF-model-hub-name>` flag.

## Caching executables

When running the application, it is possible to save/load executables to/from a cache store. This allows for reusing a saved executable instead of re-compiling the model when re-running identical model configurations. To enable saving/loading from the cache store, use `--executable-cache-dir <relative/path/to/cache/store>` when running the application.

## Running the entire pretraining and SQuAD pipeline

For Base on POD16:
```console
# Phase 1 pretraining
python run_pretraining.py --config pretrain_base_128 --checkpoint-output-dir checkpoints/pretrain_base_128

# Phase 2 pretraining
python run_pretraining.py --config pretrain_base_384 --checkpoint-output-dir checkpoints/pretrain_base_384 --pretrained-checkpoint checkpoints/pretrain_base_128/step_N/

# SQuAD fine-tuning
python run_squad.py --config squad_base_384 --pretrained-checkpoint checkpoints/pretrain_base_384/step_N/
```

For Large on POD16:
```console
# Phase 1 pretraining
python run_pretraining.py --config pretrain_large_128 --checkpoint-output-dir checkpoints/pretrain_large_128

# Phase 2 pretraining
python run_pretraining.py --config pretrain_large_384 --checkpoint-output-dir checkpoints/pretrain_large_384 --pretrained-checkpoint checkpoints/pretrain_large_128/step_N/

# SQuAD fine-tuning
python run_squad.py --config squad_large_384 --pretrained-checkpoint checkpoints/pretrain_large_384/step_N/
```

To do the same on POD64, simply append `_POD64` to the pretraining config names.

## Run the tests (optional)

Setup your environment and generate the sample dataset as explained above and run `python -m pytest` from the root folder.


## Generate sample_text dataset (optional)

The sample text provided enables training on a very small dataset for small scale testing.
For convenience it is already provided in the `/data` folder in txt and tfrecord format.
In order to re-generate the sample dataset, run the following script:

```console
python third_party/create_pretraining_data.py --input-file data/sample_text.txt --output-file data/sample_text.tfrecord --sequence-length 128 --mask-tokens 20 --duplication-factor 4 --do-lower-case --model bert-base-uncased
```

## Generate pretraining dataset (optional)

The dataset used for pretraining is WIKI-103. It can be generated from a RAW dump of Wikipedia following a four step process.

### 1. Download

Use the `wikipedia_download.sh` script to download the latest Wikipedia dump, about 20GB in size.

```console
./data/wikipedia_download.sh <chosen-path-for-dump-file>
```

Dumps are available from https://dumps.wikimedia.org/ (and mirrors) and are licensed under CC BY-SA 3.0 and GNU Free Documentation Licenses.

### 2. Extraction

In order to create the pre-training data we need to extract the Wikipedia dump and put it in this form:

```
<doc id = article1>
Title of article 1

Body of article 1

</doc>

<doc id = article2>
Title of article 2

Body of article 2
</doc>
```

and so on.

One of the tools that can be used to do so is WikiExtractor, https://github.com/attardi/wikiextractor.

You can use the the `wikipedia_extract.sh` script to use WikiExtractor to extract the data dump.

```console
./data/wikipedia_extract.sh <chosen-path-for-dump-file>/wikidump.xml <chosen-folder-for-extracted-files>
```

The result should be a folder containing directories named `AA`, `AB`...

### 3. Pre-processing

Install nltk package with `pip install nltk`.
Use the `wikipedia_preprocess.py` script to preprocess the extracted files.

```console
./data/wikipedia_preprocess.py --input-file-path <chosen-folder-for-extracted-files> --output-file-path <chosen-folder-for-preprocessed-files>
```

### 4. Tokenization

The script `create_pretraining_data.py` can accept a glob of input files to tokenise. However, attempting to process them all at once may result in the process being killed by the OS for consuming too much memory. It is therefore preferable to convert the files in groups. This is handled by the `./data/wikipedia_tokenize.py` script. At the same time, it is worth bearing in mind that `create_pretraining_data.py` shuffles the training instances across the loaded group of files, so a larger group would result in better shuffling of the samples seen by BERT during pre-training.

sequence length 128
```console
./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 128 --mask-tokens 20
```

sequence length 384
```console
./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 384 --mask-tokens 56
```

### Indexing

In order to use the multi-threaded dataloader, tfrecord index files need to be generated.
First install the `tfrecord` Python package into your Python environment:

```console
pip install tfrecord
```

Then go to the directory containing the pre-processed Wikipedia files and run:

```console
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
```

## Licensing

The code presented here is licensed under the Apache License Version 2.0, see the LICENSE file in this directory.

This directory includes derived work from the following:

BERT, https://github.com/google-research/bert

Copyright 2018 The Google AI Language Team Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
