# PyTorch BERT

This directory contains an implementation of BERT models in PyTorch for the IPU, leveraging the HuggingFace Transformers library. There are two examples:

1. BERT for pre-training - `run_pretraining.py`
2. BERT for SQuAD - `run_squad.py`

## Environment setup

First, install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for Poplar and PopART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip3 install -r requirements.txt
make
```

## Run the pre-training application

Setup your environment as explained above and run the example with the configuration of your choice.

```console
python3 run_pretraining.py --config demo_tiny_128
```

## Configurations

To see the available configurations for both SQuAD and pre-training see the `configs.yml` file.

To see the available options available to use in the command line interface use the `--help` argument:

```console
python3 run_pretraining.py --help
# or
python3 run_squad.py --help
```

## Running pre-training with checkpointing

To enable the saving of model checkpoints on a run you need to add `--checkpoint-output-dir <path/to/checkpoint/dir>` to the command line. By default this will save a model checkpoint at the start and end of training.

Additionally, for more frequent outputting of checkpoints you can add `--checkpoint-steps <nsteps>` to save a model checkpoint after every `nsteps` training steps.

To load model weights from a checkpoint directory use the flag `--pretrained-checkpoint <path/to/checkpoint/step_N>`. (You can also pass the name of a model from HuggingFace model hub here too.) To also resume a training run from a checkpoint, also add the flag `--resume-training-from-checkpoint`.

## Run the SQuAD application

The question answering with SQuAD example is found in the `run_squad.py` script. Like with pre-training there are SQuAD configs defined in `configs.yml`.

To run BERT-Base:

```console
python3 run_squad.py --config squad_base_384
```

For BERT-Large there is `squad_large_384`, which is a high performance large configuration that uses an 8 IPU pipeline, unlike the other configs that use 4.

You will also need to specify a pre-trained checkpoint to fine-tune, which is specified with the `--pretrained-checkpoint <FILE-PATH/HF-model-hub-name>` flag.

## Caching executables

When running the application, it is possible to save/load executables to/from a cache store. This allows for reusing a saved executable instead of re-compiling the model when re-running identical model configurations. To enable saving/loading from the cache store, use `--executable-cache-dir <relative/path/to/cache/store>` when running the application.

## Running the entire pre-training and SQuAD pipeline

For Base on POD16:

```console
# Phase 1 pre-training
python3 run_pretraining.py --config pretrain_base_128 --checkpoint-output-dir checkpoints/pretrain_base_128

# Phase 2 pre-training
python3 run_pretraining.py --config pretrain_base_384 --checkpoint-output-dir checkpoints/pretrain_base_384 --pretrained-checkpoint checkpoints/pretrain_base_128/step_N/

# To do phase 2 pretraining with a sequence length of 512, simply replace `384` with `512`.

# SQuAD fine-tuning
python3 run_squad.py --config squad_base_384 --pretrained-checkpoint checkpoints/pretrain_base_384/step_N/
```

For Large on POD16:

```console
# Phase 1 pretraining
python3 run_pretraining.py --config pretrain_large_128 --checkpoint-output-dir checkpoints/pretrain_large_128

# Phase 2 pretraining
python3 run_pretraining.py --config pretrain_large_384 --checkpoint-output-dir checkpoints/pretrain_large_384 --pretrained-checkpoint checkpoints/pretrain_large_128/step_N/

# To do the same on POD64, simply append `_POD64` to the pretraining config names. To do phase 2 pretraining with a sequence length of 512, simply replace `384` with `512`.

# SQuAD fine-tuning
python3 run_squad.py --config squad_large_384 --pretrained-checkpoint checkpoints/pretrain_large_384/step_N/
```

To do the same on POD64, simply append `_POD64` to the pretraining config names.

## POD128 configurations

PopDist and PopRun allow to seamlessly launch applications on large IPU-POD systems such as POD128.  Further details about them can be found in the [docs]( https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/index.html).

We provide utility scripts to run the phase 1 and phase 2 pretraining in POD128. They can be executed as:

```console
# Phase 1 pretraining in POD128
bash training_scripts/pretrain_large_128_POD128.sh

# Phase 2 pretraining in POD128
bash training_scripts/pretrain_large_384_POD128.sh
```

The resulting pretraining checkpoint can be fine-tuned for SQuAD in a POD16 as described before.

## Run the tests (optional)

Setup your environment and generate the sample dataset as explained above and run `python3 -m pytest` from the root folder.

## Generate sample_text dataset (optional)

The sample text provided enables training on a very small dataset for small scale testing.
For convenience it is already provided in the `/data` folder in `txt` and `tfrecord` format.
In order to re-generate the sample dataset, run the following script:

```console
python3 third_party/create_pretraining_data.py --input-file data/sample_text.txt --output-file data/sample_text.tfrecord --sequence-length 128 --mask-tokens 20 --duplication-factor 4 --do-lower-case --model bert-base-uncased
```

## Generate pretraining dataset (optional)

The dataset used for pretraining is WIKI-103. It can be generated from a RAW dump of Wikipedia following a five step process.

### 1. Download

Use the `wikipedia_download.sh` script to download the latest Wikipedia dump, about 20GB in size.

```console
./data/wikipedia_download.sh <chosen-path-for-dump-file>
```

Dumps are available from <https://dumps.wikimedia.org/> (and mirrors) and are licensed under CC BY-SA 3.0 and GNU Free Documentation Licenses.

### 2. Extraction

In order to create the pre-training data we need to extract the Wikipedia dump and put it in this form:

```text
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

One of the tools that can be used to do so is WikiExtractor, <https://github.com/attardi/wikiextractor>.
Install the WikiExtractor package with `pip3 install wikiextractor`.

In order not to encounter a `UnicodeEncodeError` at this step, you may want to run these two commands first:

```console
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
```

You can then use the the `wikipedia_extract.sh` script to use WikiExtractor to extract the data dump.

```console
./data/wikipedia_extract.sh <chosen-path-for-dump-file>/wikidump.xml <chosen-folder-for-extracted-files>
```

The result should be a folder containing directories named `AA`, `AB`, ...
Note that the number of directories depends on the parameters of the `wikipedia_extract.sh` script, and is not to be confused with alphabetical ordering of the wikipedia articles.
In other words you should probably not expect all of `AC`, `AD`, ... `ZX`, `ZY`, `ZZ` to be created by the script.

### 3. Pre-processing

Install nltk package with `pip3 install nltk`.
Use the `wikipedia_preprocess.py` script to preprocess the extracted files.

```console
python3 ./data/wikipedia_preprocess.py --input-file-path <chosen-folder-for-extracted-files> --output-file-path <chosen-folder-for-preprocessed-files>
```

### 4. Tokenization

The script `create_pretraining_data.py` can accept a glob of input files to tokenize.
However, attempting to process them all at once may result in the process being killed by the OS for consuming too much memory.
It is therefore preferable to convert the files in groups. This is handled by the `./data/wikipedia_tokenize.py` script.
At the same time, it is worth bearing in mind that `create_pretraining_data.py` shuffles the training instances across the loaded group of files, so a larger group would result in better shuffling of the samples seen by BERT during pre-training.

The tokenization depends on `tensorflow` which can be installed by `pip3 install tensorflow`.

sequence length 128

```console
python3 ./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 128 --mask-tokens 20
```

sequence length 384

```console
python3 ./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 384 --mask-tokens 56
```

sequence length 512

```console
python3 ./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 512 --mask-tokens 76
```

### 5. Indexing

In order to use the multi-threaded `dataloader`, `tfrecord` index files need to be generated.
First install the `tfrecord` Python package into your Python environment:

```console
pip3 install tfrecord
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
