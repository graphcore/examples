# PyTorch BERT

This directory contains an implementation of BERT models in PyTorch for the IPU, leveraging the HuggingFace Transformers library. There are two examples:

1. BERT for pre-training - `run_pretraining.py`
2. BERT for SQuAD - `run_squad.py`

Run our BERT-L Fine-tuning on SQuAD dataset on Paperspace.
<br>
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/gradient-ai/Graphcore-PyTorch?machine=Free-IPU-POD16&container=graphcore%2Fpytorch-jupyter%3A2.6.0-ubuntu-20.04-20220804&file=%2Fget-started%2FFine-tuning-BERT.ipynb)

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

To see the available configurations for both SQuAD and pre-training see the `configs_pretraining.yml` file.

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

The question answering with SQuAD example is found in the `run_squad.py` script. Like with pre-training there are SQuAD configs defined in `configs_squad.yml`.

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
python3 run_pretraining.py --config pretrain_base_512 --checkpoint-output-dir checkpoints/pretrain_base_512 --pretrained-checkpoint checkpoints/pretrain_base_128/step_N/

# To do phase 2 pretraining with a sequence length of 384, simply replace `512` with `384`.

# SQuAD fine-tuning
python3 run_squad.py --config squad_base_384 --pretrained-checkpoint checkpoints/pretrain_base_384/step_N/
```

For Large on POD16:

```console
# Phase 1 pretraining
python3 run_pretraining.py --config pretrain_large_128 --checkpoint-output-dir checkpoints/pretrain_large_128

# Phase 2 pretraining
python3 run_pretraining.py --config pretrain_large_512 --checkpoint-output-dir checkpoints/pretrain_large_512 --pretrained-checkpoint checkpoints/pretrain_large_128/step_N/

# To do the same on POD64, simply append `_POD64` to the pretraining config names. To do phase 2 pretraining with a sequence length of 384, simply replace `512` with `384`.

# SQuAD fine-tuning
python3 run_squad.py --config squad_large_384 --pretrained-checkpoint checkpoints/pretrain_large_384/step_N/
```

To do the same on POD64, simply append `_POD64` to the pretraining config names.

## Packed BERT

You can also enable sequence packing for even more efficient BERT pretraining. To enable, add the 
`--packed-data` flag to your command line when running any pretraining config and use `--input-files` to 
point to the packed version of the dataset. For instance:

```console
python3 run_pretraining.py --config demo_tiny_128 --packed-data --input-files data/packing/*.tfrecord
```

## Employing automatic loss scaling (ALS) for half precision training

ALS is an experimental feature in the Poplar SDK which brings stability to training large models in half precision, specially when gradient accumulation and reduction across replicas also happen in half precision. 

NB. This feature expects the `poptorch` training option `accumulationAndReplicationReductionType` to be set to `poptorch.ReductionType.Mean`, and for accumulation by the optimizer to be done in half precision (using `accum_type=torch.float16` when instantiating the optimizer), or else it may lead to unexpected behaviour.

To employ ALS for BERT Large pre-training on a POD16, the following command can be used:

```console
python3 run_pretraining.py --config pretrain_large_128_ALS --checkpoint-output-dir checkpoints/pretrain_large_128
```

To pre-train with ALS on a POD64:

```console
python3 run_pretraining.py --config pretrain_large_128_POD64_ALS --checkpoint-output-dir checkpoints/pretrain_large_128
```

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

### 6. Packing (optional)

Packing can lead to significant speed-ups during pretraining (details in https://arxiv.org/pdf/2107.02027.pdf). The packing scripts depend on `tensorflow` and `numpy` which can be installed by `pip3 install tensorflow numpy`. The following commands pack the 128, 384 and 512 sequence-length datasets with a maximum of 3 sequences per pack:

sequence length 128

```console
python3 -m data.packing.pack_pretraining_data --input-files=<path-of-unpacked-input-data-files> --output-dir=<path-of-output-packed-data-folder> --sequence-length 128 --mask-tokens 20
```

sequence length 384

```console
python3 -m data.packing.pack_pretraining_data --input-files=<path-of-unpacked-input-data-files> --output-dir=<path-of-output-packed-data-folder> --sequence-length 384 --mask-tokens 56
```

sequence length 512

```console
python3 -m data.packing.pack_pretraining_data --input-files=<path-of-unpacked-input-data-files> --output-dir=<path-of-output-packed-data-folder> --sequence-length 512 --mask-tokens 76
```

After packing it is recommended to shuffle again the dataset.

```console
python3 -m data.packing.shuffle_packed_data --input-files=<path-of-unshuffled-packed-data-files> --output-dir=<path-of-output-shuffled-packed-data-folder>
```

Remember to index the tfrecord files as in 5) after packing and shuffling. The following bash scripts run the full packing preprocess (packing plus shuffling) — the whole process for each sequence length takes approximately 6 hours:

sequence length 128

```console
./data/packing/pack_128.sh <path-of-input-unpacked-data-folder> <path-of-output-packed-data-folder>
```

sequence length 384

```console
./data/packing/pack_384.sh <path-of-input-unpacked-data-folder> <path-of-output-packed-data-folder>
```

sequence length 512

```console
./data/packing/pack_512.sh <path-of-input-unpacked-data-folder> <path-of-output-packed-data-folder>
```

## Benchmarking

To reproduce the benchmarks, please follow the setup instructions in this README to setup the environment, and then from this dir, use the `examples_utils` module to run one or more benchmarks. For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml
```

or to run a specific benchmark in the `benchmarks.yml` file provided:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark <benchmark_name>
```

For more information on how to use the examples_utils benchmark functionality, please see the <a>benchmarking readme<a href=<https://github.com/graphcore/examples-utils/tree/master/examples_utils/benchmarks>

## Profiling

Profiling can be done easily via the `examples_utils` module, simply by adding the `--profile` argument when using the `benchmark` submodule (see the <strong>Benchmarking</strong> section above for further details on use). For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --profile
```
Will create folders containing popvision profiles in this applications root directory (where the benchmark has to be run from), each folder ending with "_profile". 

The `--profile` argument works by allowing the `examples_utils` module to update the `POPLAR_ENGINE_OPTIONS` environment variable in the environment the benchmark is being run in, by setting:
```
POPLAR_ENGINE_OPTIONS = {
    "autoReport.all": "true",
    "autoReport.directory": <current_working_directory>,
    "autoReport.outputSerializedGraph": "false",
}
```
Which can also be done manually by exporting this variable in the benchmarking environment, if custom options are needed for this variable.

## Licensing

This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

This directory contains derived work from:
* BERT, https://github.com/google-research/bert (licensed under the Apache License, Version 2.0)
* Hugging Face Transformers, https://github.com/huggingface/transformers (licensed under the Apache License, Version 2.0)

See the headers in the source code for details.

The following files include code derived from https://github.com/huggingface/transformers which uses Apache License, Version 2.0:
* bert_fused_attention.py
* squad_data.py

The following files include code derived from https://github.com/google-research/bert which uses Apache License, Version 2.0:
* third_party/create_pretraining_data.py
