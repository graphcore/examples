# BERT (PyTorch)
Bidirectional Encoder Representations from Transformers for NLP pre-training and fine-tuning tasks (SQuAD), using the [huggingface transformers library](https://huggingface.co/docs/transformers/index), optimised for Graphcore's IPU.

Run our BERT-L Fine-tuning on SQuAD dataset on Paperspace.
<br>
[![Gradient](../../../gradient-badge.svg)](https://ipu.dev/3GTWwK7)

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | NLP | BERT | WIKI-103 | Next sentence prediction, Masked language modelling, Question/Answering | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required  | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805v2) |


## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the WIKI-103 dataset (See Dataset setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

3. Additionally, enable PopART with:
```bash
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```

4. Navigate to this example's root directory

5. Install the apt requirements:
```bash
sudo apt install $(< required_apt_packages.txt)
```

6. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).

## Dataset setup
The dataset used for pretraining is WIKI-103. It can be generated from a RAW dump of Wikipedia following a five step process.

Disk space required: 143GB - Sequence length 128 (Variable), 203GB - Sequence length 512 (Variable)

```bash
.
├── wiki_000.index
├── wiki_000.tfrecord
    .
    .
    .
├── wiki_xxx.index
└── wiki_xxx.tfrecord

0 directories, XXXX files
```

### 1. Download

Use the `wikipedia_download.sh` script to download the latest Wikipedia dump, about 20GB in size.

```bash
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
Install the WikiExtractor package with:
```bash
pip3 install wikiextractor
```

In order not to encounter a `UnicodeEncodeError` at this step, you may want to run these two commands first:

```bash
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
```

You can then use the the `wikipedia_extract.sh` script to use WikiExtractor to extract the data dump.

```bash
./data/wikipedia_extract.sh <chosen-path-for-dump-file>/wikidump.xml <chosen-folder-for-extracted-files>
```

The result should be a folder containing directories named `AA`, `AB`, ...
Note that the number of directories depends on the parameters of the `wikipedia_extract.sh` script, and is not to be confused with alphabetical ordering of the wikipedia articles.
In other words you should probably not expect all of `AC`, `AD`, ... `ZX`, `ZY`, `ZZ` to be created by the script.

### 3. Pre-processing

Install nltk package with:
```bash
pip3 install nltk
```

Use the `wikipedia_preprocess.py` script to preprocess the extracted files.

```bash
python3 ./data/wikipedia_preprocess.py --input-file-path <chosen-folder-for-extracted-files> --output-file-path <chosen-folder-for-preprocessed-files>
```

### 4. Tokenization

The script `create_pretraining_data.py` can accept a glob of input files to tokenize.
However, attempting to process them all at once may result in the process being killed by the OS for consuming too much memory.
It is therefore preferable to convert the files in groups. This is handled by the `./data/wikipedia_tokenize.py` script.
At the same time, it is worth bearing in mind that `create_pretraining_data.py` shuffles the training instances across the loaded group of files, so a larger group would result in better shuffling of the samples seen by BERT during pre-training.

The tokenization depends on `tensorflow` which can be installed by:

```bash
pip3 install tensorflow
```

sequence length 128

```bash
python3 ./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 128 --mask-tokens 20
```

sequence length 512

```bash
python3 ./data/wikipedia_tokenize.py <chosen-folder-for-preprocessed-files> <chosen-folder-for-dataset-files> --sequence-length 512 --mask-tokens 76
```

### 5. Indexing

In order to use the multi-threaded `dataloader`, `tfrecord` index files need to be generated.
First install the `tfrecord` Python package into your Python environment:

```bash
pip3 install tfrecord
```

Then go to the directory containing the pre-processed Wikipedia files and run:

```bash
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
```

### 6. Packing (optional)

Packing can lead to significant speed-ups during pretraining (details in https://arxiv.org/pdf/2107.02027.pdf). The packing scripts depend on `tensorflow` and `numpy` which can be installed by `pip3 install tensorflow numpy`. The following commands pack the 128 and 512 sequence-length datasets with a maximum of 3 sequences per pack:

sequence length 128

```bash
python3 -m data.packing.pack_pretraining_data --input-files=<path-of-unpacked-input-data-files> --output-dir=<path-of-output-packed-data-folder> --sequence-length 128 --mask-tokens 20
```

sequence length 512

```bash
python3 -m data.packing.pack_pretraining_data --input-files=<path-of-unpacked-input-data-files> --output-dir=<path-of-output-packed-data-folder> --sequence-length 512 --mask-tokens 76
```

After packing it is recommended to shuffle again the dataset.

```bash
python3 -m data.packing.shuffle_packed_data --input-files=<path-of-unshuffled-packed-data-files> --output-dir=<path-of-output-shuffled-packed-data-folder>
```

Remember to index the tfrecord files as in 5) after packing and shuffling. The following bash scripts run the full packing preprocess (packing plus shuffling) — the whole process for each sequence length takes approximately 6 hours:

sequence length 128

```bash
./data/packing/pack_128.sh <path-of-input-unpacked-data-folder> <path-of-output-packed-data-folder>
```

sequence length 512

```bash
./data/packing/pack_512.sh <path-of-input-unpacked-data-folder> <path-of-output-packed-data-folder>
```


## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. The benchmarks are provided in the `benchmarks.yml` file in this example's root directory.

For example:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```bash
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).


## Custom training

### Configurations

To see the available configurations for both SQuAD and pre-training see the `configs_pretraining.yml` file.

To see the available options available to use in the command line interface use the `--help` argument:

```bash
python3 run_pretraining.py --help
# or
python3 run_squad.py --help
# or
python3 run_benchmark_with_triton_server.py --help ./tests_serial/tritonserver/
```

### Running pre-training with checkpointing

To enable the saving of model checkpoints on a run you need to add `--checkpoint-output-dir <path/to/checkpoint/dir>` to the command line. By default this will save a model checkpoint at the start and end of training.

Additionally, for more frequent outputting of checkpoints you can add `--checkpoint-steps <nsteps>` to save a model checkpoint after every `nsteps` training steps.

To load model weights from a checkpoint directory use the flag `--checkpoint-input-dir <path/to/checkpoint/step_N>`. (You can also pass the name of a model from HuggingFace model hub here too.) To also resume a training run from a checkpoint, also add the flag `--resume-training-from-checkpoint`.

### Run the SQuAD application

The question answering with SQuAD example is found in the `run_squad.py` script. Like with pre-training there are SQuAD configs defined in `configs_squad.yml`.

To run BERT-Base:

```bash
python3 run_squad.py --config squad_base_384
```

For BERT-Large there is `squad_large_384`, which is a high performance large configuration that uses an 8 IPU pipeline, unlike the other configs that use 4.

You will also need to specify a pre-trained checkpoint to fine-tune, which is specified with the `--checkpoint-input-dir <FILE-PATH/HF-model-hub-name>` flag.


## Custom inference

### Running inference with Triton Server
To run all tests with Triton Server:
```bash
python3 run_benchmark_with_triton_server.py -s ./tests_serial/tritonserver/
```


## Other features

### Caching executables

When running the application, it is possible to save/load executables to/from a cache store. This allows for reusing a saved executable instead of re-compiling the model when re-running identical model configurations. To enable saving/loading from the cache store, use `--executable-cache-dir <relative/path/to/cache/store>` when running the application.

### Packed BERT

You can also enable sequence packing for even more efficient BERT pretraining. To enable, add the
`--packed-data` flag to your command line when running any pretraining config and use `--input-files` to
point to the packed version of the dataset. For instance:

```bash
python3 run_pretraining.py --config demo_tiny_128 --packed-data --input-files data/packing/*.tfrecord
```

### Employing automatic loss scaling (ALS) for half precision training

ALS is a feature in the Poplar SDK which brings stability to training large models in half precision, specially when gradient accumulation and reduction across replicas also happen in half precision.

NB. This feature expects the `poptorch` training option `accumulationAndReplicationReductionType` to be set to `poptorch.ReductionType.Mean`, and for accumulation by the optimizer to be done in half precision (using `accum_type=torch.float16` when instantiating the optimizer), or else it may lead to unexpected behaviour.

To employ ALS for BERT Large pre-training on a POD16, the following command can be used:

```bash
python3 run_pretraining.py --config pretrain_large_128_ALS --checkpoint-output-dir checkpoints/pretrain_large_128
```

To pre-train with ALS on a POD64:

```bash
python3 run_pretraining.py --config pretrain_large_128_POD64_ALS --checkpoint-output-dir checkpoints/pretrain_large_128
```

### Generate sample_text dataset

The sample text provided enables training on a very small dataset for small scale testing.
For convenience it is already provided in the `/data` folder in `txt` and `tfrecord` format.
In order to re-generate the sample dataset, run the following script:

```bash
python3 third_party/create_pretraining_data.py --input-file data/sample_text.txt --output-file data/sample_text.tfrecord --sequence-length 128 --mask-tokens 20 --duplication-factor 4 --do-lower-case --model bert-base-uncased
```

### Troubleshooting

If Triton server tests fails with such error:

* ```[model_runtime:cpp] [error] Error in model_runtime/source/Executable.cpp:38:Failed to deserialize XXX : Error reading executable - package hash (YYY) differs from poplar hash (ZZZ)```

	This mean that models were generated and saved with different version of SDK and needs to be recreated. Please remove `tests_serial/tritonserver/test_environment_ready.lock` and rerun tests.

* ```Failed: Failed to download and/or compile Triton Server!```

	Most probably some system packages are missing, ensure that all packages listed in `required_apt_packages.txt` are installed. Also refer to Triton Server build log file. After fixing error remove `../../../utils/triton_server/triton_environment_is_prepared.lock` and rerun tests.


## License

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
