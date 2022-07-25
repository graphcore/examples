# PyTorch GPT2

This directory contains an implementation of GPT2 models in PyTorch for the IPU, leveraging the HuggingFace Transformers library. 

There are two examples for GPT2, the one is for pretraining: `train_gpt2.py` and the second one is text generation `text_generate_gpt2.py`

## Environment setup

### 1. Install the Poplar SDK
SDK version: 2.5

First, install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for Poplar and PopART.

Then, create a virtualenv:
```
virtualenv venv -p python3.6
source venv/bin/activate
```
### 2. Compile custom ops
From inside this directory:
```
make
```
This should create `custom_ops.so`.

### 3. Python
Install the required packages:
```
pip install -r requirements.txt
```

## Run the tests (optional)
Setup your environment as explained above and run `python3 -m pytest` from the root folder.


## Quick start with generated mock dataset

Setup your environment as explained above and run the example with generated datas.

```
python train_gpt2.py \
    --model gpt2 \
    --max-len 1024 \
    --layers-per-ipu 0 4 4 4 \
    --matmul-proportion 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --replication-factor 1 \
    --epochs 3 \
    --gradient-accumulation 512 \
    --device-iterations 1 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --compile-only False
```

## Generate pretraining data

The dataset used for pretraining is WIKI-103. It can be generated from a RAW dump of Wikipedia following a five step process.

### 1. Download

Use the `wikipedia_download.sh` script to download the latest Wikipedia dump, about 20GB in size.

```
./data/wikipedia_download.sh <chosen-path-for-dump-file>
```

Dumps are available from <https://dumps.wikimedia.org/> (and mirrors) and are licensed under CC BY-SA 3.0 and GNU Free Documentation Licenses.

### 2. Extraction

In order to create the pretraining data we need to extract the Wikipedia dump and put it in this form:

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

```
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
```

You can then use the the `wikipedia_extract.sh` script to use WikiExtractor to extract the data dump.

```
./data/wikipedia_extract.sh <chosen-path-for-dump-file>/wikidump.xml <chosen-folder-for-extracted-files>
```

The result should be a folder containing directories named `AA`, `AB`, ...
Note that the number of directories depends on the parameters of the `wikipedia_extract.sh` script, and is not to be confused with alphabetical ordering of the wikipedia articles.
In other words you should probably not expect all of `AC`, `AD`, ... `ZX`, `ZY`, `ZZ` to be created by the script.

### 3. Pre-processing

Use the `wikipedia_preprocess.py` script to preprocess and tokenize the extracted files.

Huggingface's `tokenizer.GPT2Tokenizer` is used in this step as default.

```
python3 ./data/wikipedia_preprocess.py --input-file-path <chosen-folder-for-extracted-files> --output-file-path <chosen-folder-for-preprocessed-files>
```

Now you should get the `.pkl` data, which will be used in the pretraining.

## Run the pretraining application
**Notice**: The default scripts are used to get benchmarks for throughput only. You must passing path to processed data files to `--train-path` to start the actual pretraining, you may also need to specify the `--save-model-path` to save checkpoints. It is recommended to use `--gradient-accumulation 512` when pretraining on the wikipedia dataset for better convergence. It takes 20 epochs(about 0.15 days per epoch) to reach a relative low LM loss together with the SOTA accuracy on evaluation tasks.

Further arguments are described in the source file `arguments.py`.

There are four GPT2 models:
* GPT2 Small - 12 layers (transformer blocks), 117 million parameters
* GPT2 Medium - 24 layers (transformer blocks), 345 million parameters
* GPT2 Large - 36 layers (transformer blocks), 762 million parameters
* GPT2 XLarge - 48 layers (transformer blocks), 1542 million parameters

The JSON configuration files provided in the configs directory `config/` define detailed parameters for GPT2 models.

### Run GPT2-small
This script runs the 117M parameter GPT2 pretraining.
```
bash run/pretraining_small.sh
```
or
```
python train_gpt2.py \
    --model gpt2 \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 4 4 4 \
    --matmul-proportion 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --replication-factor 4 \
    --epochs 20 \
    --gradient-accumulation 2048 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False
```
### Run GPT2-medium
This script runs the 345M parameter GPT2 pretraining.
```
bash run/pretraining_medium.sh
```
or
```
python train_gpt2.py \
    --model gpt2-medium \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 3 3 3 3 4 4 4 \
    --matmul-proportion 0.30 0.15 0.15 0.15 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 8 \
    --replication-factor 2 \
    --epochs 20 \
    --gradient-accumulation 4096 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False
```
### Run GPT2-large(SL=512)
This script runs the 762M parameter GPT2 pretraining, with sequence length=512.
```
bash run/pretraining_large_512.sh
```
or
```
python train_gpt2.py \
    --model gpt2-large \
    --max-len 512 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 1 5 5 5 5 5 5 5 \
    --matmul-proportion 0.15 0.12 0.15 0.15 0.15 0.15 0.15 0.15 \
    --ipus-per-replica 8 \
    --replication-factor 2 \
    --epochs 20 \
    --gradient-accumulation 4096 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 8 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding True \
    --compile-only False
```
### Run GPT2-large(SL=1024)
This script runs the 762M parameter GPT2 pretraining, with sequence length=1024.
```
bash run/pretraining_large_1024.sh
```
or
```
python train_gpt2.py \
    --model gpt2-large \
    --max-len 1024 \
    --optimizer AdamW \
    --learning-rate 0.00015 \
    --lr-schedule cosine \
    --lr-warmup 0.01 \
    --layers-per-ipu 0 2 2 2 2 2 2 2 2 3 3 3 3 3 3 2 \
    --matmul-proportion 0.2 0.15 0.2 0.2 0.2 0.15 0.15 0.2 0.2 0.15 0.2 0.2 0.2 0.15 0.15 0.2 \
    --ipus-per-replica 16 \
    --replication-factor 1 \
    --epochs 20 \
    --gradient-accumulation 8192 \
    --device-iterations 8 \
    --batch-size 1 \
    --enable-sequence-serialized True \
    --remap-logit True \
    --embedding-serialization-factor 4 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'generated' \
    --replicated-tensor-sharding False \
    --compile-only False
```

### Run GPT2-large by PopRun
This script runs the 762M parameter GPT2 distributed pretraining using PopRun, which can scale the application from POD16 to POD64.

We advise you to first read through the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/index.html) for PopRun before running this script.
```
bash run/pretraining_large_poprun.sh
```

## TFRecord dataset (optional)
In order to use the multi-threaded `dataloader`, `tfrecord` files need to be generated.
```
cd <chosen-folder-for-preprocessed-files>
mkdir tfrecords
python write_into_tfrecord.py

cd tfrecords
for f in *.tfrecord; do python3 -m tfrecord.tools.tfrecord2idx $f `basename $f .tfrecord`.index; done
```
then add `--train-path 'tfrecord'` and `--tfrecord-path <path>/*.tfrecord` to the command lines.


## Megatron dataset (optional)
We also support mmap dataset which is used by NVIDIA's Megatron, you should first follow the [instruction](https://github.com/ningchaoar/Megatron-LM#data-preprocessing) to get `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`.
Then you need to set `--train-path dynamic` and `--data-prefix <path>/my-gpt2_text_document` in the training scripts.

If you have trained model using Megatron's dataset, you will need to set `--tokenizer-type 1` in `tasks/run_evaluate.sh` for evalutaion.

## Evaluation
### WikiText Perplexity Evaluation
we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), 
and appropriately compute perplexity given the change in tokens 
when using our generated BPE tokenizer.

We use the following command to run WikiText-103 evaluation on pretrained model.
```
bash tasks/run_evaluate.sh wiki
```

### LAMBADA Cloze Accuracy
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceeding tokens) 
we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a pretrained model.
```
bash tasks/run_evaluate.sh lmbd
```

##  Text Generation
```
bash tasks/run_text_generator.sh
```
or
```
python text_generate_gpt2.py \
      --model-name-or-path gpt2 \
      --fp16 true \
      --single-ipu true \
      --poptorch-loop true \
      --output-len 256
```
This task is interactive. You can enter one sentence such as "My name is " and it will generate the rest sentences.  

| Arguments | Meaning |
| ---- | ---- | 
| --model-name-or-path | Path to pre-trained model or shortcut names from huggingface.co. The default is 'gpt2' |
| --fp16 | Whether to use fp16 model for this task|
| --single-ipu | Whether to use single IPU for this task |
| --poptorch_loop| Whether to use poptorch_loop to optimize the latency, only supported on single ipu|
| --batch-size| batch size for inference. The defult is 1|
| --input-len| The maximum input length you want to set for this task|
| --output-len| The maximum output length you want to set for this task|

Further arguments are described in the source file `text_generate_gpt2.py`

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
