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
### 2. Python
Install the required packages:
```
pip install -r requirements.txt
```

### 3. Compile custom ops
From inside this directory:
```
make
```
This should create `custom_ops.so`.

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
    --dataset 'generated'
```

## Dataset

Wikipedia dataset and Webtext dataset can be used for GPT2 pretraining.
To obtain the data used for pretraining follow the below instructions.

### 1. Wikipedia Dataset

**Download**

Download the latest raw wikipedia dump from: <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2>

**Extract**

Once you have downloaded the raw file, extract it using [WikiExtractor](https://github.com/attardi/wikiextractor).
```
pip install wikiextractor
python -m wikiextractor.WikiExtractor --json --no-templates -o wikipedia_extracted enwiki-latest-pages-articles.xml.bz2
```
Then merge all extracted file into a single json file.
```
find ./wikipedia_extracted/ -depth -name wiki_* -exec cat {} + > wikipedia_data.json
```
**Preprocess**

We recommand to follow Nvidia's Megatron for data preprocessing and generated the training data, see <https://github.com/NVIDIA/Megatron-LM#data-preprocessing>.

```
git clone https://github.com/NVIDIA/Megatron-LM.git
python Megatron-LM/preprocess_data.py \
       --input wikipedia_data.json \
       --output-prefix wikipedia-gpt2 \
       --vocab tokenizer/gpt2-vocab-50256.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file tokenizer/gpt2-merges-50256.txt \
       --append-eod
```
The output files are named `wikipedia-gpt2_text_document.bin` and `wikipedia-gpt2_text_document.idx`, set `--dataset mmap` and `--input-files <path>/wikipedia-gpt2_text_document` in the training scripts to start gpt2 pretraining.


### 2. Webtext Dataset

**Download**

Download the open webtext dataset from: <https://skylion007.github.io/OpenWebTextCorpus/>

**Extract**

```
tar -xvf openwebtext.tar.xz
python data/extract_and_merge.py ./openwebtext ./openwebtext_extracted
```
The output file is `openwebtext_raw.json`

**Preprocess**

Before generate the binary file for training, we are going to filter, clean, and deduplicate the raw json file. See <https://github.com/NVIDIA/Megatron-LM/blob/main/tools/openwebtext>

```
git clone https://github.com/NVIDIA/Megatron-LM.git

python Megatron-LM/tools/openwebtext/cleanup_dataset.py openwebtext_raw.json openwebtext_clean.json

python Megatron-LM/tools/openwebtext/find_duplicates.py \
	   --inputs openwebtext_clean.json url
	   --output openwebtext_duplicate_url.json

python Megatron-LM/tools/openwebtext/group_duplicate_urls.py openwebtext_duplicate_url.json openwebtext_duplicate.json

python Megatron-LM/tools/openwebtext/remove_group_duplicates.py openwebtext_duplicate.json openwebtext_clean.json openwebtext_deduplicate.json

shuf openwebtext_deduplicate.json -o openwebtext_deduplicate_shuf.json

python Megatron-LM/tools/openwebtext/filter_ngrams.py \
       --tasks lambada \
       --dedup-dataset openwebtext_deduplicate_shuf.json text \
       --output openwebtext.json

python Megatron-LM/tools/preprocess_data.py \
       --input openwebtext.json \
       --output-prefix openwebtext-gpt2 \
       --vocab tokenizer/gpt2-vocab-50256.json \
       --dataset-impl mmap \
       --workers 8 \
       --chunk-size 200000 \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file tokenizer/gpt2-merges-50256.txt \
       --append-eod
```
The output files are named `openwebtext-gpt2_text_document.bin` and `openwebtext-gpt2_text_document.idx`, set `--dataset mmap` and `--input-files <path>/openwebtext-gpt2_text_document` in the training scripts to start gpt2 pretraining.


## Run the pretraining application
**Notice**: The default scripts are used to get benchmarks for throughput only. You must passing path to processed data files to `--input-files` to start the actual pretraining, you may also need to specify the `--save-model-path` to save checkpoints. It is recommended to use `--gradient-accumulation 512` when pretraining on the wikipedia dataset for better convergence. It takes 20 epochs(about 0.15 days per epoch) to reach a relative low LM loss together with the SOTA accuracy on evaluation tasks.

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
    --dataset 'generated' \
    --replicated-tensor-sharding True
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
    --dataset 'generated' \
    --replicated-tensor-sharding True
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
    --dataset 'generated' \
    --replicated-tensor-sharding True
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
    --dataset 'generated' \
    --replicated-tensor-sharding False
```

### Run GPT2-large by PopRun
This script runs the 762M parameter GPT2 distributed pretraining using PopRun, which can scale the application from POD16 to POD64.

We advise you to first read through the [User Guide](https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/index.html) for PopRun before running this script.
```
bash run/pretraining_large_poprun.sh
```


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



## Licensing

This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

This directory contains derived work from:

* GPT2, https://github.com/openai/gpt-2 (licensed under the MIT License)
* Hugging Face Transformers, https://github.com/huggingface/transformers (licensed under the Apache License, Version 2.0)
* Megatron-LM, https://github.com/NVIDIA/Megatron-LM (license file see https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)
* DeepLearningExamples, https://github.com/NVIDIA/DeepLearningExamples (licensed under the Apache License, Version 2.0)

The following files include code derived from https://github.com/huggingface/transformers which uses Apache License, Version 2.0:
* config/config_large.json
* config/config_medium.json
* config/config_xl.json
* config/config.json
* model/optimized_gpt2_attn.py
* tokenizer/gpt2-merges-50256.txt
* tokenizer/gpt2-vocab-50256.json

The following files include code derived from https://github.com/openai/gpt-2 which uses MIT License.
* tasks/detokenizer.py
* tokenizer/gpt2_tokenization.py

The following files include code derived from https://github.com/NVIDIA/Megatron-LM with license https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE
* data/indexed_dataset.py
* tasks/evaluate_lambada.py
* tasks/evaluate_utils.py
* tasks/evaluate_wiki.py

The following files include code derived from https://github.com/NVIDIA/DeepLearningExamples which uses Apache License, Version 2.0:
* data/wikipedia_preprocess.py
* data/write_into_tfrecord.py