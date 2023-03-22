# GPT-2
GPT-2 for NLP pre-training and text generation, using the [huggingface transformers library](https://huggingface.co/docs/transformers/index), optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | NLP | GPT-2 | Wikipedia | Next sentence prediction, Masked language modelling, Question/Answering | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required  | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) |


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

5. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

5. Build the custom ops:
```bash
make
```


More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).

## Dataset setup
### Wikipedia

**Download**

Download the latest raw wikipedia dump from: <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2>. You can also download a sample here <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2>.

**Extract**

Once you have downloaded the raw file, extract it using [WikiExtractor](https://github.com/attardi/wikiextractor).
```bash
pip3 install wikiextractor==3.0.6
python3 -m wikiextractor.WikiExtractor --json --no-templates -o wikipedia_extracted enwiki-latest-pages-articles.xml.bz2
```

Then merge all extracted file into a single json file.

```bash
find ./wikipedia_extracted/ -depth -name wiki_* -exec cat {} + > wikipedia_data.json
```

**Preprocess**

We recommend to follow Nvidia's Megatron for data preprocessing and generated the training data, see <https://github.com/NVIDIA/Megatron-LM/tree/0ed2f6ac943560ab0a8a58b6628a669af8c250db#data-preprocessing>.

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git@0ed2f6ac943560ab0a8a58b6628a669af8c250db
pip3 install nltk
python3 Megatron-LM/tools/preprocess_data.py	 \
       --input wikipedia_data.json \
       --output-prefix wikipedia-gpt2 \
       --vocab $APP_DIR/tokenizer/gpt2-vocab-50256.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $APP_DIR/tokenizer/gpt2-merges-50256.txt \
       --append-eod
```

Where `$APP_DIR` is the directory of this README.

The output files are named `wikipedia-gpt2_text_document.bin` and `wikipedia-gpt2_text_document.idx`, set `--dataset mmap` and `--input-files <path>/wikipedia-gpt2_text_document` in the training scripts to start gpt2 pretraining.

### Webtext

**Download**

Download the open webtext dataset from: <https://skylion007.github.io/OpenWebTextCorpus/>

**Extract**

```bash
tar -xvf openwebtext.tar.xz
python data/extract_and_merge.py ./openwebtext ./openwebtext_extracted
```

The output file is `openwebtext_raw.json`

**Preprocess**

Before generate the binary file for training, we are going to filter, clean, and deduplicate the raw json file. See <https://github.com/NVIDIA/Megatron-LM/blob/main/tools/openwebtext>

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git

python3 Megatron-LM/tools/openwebtext/cleanup_dataset.py openwebtext_raw.json openwebtext_clean.json

python3 Megatron-LM/tools/openwebtext/find_duplicates.py \
	   --inputs openwebtext_clean.json url
	   --output openwebtext_duplicate_url.json

python3 Megatron-LM/tools/openwebtext/group_duplicate_urls.py openwebtext_duplicate_url.json openwebtext_duplicate.json

python3 Megatron-LM/tools/openwebtext/remove_group_duplicates.py openwebtext_duplicate.json openwebtext_clean.json openwebtext_deduplicate.json

shuf openwebtext_deduplicate.json -o openwebtext_deduplicate_shuf.json

python3 Megatron-LM/tools/openwebtext/filter_ngrams.py \
       --tasks lambada \
       --dedup-dataset openwebtext_deduplicate_shuf.json text \
       --output openwebtext.json

python3 Megatron-LM/tools/preprocess_data.py \
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

### Run the pretraining application
**Notice**: The default scripts are used to get benchmarks for throughput only. You must passing path to processed data files to `--input-files` to start the actual pretraining, you may also need to specify the `--checkpoint-output-dir` to save checkpoints. It is recommended to use `--gradient-accumulation 512` when pretraining on the wikipedia dataset for better convergence. It takes 20 epochs(about 0.15 days per epoch) to reach a relative low LM loss together with the SOTA accuracy on evaluation tasks.

Further arguments are described in the source file `arguments.py`.

There are four GPT2 models:
* GPT2 Small - 12 layers (transformer blocks), 117 million parameters
* GPT2 Medium - 24 layers (transformer blocks), 345 million parameters
* GPT2 Large - 36 layers (transformer blocks), 762 million parameters
* GPT2 XLarge - 48 layers (transformer blocks), 1542 million parameters

The JSON configuration files provided in the configs directory `config/` define detailed parameters for GPT2 models.


## Custom inference

###  Text Generation
```bash
bash tasks/run_text_generator.sh
```

or

```bash
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
| --batch-size| batch size for inference. The default is 1|
| --input-len| The maximum input length you want to set for this task|
| --output-len| The maximum output length you want to set for this task|

Further arguments are described in the source file `text_generate_gpt2.py`


## Other features

### WikiText Perplexity Evaluation
we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip),
and appropriately compute perplexity given the change in tokens
when using our generated BPE tokenizer.

We use the following command to run WikiText-103 evaluation on pretrained model.
```bash
bash tasks/run_evaluate.sh wiki
```

### LAMBADA Cloze Accuracy
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceding tokens)
we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a pretrained model.
```bash
bash tasks/run_evaluate.sh lmbd
```


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
