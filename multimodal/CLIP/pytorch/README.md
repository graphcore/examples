# Pytorch CLIP on IPU  

Implementation of CLIP (`ViT-B/32`) model in PyTorch for the IPU. This example is based on the models provided by the [`openai-CLIP`](https://github.com/openai/CLIP). The CLIP (`ViT-B/32`) model is based on the original paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## Setup Environment  

First, install the Poplar SDK following the instructions in the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment and install the required packages.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt

# from sdk package
pip install <path to the poptorch wheel from the Poplar SDK>
```  

## Quick start with generated dataset  

Setup your environment as explained above and run the example with generated data.  

```console
python train.py \
    --config CLIP_ViT-B-32_cc3m \
    --host_generate_data True
```

## Download Vocabulary  

Download the vocabulary for word segmentation into the dataset directory from [`the official repository`](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) and then move it into datasets directory.  

```console
mv bpe_simple_vocab_16e6.txt.gz datasets/
```

## Download Dataset  

The script `datasets/download.py` will collect the Conceptual Captions images. First, download `Train_GCC-training.tsv` from [`Conceptual Captional URLs`](https://ai.google.com/research/ConceptualCaptions/download) and then move it into the data directory. Finally, run the script from our repository to grab the images:  

```console
mkdir data
mv Train_GCC-training.tsv data
mkdir -p data/cc3m/images
python datasets/download.py \
    --url_file data/Train_GCC-training.tsv \
    --save_path data/cc3m
```  

The trainset of cc3m has about 2.8 M image-text pairs.

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

## Run the application

After setting up your environment as explained above, you can run CLIP (with ViT-B/32) on cc3m dataset.

Train CLIP (with ViT-B/32) on cc3m dataset:
```console
python train.py --config CLIP_ViT-B-32_cc3m
```  

## Configurations

You can find the available configurations in the `configs.yml` file.
You can use `--help` to check the available options.

```console
python train.py --help
```  

## Run the unit test  

```console
pytest
```  

## Run the tests  

Before performing zeroshot evaluation on ImageNet, you need to download the validation set of [`ImageNet1k dataset`](http://www.image-net.org). And then move it into the data directory. Finally, run the script from our repository to filter the non-existing images and build the map file for image-label:  

```console
python datasets/preprocess.py
```

Note: CIFAR100 dataset will download automatically when performing zeroshot evaluation on CIFAR100.

After training CLIP on cc3m dataset, you can apply zeroshot classification prediction on the validation set of ImageNet1k and CIFAR100 dataset to valify the performance of trained model. You can choose to use a checkpoint saved from the IPU by setting the `is_ipu_ckpt` to `True` or the official checkpoint by setting it to `False`. Zeroshot evaluation is performed on the validation set of ImageNet1k by default. If you want to perform zeroshot evaluation on CIFAR100, please set `zeroshot_dataset` to CIFAR100.

```console
# Do zeroshot evaluation on ImageNet
python zero_shot.py \
    --config CLIP_ViT-B-32_cc3m \
    --is_ipu_ckpt True \
    --zeroshot_dataset imagenet \
    --ckpt_file output/ckpt/CLIP_epoch_K.pt

# Do zeroshot evaluation on CIFAR100
python zero_shot.py \
    --config CLIP_ViT-B-32_cc3m \
    --is_ipu_ckpt True \
    --zeroshot_dataset cifar100 \
    --ckpt_file output/ckpt/CLIP_epoch_K.pt
```

## Licensing  

This application is licensed under MIT license. Please see the LICENSE file in this directory for full details of the license conditions.  

The following files are created by Graphcore and are licensed under MIT License (<sup>*</sup> means additional license information stated following this list):  

* log.py
* args.py
* train.py
* README.md
* configs.yml
* benchmarks.yml
* preprocess.py
* checkpoint.py
* ipu_options.py
* optimization.py
* requirements.txt
* tests/import_helper.py
* tests/cpu_ipu_test.py
* datasets/preprocess.py
* datasets/text_templates.pt

The following file include code from this [`repo`](https://github.com/openai/CLIP) which uses MIT license:  

* model.py
* datasets/simple_tokenizer.py

The following file include code from this [`repo`](https://github.com/mlfoundations/open_clip).  

* zers_shot.py
* datasets/dataset.py
* datasets/download.py

External packages:  
* `wandb`, `pytest`, `pyyaml`, `transformers` are licensed under MIT License
* `torchvision` is licensed under BSD 3-Clause License
