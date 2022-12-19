# CLIP
CLIP (ViT-B/32) based on the models provided by the [openai-CLIP](https://github.com/openai/CLIP) models, optimised for Graphcore's IPU.

| Framework | domain | Model | Datasets | Tasks| Training| Inference | Reference |
|-------------|-|------|-------|-------|-------|---|-------|
| Pytorch | Vision | CLIP | Conceptual Captions (cc3m), Imagenet LSVRC 2012, CIFAR-100 | Image recognition | ✅  | ❌ | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) |


## Instructions summary
1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the ImageNet LSVRC 2012 dataset (See Dataset setup)


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

3. Additionally, enable PopArt with:
```bash 
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your environment are available in the [poplar quick start guide](https://docs.graphcore.ai/projects/graphcloud-poplar-quick-start/en/latest/).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (Pytorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```

4. Navigate to this example's root directory

5. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

## Dataset setup
### Conceptual Captions (cc3m)
Download the conceptual captions dataset in three steps with the scripts provided:
1. Download `Train_GCC-training.tsv` from the [Conceptual Captions source](https://ai.google.com/research/ConceptualCaptions/download)

2. Use the provided script to download the main dataset:
```bash
mkdir data
mv Train_GCC-training.tsv data/
mkdir -p data/cc3m/images
python3 datasets/download.py --url_file data/Train_GCC-training.tsv --save_path data/cc3m
```

3. Download the word segmentation vocabulary from the  [official CLIP repository](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) and move it into the data directory:

```bash
mv bpe_simple_vocab_16e6.txt.gz datasets/
```

Disk space required: 84G

```bash
.
├── images
└── img_cap.csv

1 directory, 1 file
```

### ImageNet LSVRC 2012 (Optional)
Download the ImageNet LSVRC 2012 dataset from [the source](http://image-net.org/download) or [via kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)

Disk space required: 144GB

```bash
.
├── bounding_boxes
├── imagenet_2012_bounding_boxes.csv
├── train
└── validation

3 directories, 1 file
```

And then pre-process the dataset using the scripts provided:
```bash
python3 datasets/preprocess.py
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


## Custom training/inference and other features

### Zero-shot evaluation  
After training CLIP on cc3m dataset, you can apply zeroshot classification prediction on the validation set of ImageNet1k and CIFAR100 dataset to valify the performance of trained model. You can choose to use a checkpoint saved from the IPU by setting the `is_ipu_ckpt` to `True` or the official checkpoint by setting it to `False`. Zeroshot evaluation is performed on the validation set of ImageNet1k by default. If you want to perform zeroshot evaluation on CIFAR100, please set `zeroshot_dataset` to CIFAR100.

```bash
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
