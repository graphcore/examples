Pytorch ViT Fine-tuning example using HuggingFace transformers
---

Implementation of ViT (Vision Transformer) model in PyTorch for IPU. This example is based on the models provided by the [`transformers`](https://github.com/huggingface/transformers) library and from [jeonsworld](https://github.com/jeonsworld/ViT-pytorch). The ViT model is based on the original paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

## Environment Setup

First, Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

## Configurations

To see the available configurations see the `configs.yml` file.
To see the available options available to use in the command line interface use `--help` argument.

```console
python train.py --help
```

## Datasets

Download the datasets:
    * ImageNet dataset (available at [http://www.image-net.org/](http://www.image-net.org/))
    * CIFAR10 dataset downloads automatically

The ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes, can be downloaded from [http://www.image-net.org/download](the ImageNet website). It is approximately 150GB for the training and validation sets. Please note you need to register and request permission to download this dataset on the Imagenet website. You cannot download the dataset until ImageNet confirms your registration and sends you a confirmation email. If you do not get the confirmation email within a couple of days, contact [support@imagenet.org](ImageNet support) to see why your registration has not been confirmed. Once your registration is confirmed, go to the download site. The dataset is available for non-commercial use only. Full terms and conditions and more information are available on the [http://www.image-net.org/download-faq](ImageNet download FAQ)

Please place or symlink the ImageNet data in `./data/imagenet1k`.

## Run the application

Setup your environment as explained above. You can run ViT fine-tuning on either CIFAR10 or ImageNet1k datasets.

CIFAR10 training:
```console
python train.py --config b16_cifar10
```

Once the training finishes, you can validate:
```console
python validation.py --config b16_cifar10_valid
```

To run ImageNet1k fine-tuning you need to first download the data as described above.

ImageNet1k training:
```console
python train.py --config b16_imagenet1k
python validation.py --config b16_imagenet1k_valid
```

Afterwards run ImageNet1k validation:
```console
python validation.py --config b16_imagenet1k_valid
```

The imagenet1k dataset folder contains `train` and `validation` folders, in which there are the 1000 folders of different classes of images.
```console
imagenet1k
|-- train [1000 entries exceeds filelimit, not opening dir]
`-- validation [1000 entries exceeds filelimit, not opening dir]
```

## Run the tests

```console
pytest test_vit.py
```

## Licensing
This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under Apache License, Version 2.0  (<sup>*</sup> means additional license information stated following this list):
* args.py
* checkpoint.py
* configs.yaml
* ipu_options.py
* log.py
* model.py
* Makefile
* metrics.py
* optimization.py
* README.md
* requirements.txt
* run_train.sh
* test_vit.py
* train.py
* **validation.py**<sup>*</sup>
* datasets/__init__.py
* datasets/dataset.py
* datasets/preprocess.py

The following files include code derived from this [repo](https://github.com/jeonsworld/ViT-pytorch) which uses MIT license:
* validation.py

External packages:
- `transformers` and `datasets` are licenced under Apache License, Version 2.0
- `pyyaml`, `wandb`, `pytest`, `pytest-pythonpath` are licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License
