PyTorch DINO Self-Supervised Learning example using ViT
---

Implementation of DINO model in PyTorch for the IPU. This example is based on the models provided by the [`DINO`](https://github.com/facebookresearch/dino). The DINO model is based on the original paper [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).

## Environment Setup

First, Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```

## Custom operation

```console
sh make_ema.sh
```

## Datasets

Download the datasets:
* ImageNet dataset (available at [http://www.image-net.org/](http://www.image-net.org/))

The ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes, can be downloaded from [the ImageNet website](http://www.image-net.org/download). It is approximately 150GB for the training and validation sets. Please note you need to register and request permission to download this dataset on the Imagenet website. You cannot download the dataset until ImageNet confirms your registration and sends you a confirmation email. If you do not get the confirmation email within a couple of days, contact [ImageNet support](support@imagenet.org) to see why your registration has not been confirmed. Once your registration is confirmed, go to the download site. The dataset is available for non-commercial use only. Full terms and conditions and more information are available on the [ImageNet download](http://www.image-net.org/download)

Please place or symlink the ImageNet data in `./data/imagenet1k`.
The imagenet1k dataset folder contains `train` and `validation` folders, in which there are the 1000 folders of different classes of images.
```console
imagenet1k
|-- train [1000 entries exceeds filelimit, not opening dir]
`-- validation [1000 entries exceeds filelimit, not opening dir]
```

## Run the application

Setup your environment as explained above. You can run DINO on ImageNet1k datasets and choosing between fp32 and fp16.

There are two model sizes available: `vit_small_pod16` and `vit_base_pod16`. The `vit_small_pod16` configuration can be run in a few minutes:


bash training_scripts/vit_small_pod16.sh


The `vit_base_pod16` configuration reaches the state-of-the-art results and takes longer to train:


bash training_scripts/vit_base_pod16.sh


Trained with the default fp16, the knn accuracy is 71.7%(top1) after 100 epochs. Trained with fp32, the knn accuracy is 72.4%(top1) after 100 epochs. We set the learning rate as 0.0005 and global batch size as 256.

Once the training finishes, you can validate with knn accuracy:

## Run the application with pod16

```
bash training_scripts/vit_base_pod16.sh
```

## Run the application with pod64

```
bash training_scripts/vit_base_pod64.sh
```

## Run the application with pod256
```
bash training_scripts/vit_base_pod256.sh
```

console
cd script
# checkpoint.pth is the latest weight of DINO
python extract_weights.py --weights path/to/checkpoint.pth --output vit_model.pth
sh eval_knn.sh vit_model.pth

Once the training finishes, you can validate with knn accuracy:
```console
cd script
# checkpoint.pth is the latest weight of DINO
python extract_weights.py --weights path/to/checkpoint.pth --output vit_model.pth
sh eval_knn.sh vit_model.pth
```


## Run the test

```console
pytest tests_serial/test_dino.py
```

## Licensing
This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under Apache License, Version 2.0  (<sup>*</sup> means additional license information stated following this list):
* train_ipu.py
* options.py
* configs.yml
* make_ema.sh
* core/dataset.py
* core/gelu.py
* core/weight_norm.py
* ema/test_ema.py
* ema/exp_avg_custom_op.cpp
* ema/Makefile
* README.md
* requirements.txt
* tests/__init__.py
* tests/conftest.py
* tests/test_dino.py
* script/alignment.sh
* script/alignment.py
* script/grad_compare.py
* script/eval_knn.sh
* script/extract_weights.py
* script/extract_feature.py
* script/knn_accuracy.py

The following files include code derived from this [repo](https://github.com/facebookresearch/dino) which uses MIT license:
* core/dino.py
* core/vision_transformer.py
* core/utils.py

External packages:
- `transformers` is licenced under Apache License, Version 2.0
- `pytest` is licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License