PyTorch Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
---

Implementation of Swin model in PyTorch for the IPU. This example is based on the models provided by the [`SWIN`](https://github.com/microsoft/Swin-Transformer). The Swin model is based on the original paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf).

## Environment Setup

First, Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages by 'pip install -r requirements.txt'.


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
Then modify the DATA.DATA_PATH parameter in config we use into ImageNet's path

## Quick start
ImageNet1k training(default is fp32.32):
```console
python train_swin.py --cfg SWIN_TINY --data-path ./data/imagenet1k --output ./output/swin_tiny_224/
```

## Run the application

Setup your environment as explained above. You can run SWIN on ImageNet1k datasets. Train SWIN on IPU. 
We can train with fp32.32 and fp16.16.(The first 16 or 32 means that input is half or float, and the second means that weight is half or float)
We can use the PRECISION parameter in config to modify the required precision

Acc as follows: 

| model | input size | precision | machine |     acc     |
|-------|------------|-----------|---------|-------------|
| tiny  |   224      |    16.16  |  pod16  |    80.9%    |
| tiny  |   224      |    32.32  |  pod16  |    81.21%   |
| tiny  |   224      |     mix   |  v100   |  81.3%(SOTA)|
| base  |   224      |    32.32  |  pod16  |    83.5%    |
| base  |   224      |     mix   |  v100   |  83.5%(SOTA)|
| base  |   384      |    32.32  |  pod16  |    84.47%    |
| base  |   384      |     mix   |  v100   |  84.5%(SOTA)|
| large |   224      |    32.32  |  pod16  |    86.27%    |
| large |   224      |     mix   |  v100   |  86.3%(SOTA)|
NOTE:The Acc which marked SOTA is quoted from the paper

In the above results, base 384 and large are finetune models it needs the pretrained model form imagenet21k, which require you to provide the path of the "pretrained-model" parameter to get the correct ACC

You can load the base 384 model pretrained on imagenet21k by:
```console
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```
and large 224 model pretrained on imagenet21k by: 
```console
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
```

Once the training finishes, you can validate accuracy:
```console
python3 validate.py --cfg YOUR_CONFIG --checkpoint /path/to/checkpoint.pth
```
Train on pod16 and train poprun on pod(64,128)  script references: swin_large_224_22k_finetune_1k_fp16_pod16.sh ,swin_large_224_22k_finetune_1k_fp16_pod64.sh

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

## Licensing
This application is licensed under Apache License 2.0. Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under Apache License, Version 2.0 :
* configs/*
* dataset/build_ipu.py
* models/build.py
* options.py
* README.md
* swin_test.py
* train_swin.py
* utils.py
* validate.py
The following files are based on code from [repo](https://github.com/microsoft/Swin-Transformer) which is licensed under the  MIT License: 
* config.py
* dataset/cached_image_folder.py
* dataset/samplers.py
* dataset/zipreader.py
* lr_scheduler.py
* model/swin_transformer.py
* optimizer.py
* train_swin.sh
See the headers in the files for details.
  
External packages:
- `transformers` is licenced under Apache License, Version 2.0
- `pytest` is licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License