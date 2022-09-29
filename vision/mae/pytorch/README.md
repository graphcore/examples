PyTorch MAE Self-Supervised Learning example using ViT
---

Implementation of MAE model in PyTorch for the IPU. This example is based on the models provided by the [`MAE`](https://github.com/facebookresearch/mae). The MAE model is based on the original paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf).

## Environment Setup

First, Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
```
## Compile custom operation

```console
cd remap
make clean
make
cd ..
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

## Execution Order

1. Pretraining
2. Finetuning
3. Validation

## Run Pretraining

Setup your environment as explained above. You can run MAE on ImageNet1k datasets.

Shell scripts that wrap up above arguments of the cluster setup and the python launcher configurations to run on POD16 and POD64 are provided. Use `bash mae_base_pod16.sh` or `bash mae_base_pod64.sh` to see meaning of the arguments. Specifically, these arguments are listed in the table below.

| argument | meaning |
|:----:|:----|
| -n | Hostnames/IPs of the hosts |
| -s | Hostname/IP of the controller server |
| -p | partition name |
| -c | cluster name |

To run pre-training on a single host using POD16:
```console
bash scripts/mae_base_pod16.sh
```

To run pre-training on multi-hosts using POD64:
```console
bash scripts/mae_base_pod64.sh -n host1,host2,host3,host4 -s host0 -p partition_name -c cluster_name
```

Trained with the default fp16, the finetune accuracy is 83.37%(top1) after 1600 epochs. Trained with fp32, the finetune accuracy is 83.4%(top1) after 1600 epochs.

Once the training finishes, you can validate with finetune accuracy:

## Run finetune

```python
python main_finetune.py --finetune ${MODEL_DIR} --data_path ${IMAGENET_DIR} \
```

## Run validate

```python
python finetune_validate.py --resume ${MODEL_DIR}  --batch_size 16 --data_path ${IMAGENET_DIR}
```

## Licensing
This application is licensed under Apache License 2.0 and Attribution-NonCommercial 4.0 International.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under Apache License, Version 2.0 (<sup>*</sup> means additional license information stated following this list):
* main_pretrain.py
* options.py
* argparser.py
* configs.yml
* README.md
* requirements.txt
* core/utils.py
* core/gelu.py
* scripts/mae_base_pod16.sh
* scripts/mae_base_pod64.sh
* scripts/alignment.py
* scripts/alignment.sh
* scripts/eval.sh
* util/pos_embed.py
* util/crop.py
* util/checkpoint.py
* util/ipu_mixup.py
* test/test_mae.py
* test/conftest.py
* remap/remap_ops/TileMappingCommon.cpp
* remap/remap_ops/TileMappingCommon.hpp
* remap/remap_ops/remap_tensor_ce.cpp
* remap/remap_ops/remap_tensor_ce.hpp



The following files include code derived from this [repo](https://github.com/facebookresearch/mae) which uses Attribution-NonCommercial 4.0 International license: 
* util/log.py
* util/lr_decay.py
* util/lr_sched.py
* util/datasets.py
* core/vision_transformer.py
* core/models_mae.py
* core/models_vit.py
* main_finetune.py
* finetune_validate.py

External packages:
- `transformers` is licenced under Apache License, Version 2.0
- `pytest` is licensed under MIT License
- `timm` is licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License