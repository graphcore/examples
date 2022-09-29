Pytorch ViT fine-tuning and pre-training example
---

Implementation of ViT (Vision Transformer) model in PyTorch for IPU. This example is based on the models provided by the [`transformers`](https://github.com/huggingface/transformers) library and from [jeonsworld](https://github.com/jeonsworld/ViT-pytorch). The ViT model is based on the original paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

## Environment Setup

First, install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` scripts for poplar and popART.

Then, create a virtual environment, install the required packages and build the custom ops.

```console
virtualenv venv -p python3.6
source venv/bin/activate
pip install -r requirements.txt
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

## Configurations

To see the available configurations see the `configs.yml` file.
To see the available options to use in the command line interface use `--help` argument.

```console
python train.py --config config_name --help
```

## Datasets

Download the datasets:
* ImageNet dataset (available at [http://www.image-net.org/](http://www.image-net.org/))
* CIFAR10 dataset downloads automatically

The ImageNet LSVRC 2012 dataset, which contains about 1.28 million images in 1000 classes, can be downloaded from [the ImageNet website](http://www.image-net.org/download). It is approximately 150GB for the training and validation sets. Please note you need to register and request permission to download this dataset on the Imagenet website. You cannot download the dataset until ImageNet confirms your registration and sends you a confirmation email. If you do not get the confirmation email within a couple of days, contact [ImageNet support](support@imagenet.org) to see why your registration has not been confirmed. Once your registration is confirmed, go to the download site. The dataset is available for non-commercial use only. Full terms and conditions and more information are available on the [ImageNet download](http://www.image-net.org/download)

Please place or symlink the ImageNet data in `./data/imagenet1k`.

The imagenet1k dataset folder contains `train` and `validation` folders, in which there are the 1000 folders of different classes of images.
```console
imagenet1k
|-- train [1000 entries exceeds filelimit, not opening dir]
`-- validation [1000 entries exceeds filelimit, not opening dir]
```
## Run the application

Setup your environment as explained above.
ViT can be configured for either pre-training or fine-tuning.
Pre-training initializes the weights and trains from scratch.
Fine-tuning loads an already pre-trained checkpoint from HuggingFace and fine-tunes on the dataset provided.
### Pre-Training
To run ImageNet1k pre-training, i.e. randomly initialize the model weights and train from scratch:
```console
python pretrain.py --config b16_in1k_pretrain
```
Afterwards run ImageNet1k validation:
```console
python validation.py --config b16_in1k_pretrain_valid
```
Note: in pre-training Imagenet1k, `micro_batch_size` is set to 8, and all other parameters are tuned to reach a validation accuracy that is higher than 74.79% (released in [google official repository](https://github.com/google-research/vision_transformer/issues/62#issuecomment-888993463)). To achieve maximum throughput, `micro_batch_size` can be set to 14, then hyperparameters tuning is required to reach this validation accuracy.
### Fine-Tuning
You can run ViT fine-tuning on either CIFAR10 or ImageNet1k datasets. The default pretrained checkpoint is loaded from [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k). The commands for fine-tuning are:

CIFAR10 fine-tuning:
```console
python finetune.py --config b16_cifar10
```

Once the fine-tuning finishes, you can validate:
```console
python validation.py --config b16_cifar10_valid
```

To run ImageNet1k fine-tuning you need to first download the data as described above.

ImageNet1k fine-tuning:
```console
python finetune.py --config b16_imagenet1k
```

Afterwards run ImageNet1k validation:
```console
python validation.py --config b16_imagenet1k_valid
```
## Employing automatic loss scaling (ALS) for half precision training

ALS is a feature in the Poplar SDK which brings stability to training large models in half precision, specially when gradient accumulation and reduction across replicas also happen in half precision. 

NB. This feature expects the `poptorch` training option `accumulationAndReplicationReductionType` to be set to `poptorch.ReductionType.Mean`, and for accumulation by the optimizer to be done in half precision (using `accum_type=torch.float16` when instantiating the optimizer), or else it may lead to unexpected behaviour.

To employ ALS for ImageNet1k fine-tuning on a POD16, the following command can be used:

```console
python3 finetune.py --config b16_imagenet1k_ALS
```

## Run on large clusters
This application can be run on large clusters. The following arguments need to be customized according to your own setup. Necessary input arguments of `poprun` command are listed as below:
* Hosts: poprun supports multi-hosts training. The hostnames need to be specified following `--host` argument.
* Controller server: the hostname of the controller needs to be specified to `--vipu-server-host` argument. Use `vipu-admin --server-version` to get related information.
* Network interfaces: specific network interfaces from the control/data plane need to be specified following `--mca oob_tcp_if_include`/`--mca btl_tcp_if_include`. See [here](https://www.open-mpi.org/faq/?category=tcp#ip-multiaddress-devices) to get more details.
* Partition name: partition name needs to be specified to `--vipu-partition`. Partition will be created if specified partition does not exist. Use `vipu-admin list partition` to get available partitions.
* Cluster name: cluster name needs to be specified to `--vipu-cluster`. Use `vipu-admin list allocation` to get available clusters.

IP addresses can also be used where hostnames are needed.

Shell scripts that wrap up above arguments of the cluster setup and the python launcher configurations to run on POD16 and POD64 are provided. Use `bash run_singlehost.sh` or `bash run_multihosts.sh` to see meaning of the arguments. Specifically, these arguments are listed in the table below.

| argument | meaning |
|:----:|:----|
| -n | Hostnames/IPs of the hosts |
| -s | Hostname/IP of the controller server |
| -o | network interface of the control plane |
| -b | network interface of the data plane |
| -p | partition name |
| -c | cluster name |

To run pre-training on a single host using POD16:
```console
bash run_singlehost.sh -s host0 -p partition_name
```

To run pre-training on multi-hosts using POD64:
```console
bash run_multihosts.sh -n host1,host2,host3,host4 -s host0 -o interface1 -b interface2 -p partition_name -c cluster_name
```

**Please note that relevant code, data and python environment on all hosts should be exactly the same and share the same paths.**

## Run the tests

```console
pytest test_vit.py
```

## Licensing
This application is licensed under Apache License 2.0.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under Apache License, Version 2.0  (<sup>*</sup> means additional license information stated following this list):

* dataset/\_\_init\_\_.py
* dataset/customized_randaugment.py<sup>*</sup>
* dataset/dataset.py
* dataset/mixup_utils.py<sup>*</sup>
* dataset/preprocess.py
* models/\_\_init\_\_.py
* models/modules.py<sup>*</sup>
* models/pipeline_model.py
* models/utils.py
* .gitignore
* args.py
* checkpoint.py
* configs.yaml
* finetune.py
* ipu_options.py
* LICENSE
* log.py
* metrics.py
* optimization.py
* pretrain.py
* README.md
* requirements.txt
* run_singlehost.sh
* run_multihosts.sh
* test_vit.py
* validation.py

The following file include code derived from this [file](https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py) which is CC-BY-NC-licensed:
* dataset/mixup_utils.py

The following file include code derived from this [file](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) which is MIT licensed, and from this [file](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py) which is Apache Version 2.0 licensed:
* models/modules.py

External packages:
- `transformers` and `horovod` are licenced under Apache License, Version 2.0
- `pyyaml`, `wandb`, `pytest`, `pytest-pythonpath`, `randaugment` and `attrdict` are licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License
- `pillow` is licensed under the open source HPND License