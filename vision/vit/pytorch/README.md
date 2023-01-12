# ViT (Vision Transformer)
Vision Transformer for image recognition, optimised for Graphcore's IPU.  Based on the models provided by the [`transformers`](https://github.com/huggingface/transformers) library and from [jeonsworld](https://github.com/jeonsworld/ViT-pytorch)

Run our ViT on Paperspace.
<br>
[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://ipu.dev/3W2Ru39)

| Framework | domain | Model | Datasets | Tasks| Training| Inference | Reference |
|-------------|-|------|-------|-------|-------|---|-------|
| Pytorch | Vision | ViT | ImageNet LSVRC 2012, CIFAR-10 | Image recognition | ✅  | ✅ | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) |


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
### ImageNet LSVRC 2012
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
### Pretraining
In pre-training Imagenet1k, `micro_batch_size` is set to 8, and all other parameters are tuned to reach a validation accuracy that is higher than 74.79% (released in [google official repository](https://github.com/google-research/vision_transformer/issues/62#issuecomment-888993463)). To achieve maximum throughput, `micro_batch_size` can be set to 14, then hyperparameters tuning is required to reach this validation accuracy.

### Fine-Tuning
You can run ViT fine-tuning on either CIFAR10 or ImageNet1k datasets. The default pretrained checkpoint is loaded from [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k). The commands for fine-tuning are:

CIFAR10 fine-tuning:
```bash
python finetune.py --config b16_cifar10
```

Once the fine-tuning finishes, you can validate:
```bash
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
### Employing automatic loss scaling (ALS) for half precision training

ALS is a feature in the Poplar SDK which brings stability to training large models in half precision, specially when gradient accumulation and reduction across replicas also happen in half precision.

NB. This feature expects the `poptorch` training option `accumulationAndReplicationReductionType` to be set to `poptorch.ReductionType.Mean`, and for accumulation by the optimizer to be done in half precision (using `accum_type=torch.float16` when instantiating the optimizer), or else it may lead to unexpected behaviour.

To employ ALS for ImageNet1k fine-tuning on a POD16, the following command can be used:

```bash
python3 finetune.py --config b16_imagenet1k_ALS
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
