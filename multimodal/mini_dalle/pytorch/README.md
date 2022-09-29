Pytorch mini DALL-E training example
---

Implementation of mini DALL-E model in PyTorch for IPU. This example is based on the models provided by the [`CompVis`](https://github.com/CompVis/taming-transformers) library and from [lucidrains](https://github.com/lucidrains/DALLE-pytorch). The mini DALL-E model is based on the original paper [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092).

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

Download the datasets from MS COCO website and unzip the downloaded files:

```console
mkdir -p ./data/COCO
wget http://images.cocodataset.org/zips/train2017.zip -P ./data/COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/COCO
cd ./data/COCO
unzip ./data/COCO/train2017.zip
unzip ./data/COCO/annotations_trainval2017.zip
```

The MS COCO 2017 training set contains about 118K image-text pairs. It is approximately 19GB for the training set. The annotations in this dataset along with this website belong to the COCO Consortium and are [https://creativecommons.org/licenses/by/4.0/legalcode](licensed under a Creative Commons Attribution 4.0 License). The COCO Consortium does not own the copyright of the images. Use of the images must abide by the [https://www.flickr.com/creativecommons/](Flickr Terms of Use). The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset. Full terms and conditions and more information are available on the [https://cocodataset.org/#termsofuse](Terms of Use)

After download the dataset, we extract captions to txt file from captions\_train2017.json, in the meantime fix some format errors in captions:

```console
python process_captions.py
```

The captions files are generated in ./data/COCO/train2017\_captions.

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

Setup your environment as explained above. You can run mini DALL-E training on MS COCO 2017 datasets.

For training on M2000:
```console
python train.py --config L16
```

For training on POD16:
```console
python train.py --config L16_POD16
```

For training on POD64:
```console
python train.py --config L16_POD64
```

Afterwards run text-image generation, for example:
```console
python generate.py --dalle_path ./output/ckpt/dalle_799.pt --text "A plate of food has potatoes and fruit." --outputs_dir ./output --bpe_path models/bpe/bpe_yttm_vocab.txt
```

## Run the unit test

```console
pytest tests/cpu_ipu_test.py
```

## Licensing
This application is licensed under MIT license.
Please see the LICENSE file in this directory for full details of the license conditions.

The following files are created by Graphcore and are licensed under MIT License  (<sup>*</sup> means additional license information stated following this list):
* configs.yaml
* README.md
* requirements.txt
* run_train.sh
* log.py
* tests/cpu_ipu_test.py
* data/process_trainset.py

The following files include code derived from this [repo](https://github.com/lucidrains/DALLE-pytorch) which uses MIT license:
* args.py
* train.py
* generate.py
* models/__init__.py
* models/attention.py
* models/dalle.py
* models/loader.py
* models/tokenizer.py
* models/transformer.py
* models/vae.py
* bpe/bpe_simple_vocab_16e6.txt.gz
* bpe/bpe_yttm_vocab.txt

External packages:
- `taming-transformer`, `youtokentome`, `pyyaml`, `wandb`, `pytest`, `pytest-pythonpath` are licensed under MIT License
- `torchvision` is licensed under BSD 3-Clause License
