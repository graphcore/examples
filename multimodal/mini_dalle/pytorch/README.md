# Mini DALL-E
Mini DALL-E for text-to-image generation, based on the models provided by the [`CompVis`](https://github.com/CompVis/taming-transformers) library and the [lucidrains repo](https://github.com/lucidrains/DALLE-pytorch)

| Framework | domain | Model | Datasets | Tasks| Training| Inference | Reference |
|-------------|-|------|-------|-------|-------|---|-------|
| Pytorch | Multimodal | Mini DALL-E | | Text-to-image generation | ✅  | ✅ | [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) |


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


## Dataset setup

### COCO 2017
Download the COCO 2017 dataset from [the source](http://images.cocodataset.org/zips/) or [via kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset), or via the script we provide:
```bash
bash utils/download_coco_dataset.sh
```

Additionally, also download  and unzip the labels:
```bash
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d '<dataset path>' && rm coco2017labels.zip
```

Disk space required: 26G

```
.
├── LICENSE
├── README.txt
├── annotations
├── images
├── labels
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt

3 directories, 7 files
```

The annotations in this dataset along with this website belong to the COCO Consortium and are [https://creativecommons.org/licenses/by/4.0/legalcode](licensed under a Creative Commons Attribution 4.0 License). The COCO Consortium does not own the copyright of the images. Use of the images must abide by the [https://www.flickr.com/creativecommons/](Flickr Terms of Use). The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset. Full terms and conditions and more information are available on the [https://cocodataset.org/#termsofuse](Terms of Use)

Then some preprocessing is required:
```bash
python process_captions.py
```

The captions files are generated in ./data/COCO/train2017\_captions.


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

### Text-to-image generation
To run text-image generation after training, with a checkpoint file (dalle_779.pt here):
```bash
python generate.py --dalle_path ./output/ckpt/dalle_799.pt --text "A plate of food has potatoes and fruit." --outputs_dir ./output --bpe_path models/bpe/bpe_yttm_vocab.txt
```

for example.

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
