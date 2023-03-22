# SWIN (Shifted Windows Vision Transformers)
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, optimised for Graphcore's IPU. Based on the models provided by [SWIN](https://github.com/microsoft/Swin-Transformer)

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | Vision | SWIN | ImageNet LSVRC 2012 | Image recognition, Image classification | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required  | <p style="text-align: center;">❌ | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf) |

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

3. Additionally, enable PopART with:
```bash
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```

4. Navigate to this example's root directory

6. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

5. Build the custom ops:
```bash
cd custom_ops
make all
```


More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).

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


## Custom training

### Precision options
We can train with fp32.32 and fp16.16 (input.weights). We can use the PRECISION parameter in config to modify the required precision

Accuracy is as follows:

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
```bash
wget https://github.com/SwinTransformer/storage/releases/download/latest_stable/swin_base_patch4_window7_224_22k.pth
```
and large 224 model pretrained on imagenet21k by:
```bash
wget https://github.com/SwinTransformer/storage/releases/download/latest_stable/swin_large_patch4_window7_224_22k.pth
```

Once the training finishes, you can validate accuracy:
```bash
python3 validate.py --cfg YOUR_CONFIG --checkpoint /path/to/checkpoint.pth
```


## License
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
