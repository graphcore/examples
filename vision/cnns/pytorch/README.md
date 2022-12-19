# CNNs (Pytorch)
Deep CNN residual learning models for image recognition and classification, optimised for Graphcore's IPU.

| Framework | domain | Model | Datasets | Tasks| Training| Inference |
|-------------|-|------|-------|-------|-------|---|
| Pytorch | Vision | CNNs | ImageNet LSVRC 2012 | Image recognition, Image classification | ✅  | ✅ |


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

5. Install the Python requirements and pillow-SIMD:
```bash
make install
```

6. Install turbojpeg:
```bash
make install-turbojpeg
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

### Inference from pre-trained weights
Pretrained models are used for inference. The weights are downloaded from the following places:
* The official PyTorch model storage
* [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) package's model storage

### Troubleshooting

If Triton server tests fails with such an error:

```bash
[model_runtime:cpp] [error] Error in model_runtime/source/Executable.cpp:38:Failed to deserialize XXX : Error reading executable - package hash (YYY) differs from poplar hash (ZZZ)
```

This mean that models were generated and saved with different version of SDK and needs to be recreated. Please remove `tests_serial/tritonserver/test_environment_ready.lock` and rerun tests.

```bash
Failed: Failed to download and/or compile Triton Server!
```

Most probably some system packages are missing, ensure that all packages listed in `required_apt_packages.txt` are installed. Also refer to Triton Server build log file. After fixing error remove `../../../utils/triton_server/triton_environment_is_prepared.lock` and rerun tests.

## License

ModelZoo.pytorch is licensed under MIT License:
[https://github.com/PistonY/ModelZoo.pytorch/blob/master/LICENSE](https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/LICENSE)

PyTorch Image Models is licensed under the Apache License 2.0:
[https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/LICENSE](https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/LICENSE)

Pillow SIMD is licensed under the Python Imaging License:
[https://github.com/uploadcare/pillow-simd/blob/v7.0.0.post3/LICENSE](https://github.com/uploadcare/pillow-simd/blob/v7.0.0.post3/LICENSE)

Libjpeg-turbo is licensed under BSD-style open-source licenses:
[https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.0/LICENSE.md](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.0/LICENSE.md)

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
