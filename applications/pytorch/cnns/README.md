PyTorch CNN reference application on IPUs
---

## Overview

Run CNN inference or training on Graphcore IPUs using PyTorch.
Look in the subfolders inference, datasets and train to learn how to use these scripts.

The following models are supported:
1. ResNet50 (`--model resnet50`)
2. EfficientNet-B0, B4 (`--model efficientnet-b0`)

To see the list of available models use `--help`

### Folder structure

* `inference` Reference application to run inference.
* `train` Reference application to run training.
* `models` Models definition shared between inference and training.
* `utils` Contains common code such as logging.
* `data` Contains the downloaded datasets.
* `datasets` Contains the dataset handling code, shared between inference and training. Also contains dataset converter, which converts imagenet data to webdataset format.
* `tests` Contains the unit tests
* `tests_serial` Contains the popdist unit tests. Runs serially.
* `README.md` This file.
* `requirements.txt` Required Python packages.
* `conftest.py` Test helper functions.
* `required_apt_packages.txt` Required system packages.
* `MAKEFILE` Commands to install dependencies

### Installation instructions

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   Getting Started guide for your IPU system. Make sure to source the
   `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.

2. Install the apt dependencies for Ubuntu 18.04 (requires admin privileges):

```console
sudo apt install $(< required_apt_packages.txt)
```

3. Install the pip dependencies and download sample images for inference. These installations are handled by running the provided makefile:

   ```console
   make
   ```

The MakeFile includes four options:

1. `make get_data` which downloads a set of sample images for inference (data/get_data.sh)
2. `make install` which installs dependencies/requirements
3. `make test` which runs pytest for this example with 10 parallel threads
4. `make all` calls `install` and `get_data`

The commands executed by make install are:

```console
	pip install -r requirements.txt
	pip uninstall pillow -y
	CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd
```

Note: pretrained models are used for inference. The weights are downloaded from the following places:
* The official PyTorch model storage
* [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) package's model storage


### Running the tests

After following installation instructions run:

```console
pytest
```


### License

ModelZoo.pytorch is licensed under MIT License:
[https://github.com/PistonY/ModelZoo.pytorch/blob/master/LICENSE](https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/LICENSE)

PyTorch Image Models is licensed under the Apache License 2.0:
[https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/LICENSE](https://github.com/rwightman/pytorch-image-models/blob/v0.4.12/LICENSE)

Pillow SIMD is licensed under the Python Imaging License:
[https://github.com/uploadcare/pillow-simd/blob/v7.0.0.post3/LICENSE](https://github.com/uploadcare/pillow-simd/blob/v7.0.0.post3/LICENSE)

Libjpeg-turbo is licensed under BSD-style open-source licenses:
[https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.0/LICENSE.md](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/2.1.0/LICENSE.md)

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###  Changelog

#### September 2021
* Single IPU ResNet50 with batchnorm. Model is recomputed.
* Webdataset improvements: Caching the dataset in the memory. Option for determining the image quality.
* EfficientNet model changed to PyTorch Image Models implementation.
