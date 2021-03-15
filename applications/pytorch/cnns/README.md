PyTorch CNN reference application on IPUs
---

## Overview

Run CNN inference or training on Graphcore IPUs using PyTorch.
Look in the subfolders inference, datasets and train to learn how to use these scripts.

The following models are supported:
1. ResNet18-34-50-101-152 (`--model resnet18`)
2. MobileNet (`--model mobilenet`)
3. EfficientNet-B0..7 (`--model efficientnetb0`)
4. ResNext50-101 (`--model resnext50`)

### Folder structure

* `inference` Reference application to run inference.
* `train` Reference application to run training.
* `models` Models definition shared between inference and training.
* `utils` Contains common code such as logging.
* `data` Contains the downloaded datasets.
* `datasets` Contains the dataset handling code, shared between inference and training. Also contains dataset converter, which converts imagenet data to webdataset format.
* `README.md` This file.
* `requirements.txt` Required Python packages.

### Installation instructions

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   Getting Started guide for your IPU system. Make sure to source the
   `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv with PopTorch installed.
   
2. Install dependencies and download sample images for inference.
   Install system package dependencies: `sudo apt-get install libjpeg-turbo8-dev libffi-dev`

   Then install pip & data dependencies using the provided makefile:
   ```
   $ make
   ```
The MakeFile includes four options.

1. `make get_data` which downloads a set of sample images for inference (data/get_data.sh)
2. `make install` which installs dependencies/requirements
3. `make test` which runs pytest for this example with 10 parallel threads
4. `make all` calls `install` and `get_data`

The commands executed by make install are:

```
	pip install -r requirements.txt
	pip install --force-reinstall --no-binary :all: horovod==0.21.0 
	pip uninstall pillow -y
	CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd
```

Note: pretrained models are used for inference. The weights are downloaded from the following places:
* The official PyTorch model storage
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) package's GitHub repository

### License

EfficientNet PyTorch package, which is applied under MIT license.
[https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE)

Pillow SIMD is licensed under the Python Imaging License:
[https://github.com/uploadcare/pillow-simd/blob/simd/master/LICENSE](https://github.com/uploadcare/pillow-simd/blob/simd/master/LICENSE)

Libjpeg-turbo is licensed under BSD-style open-source licenses:
[https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/LICENSE.md](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/LICENSE.md)

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS", AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
