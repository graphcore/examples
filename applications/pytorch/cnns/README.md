Graphcore
---

## PyTorch CNN reference application on IPUs

Run inference or training on Graphcore IPUs using PyTorch.
Look in the subfolders inference and training to learn how to use these scripts.

### Folder structure

* `inference` Reference application to run inference.
* `train` Reference application to run training.
* `models` Models definition shared between inference and training.
* `utils` Contains common code such as logging.
* `data` Contains the datasets, shared between inference and training.
* `README.md` This file.
* `requirements.txt` Required python packages.

### Installation instructions

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   Getting Started guide for your IPU system. Make sure to run the
   `enable.sh` scripts and activate a Python virtualenv with PopTorch installed.
2. Install additional python packages specified in requirements.txt
   $ pip3 install -r requirements.txt

Note: the code uses pretrained models. The weights are downloaded from the following places:
* The official PyTorch model storage
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) package's GitHub repository
