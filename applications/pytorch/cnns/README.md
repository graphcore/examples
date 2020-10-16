Graphcore
---
## PyTorch CNN reference application on IPUs

Run inference or training on Graphcore IPUs using PyTorch.
Look in the subfolders inference and training to learn how to use these scripts.

### Folder structure

* `inference` Reference application to run inference.
* `train` Reference application to run training.
* `models` Model definition shared between inference and training.
* `utils` Contains common code such as logging.
* `README.md` This file.
* `requirements.txt` Required python packages.

### Installation instructions

1. Prepare the PopTorch environment. Install the poplar-sdk following the Getting Started guide for your IPU system. Make sure to run the
   `enable.sh` script and activate a Python virtualenv with PopTorch installed.
2. Install additional python packages specified in requirements.txt
   $ pip3 install -r requirements.txt
