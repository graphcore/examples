Graphcore
---
## Pytorch CNN reference application on IPUs

Run inference or training on Graphcore IPUs using Pytorch.
Look in the subfolder inference to learn how to use these scripts.

### Folder structure

* `inference` Reference application to run inference.
* `models` Model definition.
* `README.md` This file.
* `requirements.txt` Required python packages.

### Installation instructions

1. Prepare the Poptorch environment. Install the poplar-sdk following the README provided. Make sure to run the
   `enable.sh` scripts and activate a Python virtualenv with poptorch installed.
2. Install additional python packages specified in requirements.txt
   $ pip3 install -r requirements.txt