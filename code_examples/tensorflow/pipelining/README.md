# Graphcore

---
## TensorFlow Pipelining example

This example shows how to use pipelining in TensorFlow to train a very simple model
consisting of just dense layers.

### File structure

* `pipelining.py` The main TensorFlow file showcasing pipelining.
* `README.md` This file.
* `requirements.txt` Required modules for testing.
* `test_pipelining.py` Script for testing this example.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` scripts
   for poplar, and activate a Python virtualenv with the tensorflow-1 wheel from the Poplar SDK installed.

2) Run the script.

    `python pipelining.py`

#### Options

Run pipelining.py with the -h option to list all the command line options.

### Tests

1) Install the requirements.

    `pip install -r requirements.txt`

2) Run the tests.

    `python -m pytest test_pipelining.py`
