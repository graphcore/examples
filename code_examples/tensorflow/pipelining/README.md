# Graphcore

---
## TensorFlow Pipelining example

This example shows how to use pipelining in TensorFlow to train a very simple model
consisting of just dense layers.

### File structure

* `pipelining.py` The main TensorFlow file showcasing pipelining.
* `README.md` This file.
* `test_pipelining.py` Script for testing this example.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the poplar-sdk following the README provided. Make sure to source the `enable.sh` scripts
   for poplar and gc_drivers, and activate a Python virtualenv with gc_tensorflow installed.

2) Run the script.

    `python pipelining.py`

#### Options

Run pipelining.py with -h option to list all the command line options.
