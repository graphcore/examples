# Graphcore

---
## Creating a simple TensorFlow Custom Op

Creates a simple custom op that adds two vectors of arbitrary size. The op
is created in Poplar using a custom vertex. This simple example does not show
how to create the corresponding gradient op but the mechanism for the grad-op
is similar.

### File structure

* `custom_codelet.cpp` Custom codelet used in the custom op.
* `Makefile` Simple Makefile that builds the Poplar code and codelet (gp file).
* `poplar_code.cpp` Poplar code that builds the custom op.
* `requirements.txt` Required packages.
* `tf_code.py` TensorFlow program that uses the custom op.
* `test_custom_op.py` Script for testing this example.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` script for Poplar.

2) Build the custom op and then run the Python code.

```
make
python3 tf_code.py
```
