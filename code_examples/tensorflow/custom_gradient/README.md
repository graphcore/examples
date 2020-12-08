# Graphcore

---
## Creating a TensorFlow Custom Op with Gradient

Creates a custom op (a batched dot product) defining both the
forward op and its gradient in Poplar code. Uses the custom op
in a simple logistic regression optimisation program which checks
the results with the custom op match those from the built-in op.

### File structure

* `Makefile` Simple Makefile that builds the Poplar shared object.
* `product.cpp` Poplar code that describes the forward and grad ops.
* `regression.py` TensorFlow program that uses the custom op to do logistic regression.
* `requirements.txt` Required packages.
* `test_regression.py` Script for testing this example.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to source the `enable.sh` script for Poplar.

2) Build the custom op and then run the Python code:

```
make
python3 regression.py
```
