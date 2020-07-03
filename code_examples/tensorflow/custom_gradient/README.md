# Graphcore

---
## Creating a Tensorflow Custom Op with Gradient

Creates a custom op (a batched dot product) defining both the
forward op and its gradient in Poplar code. Uses the custom op
in a simple logistic regression optimisation program which checks
the results with the custom op match those from the built-in op.

### File structure

* `Makefile` - Simple Makefile that builds the Poplar shared object.
* `product.cpp` - Poplar code that describes the forward and grad ops.
* `tf_regression.py` Tensorflow program that uses the custom op to do logistic regression.
* `test.py` Script for testing this example.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the `poplar-sdk` (version 1.2 or later) following the README provided. Make sure to source the `enable.sh`
   scripts for poplar and gc_drivers.

2) Build the custom op and then run the python code:

```
make
python3 regression.py
```
