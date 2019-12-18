# Graphcore

---
## Creating a simple Tensorflow Custom Op

Creates a simple custom op that adds two vectors of arbitrary size. The op
is created in Poplar using a custom vertex. This simple example does not show
how to create the corresponding gradient op but the mechanism for the grad-op
is similar.

### File structure

* `custom_codelet.cpp` Custom codelet used in the custom op.
* `Makefile` - Simple Makefile that builds the Poplar code and codelet (gp file).
* `poplar_code.cpp` - Poplar code that builds the custom op.
* `tf_code.py` Tensorflow program that uses the custom op. 
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
    scripts for poplar, gc_drivers (if running on hardware).

2) Build the custom op and then run the python code.

```
make
python3 tf_code.py
```
