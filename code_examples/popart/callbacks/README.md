# Graphcore

---
## PopART 

This example creates a simple computation graph and uses callbacks to feed data and
retrieve the results. Time between host-device transfer and receipt of the
result on the host is computed and displayed for a range of different data sizes.

### File structure

* `callbacks.py` The PopART example code.
* `test_callbacks.py` Test file.
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
    scripts for poplar and PopART.

2) Run the example. Note that the PopART Python API only supports Python 3.

        python3 callbacks.py
