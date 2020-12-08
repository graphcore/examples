# Graphcore
---
## Replication examples
IPU replication allows you to replicate the same graph onto N different IPUs. These N IPUs can then be used to process training data in parallel.
IPU replication requires the use of IPU Infeed queues and ipu_compile
### File structure

* `simple_replication.py` Minimal example of IPU graph replication
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the Getting Started guide for your IPU system.
   Make sure to run the enable.sh script and activate a Python virtualenv with gc_tensorflow installed.

2) Run the script.

   $ python simple_replication.py
