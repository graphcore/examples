# Graphcore
---
## popART Synthetic Benchmark Module

A python module to benchmark popART graphs in a while loop construct with **no** host to device data feed.
They are intended to give the raw items/sec without considering data transfer or TF session
management overhead.

### File structure
* `benchmark.py` Python module for running core graphs on device in a loop.
* `README.md` This file.


