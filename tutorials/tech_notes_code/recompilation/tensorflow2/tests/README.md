<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
This directory contains tests for the code examples in the tutorial.

The tests simply run each program and check that they run successfully, as is expected.

The tests use the `pytest` module.To run the tests, create a Python 3 virtual environment with Poplar and TensorFlow 2 for IPU installed.

Then you can install the specific requirements of the tests and run them using the following command lines:

```
pip3 install -r requirements.txt
python -m pytest
```
