<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
### Tests

This directory contains integration tests.

To run the tests, source the `enable.sh` script for Poplar and activate a
Python 3 virtual environment with a TensorFlow 2 wheel appropriate for your system
installed. You can then use `pip3` to install the required packages:

```
python3 -m pip install -r requirements.txt
```

Use this command to run the tests:

```
python3 -m pytest
```
