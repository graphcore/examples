# Graphcore
---
## DeviceConnectionType example

This code example demonstrates how to use `set_ipu_connection_type` to control if and when the IPU device is acquired.

|Mode          |Description                                                          |
|--------------|---------------------------------------------------------------------|
|ALWAYS        | indicates that the system will attach when configuring the device   |
|ON_DEMAND     | will defer connection to when the IPU is needed.                    |
|NEVER         | will never try to attach to a device. Used when compiling offline.  |

These options can be used to change relative ordering of compilation versus IPU acquisition.
If ALWAYS is selected (default) then the IPU device will always be acquired before compilation.
If ON_DEMAND is selected then the IPU device will only be acquired once it is required which can be after compilation.
If NEVER is selected then the IPU device is never acquired.

### File structure

* `connection_type.py` Minimal example.
* `README.md` This file.
* `requirements.txt` Required packages for the tests.
* `test_connection_type.py` pytest tests

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system.
   Make sure to run the enable.sh script for Poplar and activate a Python virtualenv with gc_tensorflow installed.

2) Run the script.

   ```
   python3 connection_type.py
   ```

3) Run the tests.

   ```
   pip3 install -r requirements.txt
   python3 -m pytest
   ```

   The test runs for each mode and checks:
    * The resultant tensor is valid if returned (not expected for NEVER).
    * The regular stderr trace for evidence that compilation and device attachment occur in the expected order.

### Usage (see connection_type --help)

```
usage: connection_type.py [-h] [--connection_type {ALWAYS,ON_DEMAND,NEVER}]

optional arguments:
  -h, --help            show this help message and exit
  --connection_type {ALWAYS,ON_DEMAND,NEVER}
                        Specify connection type
```
