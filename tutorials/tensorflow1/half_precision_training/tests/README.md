This directory contains tests for the code examples provided with the tutorial. Each suite of tests runs each program and checks that `Program ran successfully` is printed at the very end for different command line options, as is expected. They then try some illegal command-line option sets and check that exceptions are thrown. The tests use the `pytest` module.

To run the tests, create a Python 3 virtual environment with Poplar and TensorFlow 1 for IPU installed. Then you can install the specific requirements of the tests and run them by calling:

```
pip3 install -r requirements.txt
python -m pytest
```

from the command line.

