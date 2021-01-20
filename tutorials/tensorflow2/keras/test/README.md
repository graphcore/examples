This directory contains tests for the `demo.py` and `main.py` files in the `completed_example` directory.

The tests use `pytest` and helper utilities in the `utils` directory in the root of this repository.

To run the tests, create a python3 virtual environment and install the requirements:

```
pip3 install -r requirements.txt
```

Make sure you have:
* Installed and enabled Poplar
* Installed the Graphcore port of TensorFlow 2
as per the README in the parent directory.

Then run the tests using pytest:

```
python -m pytest
```
