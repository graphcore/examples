To run the tests, create a Python 3 virtual environment with Poplar and PopTorch for IPU installed. Then you can install the specific requirements of the tests and run them by calling:

```
pip3 install -r requirements.txt
python -m pytest
```

The tests can also be run in parallel by calling:

```
python -m pytest -n4 --forked
```
