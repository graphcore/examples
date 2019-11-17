# BERT Testing

### File structure

* `unit/` - Testing the functional correctness of components of the model vs a PyTorch version. The implementation of the component will be the best performing, custom_op or ONNX.
* `regression` - Testing the performance, accuracy and throughput, of the model when training.

### How to run the tests demo

#### Prepare the environment

##### PopART

Follow the instructions in the root README.

##### Poplar

Follow the instructions in the root README.

##### Python

Follow the instructions in the root README and Install pytest

```
pip install pytest
```

#### Run the test

###### Unit tests

```
pytest tests/unit
```

###### Regression tests

```
pytest tests/regression
```
