<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# Using Custom Operators in PyTorch

This example shows how to use a custom operator in the PopTorch
framework on the IPU.

To be used in PopTorch, a custom operator must first be implemented as
a PopART custom operator, then be made available to PopTorch.

This example shows the process of loading in a custom operator and using it in a simple
model creation and training process. This is shown with a CNN using the LeakyReLU custom
operator as an activation function, on the FashionMNIST dataset.

For more information on custom operators in PopTorch, please refer to the
[Creating custom operators section of our PyTorch for the IPU User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#creating-custom-ops).

## File structure

* `cube_op_example/tests/requirements.txt` Required packages to test the cube op.
* `cube_op_example/tests/test_cube_op.py` A pytest script to test the operator.
* `cube_op_example/cube_fn_custom_op.cpp` The C++ implementation of the Cube operation.
* `cube_op_example/Makefile` A pre-configured Makefile to build the Cube operator example.
* `leaky_relu_example/leaky_relu_custom_op.cpp` The C++ implementation of the Leaky ReLU operation.
* `leaky_relu_example/Makefile` A pre-configured Makefile to build the Leaky ReLU operator example.
* `leaky_relu_example/requirements.txt` Required packages to test the Leaky ReLU op.
* `leaky_relu_example/run_leaky_relu.py` A script to run a simple model that uses the Leaky ReLU op.
* `leaky_relu_example/test_lrelu.py` A pytest script to test the operator.
* `tests/test_poptorch_custom_op.py` Script for testing this example.
* `tests/requirements.txt` Required packages to test the PopTorch custom ops.
* `poptorch_custom_op.py` PopTorch CNN program using the custom operator.
* `requirements.txt` Required packages to run the Python file.

## Example custom operators

### Cube

The Cube operator takes a tensor as input, then performs an element-wise
cube operation, returning the result. This contains a main function that builds and runs
a simple model using this operator in C++.

### Leaky ReLU

The leaky rectified linear unit (Leaky ReLU) takes a tensor as input, then returns `x` for any
element `x >= 0` and `x * alpha` for any element `x < 0`, where `alpha` is provided as a scalar
attribute to the operator. This contains a Python script that demonstrates how to load the operator into the
Python runtime, build and then execute a model that uses a custom operator with the Python API.

### Building operator examples

1) Prepare the environment:
    - Ensure the Poplar SDK is installed (follow the instructions in the Getting
    Started guide for your IPU system: <https://docs.graphcore.ai/en/latest/getting-started.html>.
    - Install the requirements for the Python program with:
        ```
        cd cube_op_example # or "cd leaky_relu_example"
        python3 -m pip install -r requirements.txt
        ```
2) Build the custom operator:
    ```
    make
    ```

This creates a sub directory `build` which contains an executable `cube_fn_custom_op` (for [cube_op_example](cube_op_example)) or `custom_ops.so` (for [leaky_relu_example](leaky_relu_example)).

### Running the Cube operator example

Having built the Cube operator application, it can be run as follows:

```
Usage: cube_fn_custom_op [--help] [--ipu] <tensor_val_0> <tensor_val_1>...
Options:
    --help  Display this message
    --ipu   Run the example on an available IPU (otherwise it runs on CPU)

Example uses:
    $ cube_fn_custom_op 0.3 2.7 1.2 5
    $ cube_fn_custom_op --ipu 0.3 2.7 1.2 5
```

The input tensor is read from the command-line arguments and converted ready for
use in the model.

By default, the Cube operator model runs on the CPU; however it can be pushed onto an IPU device
by providing the `--ipu` flag.

#### Running the Leaky ReLU operator example

Having built the Leaky ReLU op, the popart script can be run as follows:

```
Usage: python3 run_leaky_relu.py [--help] [--alpha <alpha_value>] [--ipu] <tensor_val_0> <tensor_val_1>...
    Options:
        --help   Display this message
        --alpha  Set the value of the alpha attribute (default = 0.02)
        --ipu    Run the example on an available IPU (otherwise it runs on IPU Model)

    Example uses:
        $ python3 run_leaky_relu.py 0.3 -2.7 1.2 -5
        $ python3 run_leaky_relu.py --alpha 0.01 0.3 -2.7 1.2 -5
```

The input tensor is read from the command-line arguments and converted ready for
use in the model.

The alpha attribute has a default value of `0.02` but can be set using the `--alpha` option.

By default, the Leaky ReLU model runs on the IPU Model; however it can be pushed onto an IPU device
by providing the `--ipu` flag.

## How to run the example

1) Prepare the environment:
    - Ensure the Poplar SDK is installed (follow the instructions in the Getting
    Started guide for your IPU system: <https://docs.graphcore.ai/en/latest/getting-started.html>.
    - Install the requirements for the Python program with:
       ```
       python3 -m pip install -r requirements.txt
       ```
2) Build the Leaky ReLU as described in [Building operator examples](#building-operator-examples) (leave the custom operator's directory afterward with `cd ..`).
3) Run the Python example:
    ```
    python3 poptorch_custom_op.py
    ```
