# Graphcore

## Building custom operators using PopART

This directory contains two example implementations of custom operators for PopART.

Both examples create an operation definition with forward and backward parts, although only 
inference is demonstrated in the executables for simplicity.

### File structure

* `cube_op_example/cube_fn_custom_op.cpp` The C++ implementation of the cube operation  
* `cube_op_example/Makefile` A pre-configured Makefile to build the cube op example
* `leaky_relu_example/leaky_relu_custom_op.cpp` The C++ implementation of the leaky relu operation
* `leaky_relu_example/Makefile` A pre-configured Makefile to build the leaky relu op example
* `leaky_relu_example/run_leaky_relu.py` A script to run a simple model that uses the leaky relu op
* `leaky_relu_example/test_lrelu.py` A pytest script to test the operator
* `README.md` This file.


## Cube Op

The cube operator takes a tensor as input, then performs an element-wise
cube operation, returning the result. This contains a main function that builds and runs 
a simple model using this op in C++.

### Building the Cube Op example

1) Prepare the environment.

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for poplar and popart.

2) Build cube operator example
	
With your environment fully configured as above, all that remains is to build the example:

	$ cd cube_op_example
	$ make

This creates a sub directory `build` which contains an executable `cube_fn_custom_op`.

### Running the Cube Op example

Having built the cube op application, it can be run as follows:

	Usage: cube_fn_custom_op [--help] [--ipu] <tensor_val_0> <tensor_val_1>...
	Options:
	  --help	Display this message
	  --ipu		Run the example on an available IPU (otherwise it runs on CPU)
	
	Example usages:
	  $ cube_fn_custom_op 0.3 2.7 1.2 5
	  $ cube_fn_custom_op --ipu 0.3 2.7 1.2 5

The input tensor is read from the command-line arguments and converted ready for
use in the model.

By default, the cube op model runs on the CPU; however it can be pushed onto an IPU device
by providing the `--ipu` flag. 


## Leaky ReLU Op

The leaky rectified linear unit (Leaky ReLU) takes a tensor as input, then returns `x` for any 
element `x >= 0` and `x * alpha` for any element `x < 0`, where `alpha` is provided as a scalar 
attribute to the operator. This contains a python script that demonstrates how to load the op into the 
python runtime, build and then execute a model that uses a custom op with the python API.

### Building the Leaky ReLU op example

1) Prepare the environment.

  Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
  scripts for poplar and popart.

2) Build leaky rely operator example
	
With your environment fully configured as above, all that remains is to build the example:

	$ cd leaky_relu_example
	$ make

This creates a sub directory `build` which contains a C++ shared object library `custom_ops.so`.

### Running the Leaky ReLU Op example

Having built the leaky relu op, the popart script can be run as follows:

	Usage: python3 run_leaky_relu.py [--help] [--alpha <alpha_value>] [--ipu] <tensor_val_0> <tensor_val_1>...
		Options:
		  --help	Display this message
		  --alpha	Set the value of the alpha attribute (default = 0.02)
		  --ipu		Run the example on an available IPU (otherwise it runs on IPU Model)

		Example usages:
		  $ python3 run_leaky_relu.py 0.3 -2.7 1.2 -5
		  $ python3 run_leaky_relu.py --alpha 0.01 0.3 -2.7 1.2 -5

The input tensor is read from the command-line arguments and converted ready for
use in the model.

The alpha attribute has a default value of `0.02` but can be set using the `--alpha` option.

By default, the Leaky ReLU model runs on the IPU Model; however it can be pushed onto an IPU device
by providing the `--ipu` flag.
