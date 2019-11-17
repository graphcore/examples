# Graphcore

## Building a Custom Operator using PopART

This directory contains a C++ implementation of a custom operator for PopART.

The operator in question will take a tensor as input, then perform an element-wise
cube operation, returning the result.

### File structure

* `cube_fn_custom_op.cpp` The C++ implementation of the operation  
* `Makefile` A pre-configured Makefile to build the example
* `README.md` This file.

### How to use this demo

1) Prepare the environment.

  Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
  scripts for poplar, gc_drivers (if running on hardware) and popart.

2) Build the example

  With your environment fully configured as above, all that remains is to build the example:

	$ make

3) You can then run the example by running the following:

	$ ./build/cube_fn_custom_op

## More Information
This example creates an operation with both forward and backward parts, which
cubes each element of an input tensor.

The input tensor is read from the command-line arguments and converted ready for
use in the model.

By default, the model runs on the CPU; however it can be pushed onto an IPU device
by providing the `--ipu` flag.

Having built the application, it can be run as follows:

	Usage: cube_fn_custom_op [--help] [--ipu] <tensor_val_0> <tensor_val_1>...
	Options:
	  --help	Display this message
	  --ipu		Run the example on an available IPU (otherwise it runs on CPU)
	
	Example usages:
	  $ cube_fn_custom_op 0.3 2.7 1.2 5
	  $ cube_fn_custom_op --ipu 0.3 2.7 1.2 5
