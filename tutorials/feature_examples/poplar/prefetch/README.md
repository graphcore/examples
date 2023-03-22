<!-- Copyright (c) 2019 Graphcore Ltd. All rights reserved. -->
# Poplar Prefetching Callbacks Example

The example program `prefetch.cpp` defines a poplar program that runs multiple times.
The program consists on a repeated sequence:

    Copy from input stream to tensor.
    Modify tensor value
    Copy tensor to output stream.

The input stream is connected to a prefetch-able callback and all necessary
engine options are set so that the prefetch function is used.

All callback functions print a message to standard output to see what is going
on.

## File structure

* `prefetch.cpp` The main Poplar code example.
* `Makefile` A simple Makefile for building the example.
* `required_apt_packages.txt` A requirements file for this example.
* `test_prefetch.py` The testing script for this example.
* `requirements.txt` A requirements file to run testing.
* `README.md` This file.

## How to use this example

1) Prepare the environment.

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
    script for Poplar.

   The example also uses boost::program_options. You can install boost via your package manager.

2) Build and run the example.

```
make
./prefetch --help
```
