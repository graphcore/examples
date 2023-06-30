<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# Poplar Tutorial 2: Using PopLibs

To complement this tutorial, do not hesitate to read through our [Poplar
and PopLibs User
Guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/3.1.0/index.html).

## Setup

In order to run this tutorial on the IPU you will need to have a Poplar
SDK environment enabled (see the [Getting Started Guide for your IPU
system](https://docs.graphcore.ai/en/latest/getting-started.html)).

You will also need a C++ toolchain compatible with the C++11 standard,
build commands in this tutorial use GCC.

## Using PopLibs

Using `tut2_operations/start_here` as your working directory, open
`tut2.cpp` in an editor. This file contains a basic Poplar program
structure similar to that seen in tutorial 1. It creates a graph with a
couple of variables and initialises them. However, this time it includes
some extra headers from the `popops` library:

```c++
#include <popops/codelets.hpp>
#include <popops/ElementWise.hpp>
```

This gives us access to library functions for data manipulation, which
have been highly optimised for IPU devices.

- To use this, you need to add the device-side library code to the graph, so
    that it is loaded when the code is run:

    ```c++
    popops::addCodelets(graph);
    ```

A similar `addCodelets` call is required for each of the PopLibs
libraries you use in your program.

- Compile and run the code (remember to link in the `popops` and `poputil`
    libraries):

    ```console
    $ g++ --std=c++11 tut2.cpp -lpoplar -lpopops -lpoputil -o tut2
    $ ./tut2
    ```

The code doesn't do anything at the moment so let's add an operation
to the graph.

- Add the following, before the engine creation, to extend the program sequence
    with an add operation:

    ```c++
    // Extend program with elementwise add (this will add to the sequence)
    Tensor v3 = popops::add(graph, v1, v2, prog, "Add");

    prog.add(PrintTensor("v3", v3));
    ```

The `popops::add` function extends the sequence `prog` with extra steps
to perform an elementwise add. We've also created a new variable, `v3`,
in the graph for the returned result. So, after the add operation, `v3`
holds the result of adding the elements of `v1` to `v2`.

- Re-compile and re-run the program. You should see the results of the addition:

    ```console
    v3: [
      [5.0000000 4.5000000]
      [4.0000000 3.5000000]
    ]
    ```

- Add code to add `v2` to the result tensor `v3` and print the result.

That is all that is required to use the PopLibs library functions. You
can see the capability of these libraries by browsing the [PopLibs API
documentation](https://docs.graphcore.ai/projects/poplar-api/en/3.1.0/poplibs_api.html)
or the header files in the `include` directories of the Poplar
installation.

## Reshaping and transposing data

When calling libraries to perform operations, there are many ways to
arrange how data is passed to the operation. These are in the
`Tensor.hpp` header file and documented in the [Poplar API
Reference](https://docs.graphcore.ai/projects/poplar-api/en/3.1.0/poplar_api.html#poplar-tensor-hpp).

In tutorial 1 we used slicing, but there are also functions for
reshaping and transposing data.

- Add the following code to add `v1` to the transpose of the 2x2 matrix `v2`:

    ```c++
    // Example element wise addition using a transposed view of the data
    Tensor v5 = popops::add(graph, v1, v2.transpose(), prog, "Add");
    prog.add(PrintTensor("v5", v5));
    ```

- Re-compile and re-run the program to see the result.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
