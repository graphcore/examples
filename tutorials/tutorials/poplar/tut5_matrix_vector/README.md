<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# Poplar Tutorial 5: Matrix-vector Multiplication

In this tutorial you will:

- build a Poplar function and vertex to multiply a matrix by a vector, we
    recommend completing [Tutorial 3](../tut3_vertices/) before attempting this
    one.
- write the vertex code, which will compute the dot product between two given
    vectors.
- write the host code that will add several vertices to the graph, and connect
    them to appropriate tensors and tensor slices.
- Optionally create a version of this program that runs on the IPU hardware.

A brief [summary](#summary) is included at the end this tutorial. Do not
hesitate to read through the [Poplar and PopLibs User
Guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/)
to complement this tutorial. Use `tut5_matrix_vector/start_here` as your
working directory.

## Setup

In order to run this tutorial on the IPU you will need to have a Poplar
SDK environment enabled (see the [Getting Started Guide for your IPU
system](https://docs.graphcore.ai/en/latest/getting-started.html)).

You will also need a C++ toolchain compatible with the C++11 standard,
build commands in this tutorial use GCC.

## Vertex code

The file `matrix-mul-codelets.cpp` contains the outline for the vertex
code that will perform a dot product. Its input and output fields are
already defined:

```c++
class DotProductVertex : public Vertex {
public:
  Input<Vector<float>> a;
  Input<Vector<float>> b;
  Output<float> out;
}
```

**TO DO (1): write the vertex code**

For the vertex to provide the calculation to Poplar the `compute` method
of `DotProductVertex` needs to be completed. The method should calculate
the [dot product](https://en.wikipedia.org/wiki/Dot_product) of the two
input vectors `a` and `b`, and store the scalar result in `out`.

Algebraically, the dot product of two vectors $a = [a_0, a_1, ..., a_n]$ and $b = [b_0, b_1, ..., b_n]$ is computed as:

$a_0 * b_0 + a_1 * b_1 + ... + a_n * b_n$

You may also find useful to look again at [the codelet in tutorial
3](../tut3_vertices/complete/tut3_codelets.cpp). Tip: within the
`compute` method of the codelet, you can find the number of elements in
the vector `a` with `a.size()`.

## Host code

The host code follows a similar pattern to the host code in the previous
tutorials.

There are three tensors defined for the input matrix, input vector and
output vector:

```c++
Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
Tensor inputVector = graph.addVariable(FLOAT, {numCols}, "inputVector");
Tensor outputVector = graph.addVariable(FLOAT, {numRows}, "outputVector");
```

The function `buildMultiplyProgram` creates the graph and control
program for performing the multiplication. The control program executes
a single compute set called `mulCS`. This compute set consists of a
vertex for each output element of the output vector (in other words, one
vertex for each row of the input matrix).

**TO DO (2): add vertices to the graph**

The next task in this tutorial is to write the host code to add the
vertices to the compute set.

- Create a loop that performs `numRows` iterations, each of which will add a
    vertex to the graph. Hint: given a Poplar tensor `t` of shape
    `{numRows, numCols}`, you can get the size of the i-th dimension with
    `t.dim(i)`. So for example `numRows == t.dim(0)`.
- Use the `addVertex` function of the graph object to add a vertex of type
    `DotProductVertex` to the `mulCS` compute set. You may find it helpful to
    look again at how we added a vertex in
    [tutorial 3](../tut3_vertices/complete/tut3_complete.cpp).
- Use the last argument of `addVertex` to connect the fields of the vertex to
    the relevant tensor slices for that row. For example, say we want to create
    a vertex `v` that has an input `in` and an output `out`, and in the graph
    we've already defined two tensors with the same names, we can thus create
    the vertex as:

    ```c++
    VertexRef v = graph.addVertex(computeSet, "v", {{"in", in}, {"out", out}});
    ```

    In this case, each vertex takes one row of the matrix (you can use
    the index operator on the `matrix` tensor, for example, the i-th row
    is `matrix[i]`), and the entire `in` tensor, and outputs to a single
    element of the `out` tensor.

- Map the newly created vertex to a tile. If `i` is the counter of the loop
    we're in, we can map the vertex to tile `i`. Again, it may help to check
    how we did this in tutorial 3.
- Finally, use `graph.setPerfEstimate()` to specify the estimated number of
    cycles that this vertex will take to execute. This is needed only when
    using the `IPUModel` device, and is not really important other than for
    profiling. So you can just set an arbitrary integer.

After adding this code, you can build and run the example. A makefile is
provided to compile the program. You can build it by running `make`

As you can see from the host program code, you'll need to provide two
arguments to the execution command that specify the size of the matrix.
For example, running the program as shown below will multiply a 40x50
matrix by a vector of size 50:

```console
$ ./tut5_start_here 40 50
```

The host code includes a check for the correctness of the result.

## (Optional) Using the IPU

This section describes how to modify the program to use the IPU
hardware.

- Copy `tut5.cpp` to `tut5_ipu_hardware.cpp` and open it in an editor.
- Add these include lines:

```c++
#include <poplar/DeviceManager.hpp>
#include <algorithm>
```

- Remove the following lines which create an IPU model device:

```c++
IPUModel ipuModel;
Device device = ipuModel.createDevice();
```

- And add the following lines at the start of `main`:

```c++
// Create the DeviceManager which is used to discover devices
auto manager = DeviceManager::createDeviceManager();

// Attempt to attach to a single IPU:
auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
std::cout << "Trying to attach to IPU\n";
auto it = std::find_if(devices.begin(), devices.end(), [](Device &device) {
   return device.attach();
});

if (it == devices.end()) {
  std::cerr << "Error attaching to device\n";
  return 1; //EXIT_FAILURE
}

auto device = std::move(*it);
std::cout << "Attached to IPU " << device.getId() << std::endl;
```

This gets a list of all devices consisting of a single IPU that are
attached to the host and tries to attach to each one in turn until
successful. This is a useful approach if there are multiple users on the
host. It is also possible to get a specific device using its
device-manager ID with the `getDevice` function.

- Remove the line with `setPerfEstimate` in function `buildMultiplyProgram`:

```c++
graph.setPerfEstimate(v, 20);
```

This line gives an estimate of the number of cycles that the calculation
will take for a given vertex, it is only needed when we use the IPU
model and write custom vertices like `DotProductVertex` in this
tutorial. When we use IPU hardware the cycles will be measured, should
we decide to profile the program like in [tutorial
4](../tut4_profiling/).

- Compile the program.

```console
$ g++ --std=c++11 tut5_ipu_hardware.cpp -lpoplar -lpoputil -o tut5_ipu
```

Before running this you need to make sure that your system is configured
correctly in order to attach to IPUs (see the [Getting Started Guide for
your IPU
system](https://docs.graphcore.ai/en/latest/getting-started.html)).

- Run the program to see the same results as running on IPU model

```console
$ ./tut5_ipu_hardware
```

## Summary

In this tutorial, we wrote a program that performs a matrix-vector
multiplication using a custom vertex. The codelet itself computes the
dot product between two vectors, in order to compute the multiplication
between a matrix and a vector we added several of these vertices to the
Poplar graph: one for each row of the matrix. Finally we connected them
to the appropriate row and tensors. These vertices have all been added
to the same compute set, which means they will execute in parallel on
the IPU. We run the program on the IPU model, but we've also seen what
changes are needed to make it run on the IPU hardware.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
