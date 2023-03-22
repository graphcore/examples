<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# Poplar Tutorial 6: Matrix-vector Multiplication Optimisation

As always, do not hesitate to read through the [Poplar and PopLibs User
Guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/3.2.0/index.html)
to complement this tutorial.

## Setup

In order to run this tutorial on the IPU you will need to have a Poplar
SDK environment enabled (see the [Getting Started Guide for your IPU
system](https://docs.graphcore.ai/en/latest/getting-started.html)).

You will also need a C++ toolchain compatible with the C++11 standard,
build commands in this tutorial use GCC.

## Optimising matrix-vector multiplication

In the previous tutorial, we learnt how to build a more complex vertex
that multiplies a matrix by a vector. However, for a massively parallel
machine such as the IPU, the strategy in tutorial 5 is not the most
efficient. In particular:

- Allocating one vertex to each row may not create enough vertices to occupy
    all the workers on the machine.
- The input vector needs to be broadcast to every tile, which results in a
    large communication cost.

A more efficient strategy is to split each row into several segments and
have the vertices calculate the dot product of that row segment with the
corresponding segment of the input vector. After these partial sums have
been calculated, a reduction is needed to add all the partial sums
together for each output element to get the final output value.

This tutorial uses a simple algorithm to estimate the best way of
splitting the data across the tiles in order to get the best
performance. The PopLibs matrix-multiply functions use a similar, but
more sophisticated, method that also considers the best instructions to
use and different ways of reshaping the tensor data.

In this tutorial, there is no code for you to complete; the aim is to
understand the code and experiment with different matrix sizes. You can
use the command line option `--device` to select the device on which the
code is run. By default, a Mk2 `IPUModel` is used as a simulation of the
behaviour of the IPU hardware.

The device code in `matrix-mul-codelets.cpp` includes an extra vertex
class, called `ReduceVertex`, which sums a set of values in a vector.

The host file follows the same structure as the previous tutorial. The
difference in this example is in the `buildMultiplyProgram` function.
The first thing this does is work out how many segments to split the
matrix rows into:

```c++
// Get the optimal column axis split to split the number of columns
// into partial sums
unsigned colAxisSplit = calcOptimalColAxisSplit(graph, numRows, numCols);
```

Looking at the `calcOptimalColAxisSplit` function, you can see that it
just iterates through all possible splits and calls the `estimateCycles`
function for that split. The `estimateCycles` function itself tries to
estimate how many cycles the calculation will take to perform. This is
done by looking at the worst-case running time and exchange time of the
tiles involved in both the partial-sum calculation phase and the
reduction phase. Note that the cycles estimated in `estimateCycles` can
be manually adjusted by the user. The choice of exact number in this
tutorial is based on assumptions. It is important to implement the code
and run it on hardware in order to obtain reliable cycle counts.

Once the split is determined, the code creates a new tensor to hold the
intermediate partial-sum calculations:

```c++
// Create a tensor to hold the intermediate calculated partial sums
auto partials = graph.addTensor("float", {numRows, colAxisSplit}, "partials");
```

The calculation is split into two phases. The first phase calculates the
dot product of all the row segments and writes to the `partials` tensor.
The second phase reads the `partials` tensor, adds up the partial sums
and writes the output to the final `out` tensor.

These two phases are built with two loops. The first populates the
`mulCS` compute set:

```c++
// Create a compute set to hold the vertices to perform the
// partial sum calculations.
ComputeSet mulCS = graph.addComputeSet("mulCS");

// Create a vertex for each segment, for each row.
for (unsigned i = 0; i < colAxisSplit; ++i) {
    ...
    auto v = graph.addVertex(mulCS, "DotProductVertex",
    ...
}
```

The second loop builds up the `reduceCS` compute set:

```c++
// Create a compute set to calculate the reduction.
auto reduceCS = graph.createComputeSet("reduceCS");

// For each output element create a vertex.
for (unsigned row = 0; row < numRows; ++row) {
...
...
auto v = graph.addVertex(reduceCS, "ReduceVertex",
...
...
```

The final program, which performs the entire multiplication, consists of
executing the two compute sets in order:

```c++
return Sequence({Execute(mulCS), Execute(reduceCS)});
```

At the end, the program calls the `printProfileSummary` function to
display information about memory use and the number of cycles for
execution and communication.

This example includes a makefile so you can build it by running `make`.
After that, try running the program for various sizes of data. For
example:

```console
$ ./tut6 10000 1000
Multiplying matrix of size 10000x1000 by vector of size 1000
Constructing compute graph and control program
Best split chosen:
colsAxisSplit=7, total cost=3996 (compute cost=3696,
                                  exchange cost=143,
                                  reduce exchange cost=49,
                                  reduce compute cost=108)
Worst cost seen: 53807
Running graph program to multiply matrix by vector
Multiplication result OK
```

This output is followed by the profile data.

From the output above, you can see that the program splits each row into
seven segments with an estimated cycle cost of 3,996 cycles.

The profile output includes a lot of information. The section most
relevant to us is under the heading "Execution", you should see
something like:

```console
Execution:

Programs executed:

<anonymous>.

  Total cycles:                                         6,681,766 (approx 5,023.9 microseconds)
  Tile average compute cycles (including idle threads): 3,801.8 (0.1% of total)
  Tile average compute cycles (excluding idle threads): 3,717.6 (0.1% of total)
  Tile average IPU exchange cycles:                     8,697.4 (0.1% of total)
  Tile average global exchange cycles:                  0.0 (0.0% of total)
  Tile average host exchange cycles:                    6,663,550.8 (99.7% of total)
  Tile average sync cycles:                             1,134.8 (0.0% of total)
```

The figure we are most interested in is:

```console
Tile average compute cycles (excluding idle threads): 3,717.6 (0.1% of total)
```

This is the average number of compute cycles *across all tiles* and is
pretty close to the program estimate of 3996. Note that since `IPUModel`
is used here, numbers given when profiling are estimated and might
differ from the execution profiling when running on hardware (see this
[explanation of
IPUModel](https://docs.graphcore.ai/projects/poplar-user-guide/en/3.2.0/poplar_programs.html)).

The "Total cycles" line is the overall time taken to run the program;
you can also think of this as the number of cycles taken by a single
tile. It is the total cycles for compute plus exchange plus sync plus
host I/O.

The "Tile average host exchange cycles" line tells us the average
number of cycles used for transferring data to and from the host by all
tiles. If you subtract this from the "Total cycles" number, then you
get the compute + sync + exchange cycles for one tile.

You can get far more detailed insights into the behaviour of the program
by using the PopVision Graph Analyser tool. The program writes out the
`profile.pop` file that can be read by the graph analyser. For more
information about the Graph Analyser, see [PopVision Graph Analyser User
Guide](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/3.11.2/).

Note:

- To run this tutorial on a Mk1 IPU Model, the command will change to:

```console
$ ./tut6 10000 1000 --device model-ip1
```

- This tutorial can also be run with IPU hardware. The command will change to:

```console
$ ./tut6 10000 1000 --device ipu
```

The execution profile will look like:

```console
Execution:

Programs executed:

<anonymous>.

  Total cycles:                                         25,444,984 (approx 19,131.6 microseconds)
  Tile average compute cycles (including idle threads): 28,300.3 (0.1% of total)
  Tile average IPU exchange cycles:                     8,743.1 (0.0% of total)
  Tile average global exchange cycles:                  0.0 (0.0% of total)
  Tile average host exchange cycles:                    2,641,488.4 (10.4% of total)
  Tile average sync cycles:                             135,849.6 (0.5% of total)
```

Note that the total cycles per tile using IPU hardware is significantly
larger than when using the IPU Model. The main overhead comes from the
`StreamCopyBegin` program. The `StreamCopyBegin`
is measuring cycles spent during which the host is preparing I/O. To
reduce latencies in exchange fabric, the configuration of exchange in
this simulated model is set to be simplistic. The previous cycle
estimates assumed theoretical optimum cycle counts which would really
only be seen for hand crafted assembler. For simplicity, this tutorial
is using a C++ vertex for which the cycle count is much higher.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
