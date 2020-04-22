Tutorial 7: matrix-vector multiplication optimisation
-----------------------------------------------------

For a massively parallel machine such as the IPU, the strategy in
tutorial 6 is not the most efficient. In particular:

* Allocating one vertex to each row may not create enough vertices to
  occupy all the workers on the machine.

* The input vector needs to be broadcast to every tile, which
  results in a large communication cost.

A more efficient strategy is to split each row into several segments
and have the vertices calculate the dot product of that row segment
with the corresponding segment of the input vector. After these
partial sums have been calculated, a reduction is needed to add all
the partial sums together for each output element to get the final
output value.

This tutorial uses a simple algorithm to estimate the best way of splitting the
data across the tiles in order to get the best performance. The Poplibs
matrix-multiply functions use a similar, but more sophisticated, method that
also considers the best instructions to use and different ways of reshaping the
tensor data.

Make a copy of the files in ``tut7_matrix_vector_opt`` in your working
directory.

In this tutorial, there is no code for you to complete; the aim is to understand
the code and experiment with different matrix sizes.

The device code in ``matrix-mul-codelets.cpp`` includes an extra vertex class,
called ``ReduceVertex``, which sums a set of values in a vector.

The host file follows the same structure as the previous tutorial. The
difference in this example is in the ``buildMultiplyProgram`` function. The
first thing this does is work out how many segments to split the matrix rows
into:

.. code-block:: c++

    // Get the optimal column axis split to split the number of columns
    // into partial sums
    unsigned colAxisSplit = calcOptimalColAxisSplit(graph, numRows, numCols);

Looking at the ``calcOptimalColAxisSplit`` function, you can see that it just
iterates through all possible splits and calls the ``estimateCycles`` function
for that split. The ``estimateCycles`` function itself tries to estimate how
many cycles the calculation will take to perform. This is done by looking at the
worst-case running time and exchange time of the tiles involved in both the
partial-sum calculation phase and the reduction phase.

Once the split is determined, the code creates a new tensor to hold
the intermediate partial-sum calculations:

.. code-block:: c++

    // Create a tensor to hold the intermediate calculated partial sums
    auto partials = graph.addTensor("float", {numRows, colAxisSplit}, "partials");

The calculation is split into two phases. The first phase calculates the dot
product of all the row segments and writes to the ``partials`` tensor. The
second phase reads the ``partials`` tensor, adds up the partial sums and writes
the output to the final ``out`` tensor.

These two phases are built with two loops. The first populates the ``mulCS``
compute set:

.. code-block:: c++

    // Create a compute set to hold the vertices to perform the
    // partial sum calculations.
    ComputeSet mulCS = graph.addComputeSet("mulCS");

    // Create a vertex for each segment, for each row.
    for (unsigned i = 0; i < colAxisSplit; ++i) {
        ...
        auto v = graph.addVertex(mulCS, "DotProductVertex",
        ...
    }

The second loop builds up the ``reduceCS`` compute set:

.. code-block:: c++

    // Create a compute set to calculate the reduction.
    auto reduceCS = graph.createComputeSet("reduceCS");

    // For each output element create a vertex.
    for (unsigned row = 0; row < numRows; ++row) {
    ...
    ...
    auto v = graph.addVertex(reduceCS, "ReduceVertex",
    ...
    ...

The final program, which performs the entire multiplication, consists of
executing the two compute sets in order:

.. code-block:: c++

    return Sequence(Execute(mulCS), Execute(reduceCS));

At the end, the program calls the ``printProfileSummary`` function
to display information about memory use and the number of cycles for
execution and communication.

This example includes a makefile so you can build it by running ``make``. After
that, try running the program on for various sizes of data. For example:

.. code-block:: bash

    $ ./matrix-vector 10000 1000
    Multiplying matrix of size 10000x1000 by vector of size 1000
    Creating environment (compiling vertex programs)
    Constructing compute graph and control program
    Best split chosen:
    colsAxisSplit=5, total cost=4751 (compute cost=4410, exchange cost=200,
                                      reduce exchange cost=45,
                                      reduce compute cost=96)
    Worst cost seen: 64373
    Running graph program to multiply matrix by vector
    Multiplication result OK

This output is followed by the profile data.

From the output above, you can see that the program splits each row into five
segments with an estimated cycle cost of 4,751 cycles.

The profile output includes a lot of information. The section most relevant to
us is under the heading "Execution", you should see something like:

.. code-block:: console

    Execution:

    Total cycles:                                  8,190,756 (approx 5,119.2 microseconds)
    Total compute cycles (including idle threads): 5,513,152 Estimated (must enable debug.instrumentCompute)
    Total compute cycles (excluding idle threads): 5,334,536
    Total IPU exchange cycles:                     963,855
    Total global exchange cycles:                  Must enable debug.instrumentExternalExchange
    Total host exchange cycles:                    9,952,701,003
    Total shared structure copy cycles:            0
    Total sync cycles:                             781,286

    Cycles by vertex type:
        DotProductVertex            (50000 instances):    5,250,000
        ReduceVertex                (10000 instances):       80,000
        poplar_rt::ShortMemcpy         (63 instances):        4,536

The figure we are most interested in is:

.. code-block:: console

    Total compute cycles (excluding idle threads): 5,334,536

This is the total number of compute cycles *across all tiles*. If we divide this by
1,216 (the number of tiles in an IPU) we get 4,387 which is pretty close to the program’s
estimate of 4,410.

The "Total cycles" line is the overall time taken to run the program; you can also
think of this as the number of cycles taken by a single tile. It is the total cycles
for compute plus exchange plus sync plus host IO.

The "Total host exchange cycles" line tells us the total number of cycles used
for transferring data to and from the host by all tiles. If you divide this by 1,216
and subtract that from the "Total cycles" number, then you get the compute + sync + exchange
cycles for one tile.

You can get far more detailed insights into the behaviour of the program by using the
PopVision Graph Analyser tool. The program writes out ``graph.json`` and ``execution.json``
files that can be read by the graph analyser.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.

