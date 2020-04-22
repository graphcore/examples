Tutorial 6: matrix-vector multiplication
----------------------------------------

This tutorial builds up a more complex calculation on vertices: multiplying a
matrix by a vector. Make a copy of the files from
``tut6_matrix_vector/start_here`` in your working directory.

The file ``matrix-mul-codelets.cpp`` contains the outline for the vertex code
that will perform a dot product. Its input and output fields are already
defined:

.. code-block:: c++

  class DotProductVertex : public Vertex {
  public:
    Input<Vector<float>> a;
    Input<Vector<float>> b;
    Output<float> out;
  }

* Complete the ``compute`` function of ``DotProductVertex``.

The host code follows a similar pattern to the host code in the previous
tutorials. To demonstrate other ways of running Poplar code, this tutorial
uses the host CPU as the target. This can be useful for functional testing
when initially developing code. It is faster to compile and, because it
only models a single tile, there is no need to worry about mapping tensors
or compute sets to tiles. (Note that the CPU target cannot be used for
profiling.)

There are three tensors defined for the input matrix, input vector
and output vector:

.. code-block:: c++

  Tensor matrix = graph.addVariable(FLOAT, {numRows, numCols}, "matrix");
  Tensor inputVector = graph.addVariable(FLOAT, {numCols}, "inputVector");
  Tensor outputVector = graph.addVariable(FLOAT, {numRows}, "outputVector");

The function ``buildMultiplyProgram`` creates the graph and control program for
performing the multiplication. The control program executes a single compute set
called ``mulCS``. This compute set consists of a vertex for each output element
of the output vector (in other words, one vertex for each row of the input
matrix).

The next task in this tutorial is to write the host code to add the vertices to
the compute set.

* Create a loop that performs ``numRows`` iterations, each of which will add a
  vertex to the graph.

  * Use the ``addVertex`` function of the graph object to add a vertex of type
    ``DotProductVertex`` to the ``mulCS`` compute set.

  * Use the final argument of ``addVertex`` to connect the fields of the
    vertex to the relevant tensor slices for that row. Each vertex takes one
    row of the matrix (you can use the index operator on the ``matrix``
    tensor), and the entire ``in`` tensor, and outputs to a single element of
    the ``out`` tensor.

After adding this code, you can build and run the example. A makefile is provided
to compile the program.

As you can see from the host program code, you'll need to provide two arguments
to the execution command that specify the size of the matrix. For example,
running the program as shown below will multiply a 40x50 matrix by a vector of
size 50:

.. code-block:: bash

  $ ./matrix-vector 40 50

The host code includes a check that the result is correct.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
