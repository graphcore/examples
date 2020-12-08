Tutorial 2: using Poplibs
-------------------------

Make a copy of the file ``tut2_operations/start_here/tut2.cpp`` in your working
directory and open it in an editor. This file contains a basic Poplar program
structure similar to that seen in tutorial 1. It creates a graph with a couple
of variables and initialises them. However, this time it includes some extra
headers from the ``popops`` library:

.. code-block:: c++

  #include <popops/codelets.hpp>
  #include <popops/ElementWise.hpp>

This gives us access to library functions for data manipulation, which have been
highly optimised for IPU devices.

* To use this, you need to add the device-side library code to the graph, so
  that it is loaded when the code is run:

  .. code-block:: c++

    popops::addCodelets(graph);

A similar ``addCodelets`` call is required for each of the Poplibs libraries you
use in your program.

* Compile and run the code (remember to link in the ``popops`` and ``poputil`` libraries):

  .. code-block:: bash

    $ g++ --std=c++11 tut2.cpp -lpoplar -lpopops -lpoputil -o tut2
    $ ./tut2

The code doesn't do anything at the moment so let's add an operation to
the graph.

* Add the following, before the engine creation, to extend the program
  sequence with an add operation:

  .. code-block:: c++

    // Extend program with elementwise add (this will add to the sequence)
    Tensor v3 = popops::add(graph, v1, v2, prog, "Add");

    prog.add(PrintTensor("v3", v3));

The ``popops::add`` function extends the sequence ``prog`` with extra steps to
perform an elementwise add. We've also created a new variable, ``v3``, in the
graph for the returned result. So, after the add operation, ``v3`` holds the
result of adding the elements of ``v1`` to ``v2``.

* Re-compile and re-run the program. You should see the results of the
  addition:

  .. code-block:: console

    v3: {
     {5,4.5},
     {4,3.5}
    }

* Add code to add ``v2`` to the result tensor ``v3`` and print the
  result.

That is all that is required to use the Poplibs library functions. You can see
the capability of these libraries by browsing the `Poplibs API documentation
<https://www.graphcore.ai/docs/poplar-api-reference#document-poplibs_api>`_
or the header files in the ``include`` directories of the Poplar installation.

Reshaping and transposing data
..............................

When calling libraries to perform operations, there are many ways to
arrange how data is passed to the operation. These are in the ``Tensor.hpp`` header
file and documented in the `Poplar API Reference
<https://www.graphcore.ai/docs/poplar-api-reference#poplar-tensor-hpp>`_.

In tutorial 1 we used slicing, but there are also functions for reshaping and
transposing data.

* Add the following code to add ``v1`` to the transpose of the 2x2 matrix ``v2``:

  .. code-block:: c++

    // Example element wise addition using a transposed view of the data
    Tensor v5 = popops::add(graph, v1, v2.transpose(), prog, "Add");
    prog.add(PrintTensor("v5", v5));

* Re-compile and re-run the program to see the result.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
