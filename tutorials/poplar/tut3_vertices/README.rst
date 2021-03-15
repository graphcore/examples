Tutorial 3: writing vertex code
-------------------------------

In this tutorial we will look at how compute steps are built up from running
pieces of work (the *vertices* of the compute graph) in parallel as *compute sets*. The
process for constructing compute sets described here is the same method that the
Poplibs libraries use.

Make a copy of the file ``tut3_vertices/start_here/tut3.cpp`` and open it in an
editor. This file has an outline program like tutorial 2, but does not use the
Poplibs libraries. Instead, we will write the device code for the vertices in
C++.

The program initially adds two 4-element vectors to the graph (``v1`` and
``v2``). The code we are going to add will set each element of ``v2`` to the
suffix sum of ``v1``. So ``v2[0]`` will contain the sum of all the elements of
``v1``, ``v2[1]`` will contain the sum of the elements of ``v1``, starting at
element 1, and so on.

Creating a codelet
..................

To implement this operation, we have to write some code to run on the device,
known as a *codelet*. A file is provided for this in the tutorial directory,
called ``tut3_codelets.cpp``. Make a copy of this file in your working
directory.

* Add the following code to ``tut3.cpp`` after the graph object is
  created:

  .. code-block:: c++

    // Add codelets to the graph
    graph.addCodelets("tut3_codelets.cpp");

This instructs the host program to load the device code into the graph and
compile it to run on the device.

Inside ``tut3_codelets.cpp`` is the outline of a codelet. Like all Poplar
codelets, it is a C++ class derived from the ``poplar::Vertex`` class, with a
single member function called ``compute``. This function defines the work done
by the vertex. The ``compute`` function returns true to indicate successful
completion.

We'll add code to this vertex to take in a set of numbers and write out the sum
of those numbers.

* Alter the class in the codelets file, adding the following fields to the
  vertex definition:

  .. code-block:: c++

    class SumVertex : public poplar::Vertex {
    public:
      // Fields
      poplar::Input<poplar::Vector<float>> in;
      poplar::Output<float> out;

The fields named ``in`` and ``out`` represent the vertex's connections to
external tensors. They are used in the body of the ``compute``
function to read and write the tensor data being operated on.

* Fill in the body of the ``compute`` function to calculate the output as the
  sum of the inputs:

  .. code-block:: c++

    // Compute function
    bool compute() {
      *out = 0;
      for (const auto &v : in) {
        *out += v;
      }
      return true;
    }

Note that the ``out`` field can be updated even if the destination tensor is on
another tile. This is because the vertex operates on a local copy of the data.
The final result is transferred to the destination tile in the exchange phase
after the compute is complete.

Creating a compute set
......................

Now that we have some device code, we can build a step to execute it and add
this to our control program. To do this, you need to:

#. Create a compute set that defines the set of vertices that are executed
   in parallel at each step

#. Add vertices to the compute set to execute the task

#. Connect data to the vertices (in other words, define the *edges* of the graph)

#. Set the tile mapping of the vertices

These are described in more detail below.

#. **Create a compute set:** add the following declaration to the control program
   in ``tut3.cpp``, after the code to initialise ``v1`` (the string argument is a
   debug identifier):

   .. code-block:: c++

     ComputeSet computeSet = graph.addComputeSet("computeSet");

#. **Add four vertices to the compute set:** add the following loop to the code,
   after the compute set definition. This passes the name of the class defined in
   the codelet, which will create an instance of that class for each vertex. Each
   vertex will output to a different element of ``v2``.

   .. code-block:: c++

     for (unsigned i = 0; i < 4; ++i) {
       VertexRef vtx = graph.addVertex(computeSet, "SumVertex");
     }

   Note that the ``"SumVertex"`` argument specifies the type of vertex to use, in
   this case it's the one we defined in the ``tut3_codelets.cpp`` file that was
   loaded into the graph.

#. **Define the connections:** add the following code to the body of the loop you
   just created. This connects the input and output variables to the vertices. By
   using tensor operators and the loop index, each vertex is connected to
   different tensor elements.

   .. code-block:: c++

     graph.connect(vtx["in"], v1.slice(i, 4));
     graph.connect(vtx["out"], v2[i]);


#. **Set the tile mapping:** Add the following code to the body of the same loop:

   .. code-block:: c++

     graph.setTileMapping(vtx, i);

   Here, each vertex is mapped to a different tile.

Executing the compute set
.........................

If you are using the IPU Model simulation and want to profile the performance,
you can set a cycle estimate for the vertex, if known. This is the number of
cycles it takes to execute the codelet on the IPU. Here we set the cycle
estimate to be 20 cycles.

.. code-block:: c++

  graph.setPerfEstimate(vtx, 20);

After creating the compute set, the final task is to add a step to the control
program to execute the compute set:

* Add the following code (anywhere after the ``prog`` sequence has been defined,
  but before ``v2`` is printed):

  .. code-block:: c++

    // Add step to execute the compute set
    prog.add(Execute(computeSet));

* Now you can compile and run the program. You do not need to compile the
  codelet because your program can load and compile the vertex at run time.

You should now see that the ``v2`` tensor has been updated to the expected
values:

.. code-block:: console

  v2: {7,6,4.5,2.5}

You can also compile the vertex code from the command line, with the ``popc``
command:

.. code-block:: bash

  $ popc tut3_codelets.cpp -o tut3_codelets.gp

You can then use the compiled code by loading it, instead of the source, in your
program:

.. code-block:: c++

    // Add codelets to the graph
    graph.addCodelets("tut3_codelets.gp");

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
