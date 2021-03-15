Tutorial 1: programs and variables
----------------------------------

Copy the file ``tut1_variables/start_here/tut1.cpp`` to your working directory
and open it in an editor. The file contains the outline of a C++ program
including some Poplar library headers and a namespace.

Graphs, variables and programs
..............................

All Poplar programs require a ``Graph`` object to construct the computation
graph. Graphs are always created for a specific target (where the target is a
description of the hardware being targeted, such as an IPU). To obtain the
target we need to choose a device.

The tutorials use a simulated target by default, so will run on any
machine even if it has no Graphcore hardware attached. On systems with
accelerator hardware, the header file ``poplar/DeviceManager.hpp`` contains API
calls to enumerate and return ``Device`` objects for the attached hardware.

Simulated devices are created with the ``IPUModel`` class, which models the
functionality of an IPU on the host. The ``createDevice`` function creates a new
virtual device to work with. Once we have this device we can create a ``Graph``
object to target it.

* Add the following code to the body of ``main``:

  .. code-block:: c++

    // Create the IPU Model device
    IPUModel ipuModel;
    Device device = ipuModel.createDevice();
    Target target = device.getTarget();

    // Create the Graph object
    Graph graph(target);

Any program running on an IPU needs data to work on. These are defined as
*variables* in the graph.

* Add the following code to create the first variable in the program:

  .. code-block:: c++

    // Add variables to the graph
    Tensor v1 = graph.addVariable(FLOAT, {4}, "v1");

This adds one vector variable with four elements of type ``float`` to the graph.
The final string parameter, ``"v1"``, is used to identify the data in
debugging/profiling tools.

* Add three more variables:

  - ``v2``: another vector of 4 floats.
  - ``v3``: a two-dimensional 4x4 tensor of floats.
  - ``v4``: a vector of 10 integers (of type INT).

Note that the return type of ``addVariable`` is ``Tensor``. The ``Tensor`` type
represents data on the device in multi-dimensional tensor form. This type is
used to reference the whole variable but, as we will see later, it can also be
used to reference partial slices of variables, or data constructed from multiple
variables.

Variables must be allocated to tiles. One option is to allocate the whole
variable to one tile.

* Add the following code:

  .. code-block:: c++

    // Allocate v1 to reside on tile 0
    graph.setTileMapping(v1, 0);

Most of the time, programs actually deal with data spread over multiple tiles.

* Add the following code:

  .. code-block:: c++

    // Spread v2 over tiles 0..3
    for (unsigned i = 0; i < 4; ++i)
      graph.setTileMapping(v2[i], i);

This calls ``setTileMapping`` on sub-tensors of the variable ``v2`` to spread it
over multiple tiles.

* Add code to allocate ``v3`` and ``v4`` to other tiles.

Now that we have created some variables in the graph, we can create a control
program to run on the device. Programs are represented as sub-classes of the
``Program`` class. In this example we will use the ``Sequence`` sub-class, which
represents a number of steps executed sequentially.

* Add this declaration:

  .. code-block:: c++

    // Create a control program that is a sequence of steps
    program::Sequence prog;

    // Debug print the tensor to the host console
    prog.add(program::PrintTensor("v1-debug", v1));

Here, the sequence has one step that will perform a debug print (via the host)
of the data on the device.

Now that we have a graph and a program, we can see what happens when it is
deployed on the device. To do this we must first create an ``Engine`` object.

* Add to the code:

  .. code-block:: c++

    // Create the engine
    Engine engine(graph, prog);
    engine.load(device);

This object represents the compiled graph and program, which are ready to run on
the device.

* Add code to run the control program:

  .. code-block:: c++

    // Run the control program
    std::cout << "Running program\n";
    engine.run(0);
    std::cout << "Program complete\n";

* Now compile the host program (remembering to link in the Poplar library using
  the ``-lpoplar`` flag):

  .. code-block:: bash

    $ g++ --std=c++11 tut1.cpp -lpoplar -o tut1

* Then run the compiled program:

  .. code-block:: bash

    $ ./tut1

When the program runs, the debug output prints out uninitialised values, because
we allocated a variable in the graph which is never initialised or written to:

.. code-block:: console

  v1-debug: {0,0,0,0}

Initialising variables
......................

In addition to variables, the graph can contain constant values. This is one way
to initialise data in the graph.

* After the code adding variables to the graph, add the following:

  .. code-block:: c++

    // Add a constant tensor to the graph
    Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});

This line adds a new constant tensor to the graph whose elements have the values
shown.

* Allocate the data in ``c1`` to tile 0:

  .. code-block:: c++

     // Allocate c1 to tile 0
     graph.setTileMapping(c1, 0);

* Now add the following to the sequence program, just before the ``PrintTensor``
  program:

  .. code-block:: c++

    // Add a step to initialise v1 with the constant value in c1
    prog.add(program::Copy(c1, v1));

Here we have used a predefined control program called ``Copy``, which copies
data between tensors on the device. Copying the constant tensor ``c1`` into the
variable ``v1`` will result in ``v1`` containing the same data as ``c1``.

Note that the synchronisation and exchange phases of IPU execution described in
the `IPU Programmer's Guide
<https://www.graphcore.ai/docs/ipu-programmers-guide#programming-model>`_
are performed automatically by the Poplar library functions and do not need
to be specified explicitly.

If you recompile and run the program you should see the debug print of ``v1``
shows initialised values:

.. code-block:: console

  v1-debug: {1,1.5,2,2.5}

Copying can also be used between variables.

* After the ``v1`` debug print command, add the following:

  .. code-block:: c++

     // Copy the data in v1 to v2
     prog.add(program::Copy(v1, v2));
     // Debug print v2
     prog.add(program::PrintTensor("v2-debug", v2));

Now running the program will print both ``v1`` and ``v2`` with the same values.

Getting data into and out of the device
.......................................

Most data to be processed will not be constant, but will come from the host.
There are a couple of ways of getting data in and out of the device from the
host. The simplest is to create a read or write handle connected to a tensor.
This allows the host to transfer data directly to and from that variable.

* Add code (before the engine creation instruction) to create read and write
  handles for the ``v3`` variables:

  .. code-block:: c++

    // Create host read/write handles for v3
    graph.createHostWrite("v3-write", v3);
    graph.createHostRead("v3-read", v3);

These handles are used after the engine is created.

* Add the following code after the engine creation instruction:

  .. code-block:: c++

    // Copy host data via the write handle to v3 on the device
    std::vector<float> h3(4 * 4, 0);
    engine.writeTensor("v3-write", h3.data(), h3.data() + h3.size());

Here, ``h3`` holds data on the host (initialised to zeros) and the
``writeTensor`` call performs a synchronous write over the PCIe bus (simulated
in this case) to the tensor on the device. After this call, the values of ``v3``
on the device will be set to zero.

* After the call to ``engine.run(0)``, add the following:

  .. code-block:: c++

    // Copy v3 back to the host via the read handle
    engine.readTensor("v3-read", h3.data(), h3.data() + h3.size());

    // Output the copied back values of v3
    std::cout << "\nh3 data:\n";
    for (unsigned i = 0; i < 4; ++i) {
      std::cout << "  ";
      for (unsigned j = 0; j < 4; ++j) {
        std::cout << h3[i * 4 + j] << " ";
      }
      std::cout << "\n";
    }

Here, we are copying device data back to the host and printing it out. When the
program is re-compiled and re-run, this prints all zeros (because the program on
the device doesn't modify the ``v3`` variable):

.. code-block:: console

  h3 data:
    0 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0

Let's see what happens when ``v3`` is modified on the device. We will use
``Copy`` again, but also start to look at the flexible data referencing
capabilities of the ``Tensor`` type.

* Add the following code to create slices of ``v1`` and ``v3`` immediately
  after the creation of the host read/write handles for ``v3``:

  .. code-block:: c++

    // Copy a slice of v1 into v3
    Tensor v1slice = v1.slice(0, 3);
    Tensor v3slice = v3.slice({1,1},{2,4});

These lines create a new ``Tensor`` object that references data in the graph.
This does not create new state but just references parts of ``v1`` and ``v3``.

* Now add this copy program:

  .. code-block:: c++

    prog.add(program::Copy(v1slice, v3slice));

This step copies three elements from ``v1`` into the middle of ``v3``.
Re-compile and re-run the program to see the results:

.. code-block:: console

  h3 data:
    0 0 0 0
    0 1 1.5 2
    0 0 0 0
    0 0 0

Data streams
............

The most efficient way to get data in and out of the device is to use data
streams (see the the `Poplar and Poplibs User Guide
<https://www.graphcore.ai/docs/poplar-and-poplibs-user-guide#data-streams-and-remote-buffers>`_
for more information).

During machine learning training, for example, data streams are the best
mechanism to use for getting example data into the device. Data streams need to be
created and explicitly named in the graph.

* Add the following code to the program definition:

  .. code-block:: c++

    // Add a data stream to fill v4
    DataStream inStream = graph.addHostToDeviceFIFO("v4-input-stream", INT, 10);

    // Add program steps to copy from the stream
    prog.add(program::Copy(inStream, v4));
    prog.add(program::PrintTensor("v4-0", v4));
    prog.add(program::Copy(inStream, v4));
    prog.add(program::PrintTensor("v4-1", v4));

These instructions copy from the input stream to the variable ``v4`` twice.
After each copy, ``v4`` holds new data from the host.

After the engine is created, the data streams need to be connected to data on
the host. This is achieved with the ``Engine::connectStream`` function.

* Add the following code after the creation of the engine:

  .. code-block:: c++

    // Create a buffer to hold data to be fed via the data stream
    std::vector<int> inData(10 * 3);
    for (unsigned i = 0; i < 10 * 3; ++i)
      inData[i] = i;

    // Connect the data stream
    engine.connectStream("v4-input-stream", &inData[0], &inData[10 * 3]);

Here, we've connected the stream to a data buffer on the host, using it as a
circular buffer of data. Recompile and run the program again, and you can see
that after each copy from the stream, ``v4`` holds new data copied from the host
memory buffer:

.. code-block:: console

  v4-0: {0,1,2,3,4,5,6,7,8,9}
  v4-1: {10,11,12,13,14,15,16,17,18,19}

(Optional) Using the IPU
........................

This section describes how to modify the program to use the IPU hardware.

* Copy ``tut1.cpp`` to ``tut1_ipu_hardware.cpp`` and open it in an editor.

* Remove the import declaration:

  .. code-block:: c++

    #include <poplar/IPUModel.hpp>

* Add this import declaration:

  .. code-block:: c++

    #include <poplar/DeviceManager.hpp>

* Replace the following lines from the start of ``main``:

  .. code-block:: c++

    // Create the IPU Model device
    IPUModel ipuModel;
    Device device = ipuModel.createDevice();

  with this code:

  .. code-block:: c++

    // Create the DeviceManager which is used to discover devices
    DeviceManager manager = DeviceManager::createDeviceManager();

    // Attempt to attach to a single IPU:
    Device device;
    bool success = false;
    // Loop over all single IPU devices on the host
    // Break the loop when an IPU is successfully acquired
    for (auto &hwDevice : manager.getDevices(poplar::TargetType::IPU, 1)) {
      device = std::move(hwDevice);
      std::cerr << "Trying to attach to IPU " << device.getId() << std::endl;
      if ((success = device.attach())) {
        std::cerr << "Attached to IPU " << device.getId() << std::endl;
        break;
      }
    }
    if (!success) {
      std::cerr << "Error attaching to device" << std::endl;
      return -1;
    }

This gets a list of all devices consisting of a single IPU that are attached to
the host and tries to attach to each one in turn until successful.
This is a useful approach if there are multiple users on the host.
It is also possible to get a specific device using its device-manager ID with the
``getDevice`` function.

* Compile the program.

  .. code-block:: bash

    $ g++ --std=c++11 tut1_ipu_hardware.cpp -lpoplar -o tut1_ipu_hardware

Before running this you need to make sure that you have set the environment
variables for the Graphcore drivers (see the Getting Started Guide for your IPU
system).

* Run the program to see the same results.

  .. code-block:: bash

    $ ./tut1_ipu_hardware

You can make similar modifications to the programs in the other tutorials
in order to use the IPU hardware.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
