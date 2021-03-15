Tutorial 4: profiling output
----------------------------

There are three ways of profiling code for the IPU:

  - outputting summary profile data to the console;
  - using the PopVision Analysis API to query profile information in C++ or Python;
  - using the PopVision Graph Analyser tool.

The PopVision Graph Analyser provides much more detailed information.
Profiling is documented in the `Profiling
<https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/profiler.html>`_
section of the Poplar and PopLibs User Guide.

This tutorial shows how to profile Poplar programs, but these techniques are applicable
to the PyTorch and TensorFlow frameworks.
As with earlier tutorials, this is running on the ``IPUModel`` to allow you run the tutorial
even without access to IPU hardware. Tutorial 1 provides an explanation of how to modify the
tutorial code to run on IPU hardware - all the instructions in this tutorial are unchanged.


Command line Profile Summary
..................................................

Make a copy of the file ``tut4_profiling/start_here/tut4.cpp`` and open it in an
editor.

* Use the ``matMul`` function from the ``poplin`` library to extend this
  example to calculate ((m1 * m2) * m3). The ``matMul`` function is documented
  in the `Poplar API Reference
  <https://docs.graphcore.ai/projects/poplar-api/en/latest/poplibs_api.html#poplin-matmul-hpp>`_.

* Compile and run the program. Note that you will need to add the PopLibs
  libraries ``poputil`` and ``poplin`` to the command line:

    .. code-block:: bash

      $ g++ --std=c++11 tut4.cpp -lpoplar -lpopops -lpoplin -lpoputil -o tut4

  The code already contains the ``addCodelets`` functions to add the device-side
  library code. See the `PopLibs section
  <https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplibs.html#using-poplibs>`_
  of the Poplar and PopLibs User Guide for more information.

When the program runs it prints profiling data. You should redirect this to a
file to make it easier to study.

Take some time to review and understand the execution profile. For example:

* Determine what percentage of the memory of the IPU is being used

* Determine how long the computation took

* Determine which steps belong to which matrix-multiply operation

* Identify how much time is taken by communication during the exchange phases

See the `Profile Summary
<https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/profiler.html#profile-summary>`_
section of the Poplar and PopLibs User Guide for further details of the profile summary.


Generating Profile Report Files
..............................................................

Creating a report:

* (Optional) First remove the ``printProfileSummary`` line in the tutorial code:

    .. code-block:: c++

      engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});

  This is not needed when using the PopVision Graph Analyser, but doesn't
  conflict with it either. Generating profile report files is enabled by
  environment variables.

* Recompile:

    .. code-block:: bash

      $ g++ --std=c++11 tut4.cpp -lpoplar -lpopops -lpoplin -lpoputil -o tut4

* Run with the following command, setting these environment variables:

    .. code-block:: bash

      $ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report"}' ./tut4

  - ``autoReport.all`` turns on all the default profiling options.
  - ``autoReport.directory`` sets the output directory, relative to the current directory.

* You should see a new directory ``report`` in your current directory.
  It will contain several files (``profile.pop``, ``vars.capnp`` and ``debug.cbor``)
  - the files that are created depends on which profiling options you have turned on.
  Making use of these files is explained in the following sections.


Using The PopVision Analysis API in C++ or Python
..................................................

This section explains how the PopVision analysis API (libpva) can be used to
query information from a profile file using C++ or Python. Please note, this is
a preview release of the PopVision analysis API and may change before final
release.

libpva is used to query ``profile.pop`` files, so copy your ``profile.pop`` file
created in the previous section to the ``tut4_profiling/libpva`` directory and
make this your working directory.

You should now see three files in your current working directory:

  - `CppExample.cpp` - Example C++ program that queries a profile.
  - `profile.pop` - Example profile file.
  - `PythonExample.py` - Example Python program that queries a profile.

Study the C++ and Python source files to understand how they work. Compile the
C++ program with:

    .. code-block:: bash

      $ g++ -g -std=c++11 CppExample.cpp -lpva -ldl -o CppExample

Now you can run the C++ program with:

    .. code-block:: bash

      $ ./CppExample

Or you can run the Python program with:

    .. code-block:: bash

      $ python3 PythonExample.py

Both programs should print the same example information similar to this:

    Example information from profile:
    Number of compute sets:  9
    Number of tiles on target:  1472
    Version of Poplar used:  2.0.0 (9c1df82ba0)

You may want to modify the source files to extend this example information.

Using PopVision Graph Analyser - Creating and loading a report
..............................................................

Download and install the PopVision Graph Analyser from the Downloads Portal:
`<https://downloads.graphcore.ai/>`_

It is also useful to watch the Getting Started with PopVision video
`<https://www.graphcore.ai/resources/how-to-videos>`_
both before the tutorial as a preview, and after to give you further things to try.

* Load the profile in the PopVision Graph Analyser.

  - You can either open a local copy of the ``report`` folder above, or open it remotely via ssh.
  - Launch the PopVision Graph Analyser, and click on ``'Open a Report..'`` .
  - Navigate to either the local or remote copy of the folder.
  - Click Open - this opens into the Summary tab, you can also open a
    specific file and it will take you straight to the corresponding tab.

* You should see the ``Summary`` tab:

  .. image:: screenshots/PopVision_GA_summary.png
    :width: 800

* There are multiple tabs that can be opened via the icons on the left hand side
  of the trace - ``Summary``, ``Memory``, ``Liveness``,
  ``Program Tree``, ``Computation Graph`` and ``Execution Trace``.
  The ``Execution Trace`` tab for example should look like:

  .. image:: screenshots/PopVision_GA_execution.png
    :width: 800

* Click through the different tabs and mouse around to investigate some of the functionality.
  Hovering over most things gives a tool tip or a link to the documentation.
  This documentation is contained both in the the application itself
  (``Help -> Documentation`` or the documentation icon, bottom left) and
  in the `PopVision User Guide.
  <https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/>`_

* The whole report can be reloaded via the reload icon (bottom left);
  closing the report and re-opening it (close icon, bottom left);
  or by directly opening a new file (``File -> Open New Window``).


Using PopVision Graph Analyser - General Functionality
......................................................

This section of the tutorial is an introduction to the basic functionality -
the PopVision User Guide gives full detailed instructions:
`<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide>`_

Capturing IPU Reports - setting ``POPLAR_ENGINE_OPTIONS``.
==========================================================

The amount and type of profiling data captured is set with the
``POPLAR_ENGINE_OPTIONS`` environment variable.
The default ``POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}'``
captures all the default profiling information apart from the serialized graph.

If you only want to collect specific aspects of the profiling data,
you can turn each one on individually:

  .. code-block:: bash

    $ POPLAR_ENGINE_OPTIONS='{"autoReport.outputGraphProfile":"true"}'

Conversely, if you want to exclude specific aspects, you can set ``autoReport.all`` to true,
and individually disable them:

  .. code-block:: bash

    $ POPLAR_ENGINE_OPTIONS='{"autoReport.outputGraphProfile":"true","autoReport.outputExecutionProfile":"false"}'

The environment variables can be made to persist using ``export``,
however common usage is to specify them on the same line as the
program to be profiled to scope them. Experiment with turning different
profiling functionality on and off. Note that the Poplar progam only overwrites
those files in t folder that correspond to the functionality turned on for that run.
So it won't delete files that aren't written in that run.

This is fully detailed in the `Capturing IPU Reports
<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/popvision.html#capturing-ipu-reports>`_
section of the PopVision Graph Analyser documentation.


Comparing two reports
=====================

Another useful function is the ability to compare two reports directly.
Instead of clicking ``'Open a Report…'`` in the main menu, simply click on
``'Compare two Reports…'``, navigate the file open windows to the two reports and click ``Compare``.
For this you'll need two reports, so modify the dimensions of one or more of the tensors,
for example m1 ``{800, 500} -> {1600, 700}``, m2 ``{500, 400} -> {500, 400}``.

Recompile and capture a second report to a second directory:

  .. code-block:: bash

    $ g++ --std=c++11 tut4.cpp -lpoplar -lpopops -lpoplin -lpoputil -o tut4
    $ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report_2"}' ./tut4

Compare the original report you created and your 2nd report. Look at the Summary,
Memory and Liveness tabs to start with. The Liveness tab for example should look like:

  .. image:: screenshots/PopVision_GA_liveness_2_reports.png
    :width: 800

We will use this extra report in the next couple of sections as well.

If you face any difficulties, a full walkthrough of opening reports is given in the `Opening IPU Reports
<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/popvision.html#opening-ipu-reports>`_
section of the PopVision Graph Analyser documentation.


Profiling an Out Of Memory program
==================================

If your program doesn't fit on the IPU tiles,
you will hit an Out Of Memory (OOM) error when compiling for hardware.
Usefully, one of the debug ``POPLAR_ENGINE_OPTIONS``:

  .. code-block:: bash

    POPLAR_ENGINE_OPTIONS='{"debug.allowOutOfMemory":"true"}'

allows us to still compile and profile that compilation, so we can investigate what happened to cause this.
So the equivalent command if our original example was OOM on hardware would be:

  .. code-block:: bash

    $ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report_OOM","debug.allowOutOfMemory":"true"}' ./tut4

It is important to note that if you run with ``"debug.allowOutOfMemory":"true"``
the compilation will finish but it won't succeed, and execution won't happen.
This means that even if you use ``"autoReport.all":"true"`` you won't get an execution trace.


Using PopVision Graph Analyser - Different tabs in the application
..................................................................

The next part of the tutorial takes a deeper look at each tab and the information they contain.

Memory profiling - investigating memory utilisation across the tiles
====================================================================

Open one of your reports from above, and click on the ``Memory`` tab icon on the left.

* You should see the ``Memory`` tab:

  .. image:: screenshots/PopVision_GA_memory.png
    :width: 800

  See how the Details section shows data for all tiles.

* With your mouse hovering over the graph, scroll with the mouse wheel
  up and down and see how this zooms in and out on regions of tiles.

* In the top right there is a ``Select Tile`` box - type in a tile you are
  interested in and see how the Details section shows details on just that specific tile.

  - You can enter two tile or more tile numbers, comma separated, to compare two or more different tiles.
  - You can also Shift-click on the lines of the graph to achieve the same behaviour.

*  In the top right there is also a set of options. Turn on ``Include Gaps`` and ``Show Max Memory``.

  - ``Show Max Memory`` shows the maximum available memory per tile - if 1+ tiles is over, it goes OOM.
  - ``Include Gaps`` shows the gaps in memory - some memory banks in IPU tiles are reserved for certain types of data.
    This leads to 'gaps' appearing in the tile memory.
  - The gaps can be enough to push you OOM, so it is useful to have both of these on when investigating an OOM issue.

* Compare your two reports, with ``Show Max Memory`` and ``Include Gaps`` turned on.

* Vary the tensors and the mapping and see the effects on the memory utilisation.

Full details of the Memory Report are given in the `Memory Report
<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/popvision.html#memory-report>`_
section of the PopVision Graph Analyser documentation.


Program Tree and Computation Graph (Serialized Graph) - visualising your compiled code
======================================================================================

Both the ``Computation Graph`` and ``Program Tree`` tabs allow you to visualise your compiled code.

The Program Tree shows a hierarchical view of the steps in the program that is run on the IPU.
Open one of your reports from above, and click on the Program Tree tab icon on the left.

* You should see the ``Program Tree`` tab:

  .. image:: screenshots/PopVision_GA_program_tree.png
    :width: 800

* Observe the sequences of stream copies, exchanges and on-tile-executions.
* Clicking on each line in the top panel gives full details in the bottom panel -
  observe the different info given for each type.


Serialized Graph capture is turned off by default because is has the potential to
create extremely large files on large models.
For this you'll need to re-capture with the serialized graph specifically enabled:

  .. code-block:: bash

   $ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report","autoReport.outputSerializedGraph":"true"}' ./tut4

The Computation Graph (Serialized Graph) shows a flow diagram of your compiled code.

* Look through the flow diagram and observe the dependencies in the code.
* Each stage can be clicked on and Details shown in the lower panel tabs.



More details on the Program Tree are given in the `Program Tree
<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/popvision.html#program-tree>`_
section of the PopVision Graph Analyser documentation.

Liveness Report
===============

This gives a detailed breakdown of the state of the variables at each step of your program.
Some variables persist in memory for the entirety of your program - these are known as 'Always Live' variables.
Some variables are allocated and deallocated as memory is reused - these are known as 'Not Always Live' variables.
While the Memory report does track this, the Liveness report visualises it.

Open one of your reports from above, and click on the ``Liveness`` tab icon on the left.

* You should see the ``Liveness`` tab:

  .. image:: screenshots/PopVision_GA_liveness.png
    :width: 800

* From the Options turn on ``Include Always Live``
* Click through different time steps, noting what details are given in the
  ``Always Live Variables`` / ``Not Always Live Variables`` / ``Vertices`` and
  ``Cycle Estimates`` tabs in the bottom panel.
* Note the program steps matching up with the Program Tree.

More details on the Liveness Report are given in the `Liveness Report
<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/popvision.html#liveness-report>`_
section of the PopVision Graph Analyser documentation.

Execution Trace
===============

This shows how many clock cycles each step of an instrumented program consumes.
Open one of your reports from above, and click on the ``Execution Trace`` tab icon on the left.

* You should see the ``Execution Trace`` tab:

  .. image:: screenshots/PopVision_GA_execution.png
    :width: 800

* Switch the ``Execution View`` between ``Flame`` and ``Flat``, and with ``BSP`` on and off.
* Observe the sync, exchange and execution code across the tiles.
* Observe how these correspond to the different operations, and in the program tree.
* Click on the ``Summary`` and ``Details`` tabs in the lower panel and observe the different information given for different operations.
* Note that all the measurements are in clock cycles not time.

More details on the Liveness Report are given in the `Execution Trace
<https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/popvision.html#execution-trace>`_
section of the PopVision Graph Analyser documentation.

Follow-ups
==========

Modify the tutorial code with extra operations and see the effects in the different tabs,
or try with your own code.

This tutorial shows how to profile Poplar programs, but using the PopVision Graph Analyser for
TensorFlow and PyTorch applications on the IPU is a case of setting the same environment variables.
This is described in the user guides of each framework.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
