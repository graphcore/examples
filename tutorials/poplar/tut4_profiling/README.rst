Tutorial 4: profiling output
----------------------------

Make a copy of the file ``tut4_profiling/start_here/tut4.cpp`` and open it in an
editor.

* Use the ``matMul`` function from the ``poplin`` library to extend this
  example to calculate ((m1 * m2) * m3). The ``matMul`` function is documented
  in the `Poplar API Reference
  <https://www.graphcore.ai/docs/poplar-api-reference#poplin-matmul-hpp>`_.

* Compile and run the program. Note that you will need to add the Poplibs
  libraries ``poputil`` and ``poplin`` to the command line:

  .. code-block:: bash

    $ g++ --std=c++11 tut4.cpp -lpoplar -lpopops -lpoplin -lpoputil -o tut4

  The code already contains the ``addCodelets`` functions to add the device-side
  library code. See the `Poplar and Poplibs User Guide
  <https://www.graphcore.ai/docs/poplar-and-poplibs-user-guide#using-poplibs>`_
  for more information.

When the program runs it prints profiling data. You should redirect this to a
file to make it easier to study.

Take some time to review and understand the execution profile. For example:

  * Determine what percentage of the memory of the IPU is being used

  * Determine how long the computation took

  * Determine which steps belong to which matrix-multiply operation

  * Identify how much time is taken by communication during the exchange phases

See the `Poplar and Poplibs User Guide
<https://www.graphcore.ai/docs/poplar-and-poplibs-user-guide#document-profiler>`_
for an description of the profile summary.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
