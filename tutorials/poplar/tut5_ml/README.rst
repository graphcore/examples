Tutorial 5: a basic machine learning example
--------------------------------------------

This tutorial contains a complete training program that performs a logistic
regression on the MNIST data set, using gradient descent. The files for the demo
are in ``tut5_ml``. There are no coding steps in the tutorial. The task is to
understand the code, build it and run it. You can build the code using the
supplied makefile.

Before you can run the code you will need to run the ``get_mnist.sh`` script to
download the MNIST data.

The program accepts an optional command line argument to make it use the IPU
hardware instead of a simulated IPU.

As you would expect, training is significantly faster on the IPU hardware.

Copyright (c) 2018 Graphcore Ltd. All rights reserved.
