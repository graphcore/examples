# Poplar prefetching callbacks

The example program `prefetch.cpp` defines a poplar program that runs multiple times.
The program consists on a repeated sequence:

    Copy from input stream to tensor.
    Modify tensor value
    Copy tensor to output stream.

The input stream is connected to a prefetch-able callback and all necessary
engine options are set so that the prefetch function is used.

All callback functions print a message to standard output to see what is going
on.

