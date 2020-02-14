# PopART Recomputing demo

> Copyright 2019 Graphcore Ltd.

This example runs synthetic data through a 4 layers DNN.
It shows how to use manual and automatic recomputation in popART.

## Usage

Run the demo

        python recomputation.py

## Options

The program has a few command-line options:

`--help` Show usage information

`--export FILE` Export the model created to FILE

`--recomputing STATUS` Choice amongst ON (default), AUTO and OFF

* ON equates to all output are recomputed during the backward pass
* AUTO uses popART auto recomputation which has its own heuristic to decide which ops are recomputed
* OFF deactivate recomputing altogether

`--show-logs` show execution logs
