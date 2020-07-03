# PopART Recomputing demo

> Copyright 2019 Graphcore Ltd.

This example runs generated data through a seven layer DNN.
It shows how to use manual and automatic recomputation in popART.

Manual recomputing allows the user to choose which ops to recompute.

Automatic recomputing uses an heuristic technique.
See https://arxiv.org/abs/1604.06174


## Usage

install the needed packages, then use pytest to run the example in all three recomputation modes.

        pip install -r requirements.txt
        pytest -v -s

## Options

The program has a few command-line options:

`--help` Show usage information

`--export FILE` Export the model created to FILE

`--recomputing STATUS` Choice amongst ON (default), AUTO and OFF

* ON recomputes activations for all but checkpointed layers
* AUTO uses popART auto recomputation
* OFF deactivate recomputing altogether

`--show-logs` show execution logs

## Run the tests

        pip install -r requirements.txt
        pytest
