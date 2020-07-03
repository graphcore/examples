# PopART Pipelining demo

> Copyright 2020 Graphcore Ltd.

This demo shows how to use pipelining in PopART on a very simple model
consisting of two dense layers. Run one pipeline length and compute loss.

## File structure

* `pipelining.py` The main PopART file showcasing pipelining.
* `test_pipelining.py` Test script.
* `README.md` This file.

## How to use this demo

1) Prepare the environment.

   Install the `poplar-sdk` following the README provided. Make sure to source the `enable.sh`
    scripts for poplar, gc_drivers (if running on hardware) and popart.

2) Run the graph. Note that the PopART Python API only supports Python 3.

    python3 pipelining.py [-h] [--export FILE] [--report] [--no_pipelining] [--test]

### Options

Run pipelining.py with -h option to list all the command line options.

### Running the tests

Install the needed package and use pytest.

```cmd
pip install -r requirements.txt
pytest -s
```
