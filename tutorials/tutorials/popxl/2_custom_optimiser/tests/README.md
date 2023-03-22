<!-- Copyright (c) 2022 Graphcore Ltd. All rights reserved. -->
# `tut2_custom_optimiser` tests

To run the tests:

- Create a Python 3 virtual environment with Poplar and install popxl.addons, as
  explained in [the tutorial](../).
- In the same environment, install the script requirements: `pip3 install -r
  tut0_basic_concepts/requirements.txt`
- Install test-specific requirements in this folder: `pip3 install -r
  requirements.txt`
- Run the tests: `python -m pytest`
- The tests can also be run in parallel by calling: `pytest -n4 --forked`
