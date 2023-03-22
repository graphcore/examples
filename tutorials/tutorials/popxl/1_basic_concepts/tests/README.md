<!-- Copyright (c) 2022 Graphcore Ltd. All rights reserved. -->
# `tut1_basic_concepts` tests

To run the tests:

- Create a Python 3 virtual environment with Poplar and install popxl.addons, as
explained in tut1_basic_concepts/README
- In the same environment, install the script requirements `pip3 install -r
  tut1_basic_concepts/requirements.txt`
- Install test-specific requirements in this folder `pip3 install -r
  requirements.txt`
- Run the tests: `python -m pytest`
- The tests can also be run in parallel by calling: `pytest -n4 --forked`
