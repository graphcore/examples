<!-- Copyright (c) 2018 Graphcore Ltd. All rights reserved. -->
# Graphcore Tutorials

This folder contains tutorials, feature examples and simple applications to
help you learn how to use Graphcore IPUs.

If you encounter a problem or want to suggest an improvement to our tutorials
please raise a Github issue, contact us at
[support@graphcore.ai](mailto:support@graphcore.ai?subject=General%20Feedback), or
get in touch through the #help channel of our slack community!

[![Join our Slack Community](https://img.shields.io/badge/Slack-Join%20Graphcore's%20Community-blue?style=flat-square&logo=slack)](https://www.graphcore.ai/join-community)

The latest version of the documentation for the Poplar software stack, and other
developer resources, is available at <https://www.graphcore.ai/developer>.

Please install and enable the Poplar SDK following the instructions in the
[Getting Started guide for your IPU
system](https://docs.graphcore.ai/en/latest/getting-started.html).

Unless otherwise specified by a LICENSE file in a subdirectory, the LICENSE
referenced at the top level applies to the files in this folder.

## Folder contents

### Tutorials

The [tutorials/](tutorials) folder contains tutorials to help you get started
using the Poplar SDK and Graphcore tools.

* [tutorials/poplar](tutorials/poplar) - A set of tutorials to introduce the
  Poplar graph programming framework and the PopLibs libraries.
* [tutorials/pytorch](tutorials/pytorch) - A set of tutorials to introduce the
  PyTorch framework support for the IPU.
* [tutorials/tensorflow2](tutorials/tensorflow2) - A set of tutorials to
  introduce the TensorFlow 2 framework support for the IPU.
* [tutorials/popvision](tutorials/popvision) - A set of tutorials to introduce
  PopVision, our suite of graphical application analysis tools.
* [tutorials/popxl](tutorials/popxl) - A set of tutorials introducing PopXL, a
  Python module with which you can create models directly in PopART's
  intermediate representation.
* [tutorials/standard_tools](tutorials/standard_tools) - Explanations of how to
  use standard deep learning tools with the Graphcore IPU.

A complete list of available tutorials can be found in the
[tutorials/](tutorials) folder.

The README files for the tutorials are best viewed on GitHub.

### Feature examples

The [feature_examples/](feature_examples) folder contains small code examples
showing you how to use various software features when developing for IPUs. See
the READMEs in each folder for details.

### Simple application examples

The [simple_applications/](simple_applications) folder contains example
applications written in different frameworks targeting the IPU. See the READMEs
in each folder for details on how to use these applications.

### Code used in technical notes, videos and blogs

The [tech_notes_code/](tech_notes_code), [videos_code/](videos_code) and
[blogs_code/](blogs_code) folders contain code used in Graphcore technical notes,
videos and blogs (respectively).

### Utilities

The [utils/](utils) folder contains utilities libraries and scripts.
