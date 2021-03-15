Graphcore
---
## SSD Single Image Example

This is an implementation of the Single Shot MultiBox Detector (SSD) model using a dual-device, single-graph
framework for inference applications.

## Overview

This model implements SSD using the original VGG-16 input backbone as published by Liu et al in

https://arxiv.org/abs/1512.02325

The model uses a segregated approach where the convolutional components of the model are deployed on the IPU,
while the decoding component lives on host. The model uses infeed and outfeed queues for performance
benchmarking; a single-image processing version of the script (*ssd_single_image.py*) is included that can be
adapted for real-world prediction/classification applications.

## License

This example is licensed under the Apache License 2.0 - see the LICENSE file in this directory.

This directory contains derived work from the following:

Pierluigi Ferrari's Single-Shot MultiBox Detector implementation in Keras,
https://github.com/pierluigiferrari/ssd_keras

Copyright 2018 Pierluigi Ferrari.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Notes on implementation

In the original implementation, Keras abstraction was used throughout. In this derived work, native TensorFlow libraries
were used directly to leverage Poplar functionality.

In addition, due to compatibility conflicts between TensorFlow's XLA compiler and various components of the decoder,
along with the decoder's low compute requirements, this facet of the model was moved to host. The result is a dual-
device model in which the IPU is tasked with all convolutional operations, while the host delivers the final
decoding required for detection and classification of the objects within the images.

## How to run the scripts

1) Prepare the TensorFlow environment

   Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to run the enable.sh script and
   activate a Python3 virtualenv with the tensorflow-1 wheel from the Poplar SDK installed.

2) Install the required libraries:

    In addition to *Numpy* and *Matplotlib*, the *hp5y* library is used
    to read the trained weight file in those cases where actual detections are desired, and the *pillow* library for
    image loading and processing. (If no actual detections are required, please now skip to the *Run Script* section
    below.)

        (gc_virtualenv)$ pip3 install -r requirements.txt

3) Download the trained weight file

    If trained weights are desired, they can be found here:

    https://drive.google.com/file/d/121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q/view

    (link from the README in https://github.com/pierluigiferrari/ssd_keras; author of the trained weights is unknown.)

    Place the downloaded *.h5* file (HDF5 format) in the 'trained_weights' folder and switch the RANDOM_WEIGHTS flag to
    false, which is at line 50 of the script.

4) Run the script

    From your virtual environment run:

        (gc_virtualenv)$ python ssd_model.py

### Running a single image detection

A alternative version of this code is provided in *ssd_single_image.py*, that does not use infeed and outfeed queues and can be
run on a single image to test actual detection/prediction capabilities of the model. To run it, use

    (gc_virtualenv)$ python ssd_single_image.py

An example output of the script is given in

    ./example_images/example_detection.png

### Options

Here is a list of the various flags that can be set within the scripts:

* 'IPU_MODEL': use the IPU simulator which runs on the host

* 'SAVE_IMAGE' save a detection to a *png* file (use a trained weight file in this case)

* 'RANDOM_WEIGHTS': use randomly generated weights, or load a trained weight file

* 'REPORT': set to True in order to generate profiling information that can be viewed using
    the PopVision Graph Analyser. The execution report is only generated for the
    *ssd_single_image.py* script.

* 'REPORT_DIR': set the location for the profiling information
