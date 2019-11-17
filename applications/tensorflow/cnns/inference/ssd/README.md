Graphcore
---
## SSD Single Image Example

This is implementation of the Single Shot MultiBox Detector (SSD) model using a dual-device, single-graph
framework for inference applications. 

## Overview

This model implements SSD using the original VGG-16 input backbone as published by Liu et al in 

https://arxiv.org/abs/1512.02325

The model uses a segregated approach where the convolutional components of the model are deployed on the IPU,
while the decoding component lives on host. The model is currently implemented with *in-feeds* for performance 
benchmarking, but a single-image processing version of the script (*ssd_single_image.py*) is included that can be
adapted for real-world prediction/classification applications. 


## Notes on implementation

The current work is heavily derived from the code base presented by Pierluigi Ferrari in his Github repository:

https://github.com/pierluigiferrari/ssd_keras

In his original implementation, Keras abstraction was used throughout. In this adaption, native Tensorflow libraries
were used directly to leverage Poplar functionality. 

In addition, due to compatibility conflicts between Tensorflow's XLA compiler and various components of the decoder, 
along with the decoder's low compute requirements, this facet of the model was moved to host. The result is a dual-
device model in which the IPU is tasked with all convolutional operations, while the host delivers the final
decoding required for detection and classification of the objects within the images.

## How to run the script

1) Download the *h5py* and *pillow* libraries:
    
    Besides the usual suspects of required libraries, (i.e. *Numpy* and *Matplotlib*), the *hp5y* library is used 
    to read the trained weight file in those cases where actual detections are desired, and the *pillow* library for
    image loading and processing. (If no actual detections are required, please now skip to the *Run Script* section
    below.)
    
```
    (gc_virtualenv)$ pip3 install h5py pillow
```

2) Download the trained weight file

    If trained weights are desired, these can be found here:
    
    https://drive.google.com/file/d/121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q/view
    
    Place the downloaded *.h5* file (HDF5 format) in the 'trained_weights' folder and switch the RANDOM_WEIGHTS flag to
    false, which is at line 50 of the script. 

3) Run the script

    To run the script, switch to your Graphcore virtual-environment and run:
    
```
    (gc_virtualenv)$ python ssd_model.py
```

### Running a single image detection

A replicated version of this code is provided in *ssd_single_image.py*, that does not use *in-feeds* and can be
run on a single image to test actual detection/prediction capabilities of the model. To run it, use

    ```
        (gc_virtualenv)$ python ssd_single_image.py
    ```
An example output of the script is given in 

    ```
       ./example_images/example_detection.png)
    ```
### Options

Here is a list of the various flags that can be set:

* 'IPU_MODEL': use the IPU-simulator which runs on host

* 'REPORT': generate a *GC-Profile* compatible report

* 'SAVE_IMAGE' save a detection to a *png* file

* 'RANDOM_WEIGHTS': use randomly generated weights, or load a trained weight file
