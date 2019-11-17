
Graphcore
---
## Densenet-121 Inference Demo

This example uses a pre-trained Densenet121-network to perform inference on images using a port
of TensorFlow that targets Graphcore's Poplar libraries.

### File structure

* `densenet_inference.py` Driver script for running inference.
* `get_images.sh` Script to fetch sample test images.
* `README.md` This file.
* `requirements.txt` Requirements for python packages.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the latest poplar-SDK following the README provided. Make sure to run the enable.sh scripts and 
   activate a Python virtualenv with gc_tensorflow installed.
   
   Pillow is an additional requirement to use keras based image pre-processing.
  
  
   ```
        (gc_virtualenv)$ pip3 install pillow
   ```
   
   Make sure the root directory of this repository is in the `PYTHONPATH`.
   
  ```
          cd $PATH_TO_REPO_ROOT
          export PYTHONPATH=$PWD:$PYTHONPATH
  ```
    

2) Download the images.

       ./get_images.sh

  This will create and populate the `images/` directory with sample test images.
  
2) Run the graph. 

   To classify all images in a directory you can use

       python densenet_inference.py images/


### Extra information

#### Options
The `densenet_inference.py` script has a few options. Use the `-h` flag or examine the code to understand them.

To run the demo on the Graphcore IPU simulator use the flag `--ipu-model`.
To endlessly loop through the images use `--loop`.


### Troubleshooting

If you see an error saying cannot load pywrap_tensorflow.so then TensorFlow can probably
not find the Poplar libraries. You need to have Poplar installed and referenced by
LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.