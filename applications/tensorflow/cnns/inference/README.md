
Graphcore
---
## Image classification inference

Run inference using optimized data pipelining for image classification with pre-trained weights.
Optionally, this harness can generate compilation reports for these models.


The following models can be run on a single IPU using this inference harness.
1. ResNet50 (up to bs=8, fp16)
2. InceptionV1 (up to bs=8, fp16)
2. InceptionV3 (up to bs=2, fp16)
3. MobileNetV2 (up to bs=4, fp16)
4. MobileNet (up to bs=4, fp16)
5. EfficientNet-Edgetpu-S (up to bs=4, fp32)
6. EfficientNet-Edgetpu-M (up to bs=2, fp32)
7. EfficientNet-Edgetpu-L (up to bs=1, fp32)

The following models require multiple IPUs to be able to run using this inference harness.
1. DenseNet (up to bs=4 (with sharding=2), fp16)

### File structure

* `run_benchmark.py` Driver script for running inference.
* `get_images.sh` Script to fetch sample test images.
* `README.md` This file.
* `requirements.txt` Requirements for python packages.
* `get_weights.py` Script to fetch Keras pre-trained weights.
* `tests\test_get_weights.py` Test that weight download works.
* `data.py` Creates the tf.Dataset for handling images.
* `inference_network_base.py` Base class for creating an inference optimized graph.
* `inference_networks.py` Inference classes for each model.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the latest poplar-SDK following the README provided. Make sure to run the enable.sh scripts and 
   activate a Python virtualenv with gc_tensorflow installed.
   
   Install additional python packages specified in requirements.txt
  
    (gc_virtualenv)$ pip3 install -r requirements.txt
   
2) Download the images.

       ./get_images.sh

  This will create and populate the `images/` directory with sample test images.

3) Run the graph.

   To classify all images in a directory in a loop for 100 iterations,

       python run_benchmark.py resnet50 images --batch-size 1 --batches-per-step 100 --num-iterations 500


### Extra information

#### Options
The `run_benchmark.py` script has a few options. Use the `-h` flag or examine the code to understand them.

To run the demo on the Graphcore IPU simulator use the flag `--ipu-model`.

### Troubleshooting

If you see an error saying cannot load pywrap_tensorflow.so then TensorFlow can probably
not find the Poplar libraries. You need to have Poplar installed and referenced by
LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
