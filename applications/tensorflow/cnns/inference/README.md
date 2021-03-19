Graphcore
---

## Benchmarking

Inference benchmarking is run using the training harness. To reproduce the published Mk2 inference throughput benchmarks, please follow the setup instructions in [../training/README.md](../training/README.md), and then follow the instructions in [../training/README_Benchmarks.md](../training/README_Benchmarks.md) 

## Image classification inference


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

The application example downloads pre-trained weights using the `tensorflow.compat.v1.keras.applications` API.
Further details can be found [on the TensorFlow website](https://www.tensorflow.org/api_docs/python/tf/keras/applications/).

InceptionV1 pre-trained weights are available from [this TensorFlow link](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) from the [TensorFlow models repository](https://github.com/tensorflow/models/).

EfficentNet pre-trained weights are available from the [TensorFlow TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu).

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

   Install the Poplar SDK following the the instructions in the Getting Started
   guide for your IPU system. Make sure to run the enable.sh script and
   activate a Python virtualenv with tensorflow-1 wheel from the Poplar SDK installed.

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
