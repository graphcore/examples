Graphcore
---

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
* `send_request.py` TensorFlow Serving executor

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

       python run_benchmark.py resnet50 images --batch-size 1 --device-iterations 100 --num-iterations 500

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

### TensorFlow Serving example

Example of TensorFlow Serving usage can be found in `send_request.py` file. Script exports selected model to SavedModel format, initializes serving server, and sends images to server for predictions. Execution ends after given number of prediction requests. The model should be defined using the same options that have been provided to `run_benchmark.py` for classify.

Basic usage example for Resnet50 model export in batch-size 16 and serving in batch-size 8:

    `python3 send_request.py resnet50 images --batch-size 8 --model-batch-size 16 --port 8502 --num-threads 32`

### Additional arguments for send_request.py script

   --port PORT  
                        Serving service acces port
                        
   --batch-size BATCH_SIZE  
                        Size of data batch used in single prediction request. Might be smaller or equal to MODEL_BATCH_SIZE
                        
   --model-batch-size MODEL_BATCH_SIZE  
                        Batch size of exported model, should be equal or larger to BATCH_SIZE
                        
   --num-threads NUM_THREADS  
                        Number of threads/processes used for prediction requests, optimal value depends from used model and host system specification
                        
   --num-images NUM_IMAGES  
                        Number of image prediction requests
                        
   --serving-bin-path SERVING_BIN_PATH  
                        Path to TensorFlow serving binary file
                        
   --use-async  
                        When enabled client will send next prediction requests without blocking/waitg for server response. Each request returns `Future Object`.
                        
   --verbose  
                        Enables printing of each request execution time. Expect degradation in overall performace caused by printing
                        

### Extra information

### Troubleshooting

If you see an error saying cannot load pywrap_tensorflow.so then TensorFlow can probably
not find the Poplar libraries. You need to have Poplar installed and referenced by
LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.

# Licenses

The images provided by the `get_images.sh` script are attributed to employees of [Graphcore](https://graphcore.ai) and are provided permissively under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)

![Copyright Icon](https://i.creativecommons.org/l/by/4.0/88x31.png)
