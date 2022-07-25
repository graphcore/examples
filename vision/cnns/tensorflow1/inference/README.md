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

## Benchmarking

To reproduce the benchmarks, please follow the setup instructions in this README to setup the environment, and then from this dir, use the `examples_utils` module to run one or more benchmarks. For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml
```

or to run a specific benchmark in the `benchmarks.yml` file provided:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark <benchmark_name>
```

For more information on how to use the examples_utils benchmark functionality, please see the <a>benchmarking readme<a href=<https://github.com/graphcore/examples-utils/tree/master/examples_utils/benchmarks>

## Profiling

Profiling can be done easily via the `examples_utils` module, simply by adding the `--profile` argument when using the `benchmark` submodule (see the <strong>Benchmarking</strong> section above for further details on use). For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --profile
```
Will create folders containing popvision profiles in this applications root directory (where the benchmark has to be run from), each folder ending with "_profile". 

The `--profile` argument works by allowing the `examples_utils` module to update the `POPLAR_ENGINE_OPTIONS` environment variable in the environment the benchmark is being run in, by setting:
```
POPLAR_ENGINE_OPTIONS = {
    "autoReport.all": "true",
    "autoReport.directory": <current_working_directory>,
    "autoReport.outputSerializedGraph": "false",
}
```
Which can also be done manually by exporting this variable in the benchmarking environment, if custom options are needed for this variable.

### Extra information

### Troubleshooting

If you see an error saying cannot load pywrap_tensorflow.so then TensorFlow can probably
not find the Poplar libraries. You need to have Poplar installed and referenced by
LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.

# Licenses

The images provided by the `get_images.sh` script are attributed to employees of [Graphcore](https://graphcore.ai) and are provided permissively under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)

![Copyright Icon](https://i.creativecommons.org/l/by/4.0/88x31.png)
