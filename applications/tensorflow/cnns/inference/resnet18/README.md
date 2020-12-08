
Graphcore
---
## ResNet-18 Inference Demo

This example uses a pre-trained network to perform inference on images using a port
of TensorFlow that targets Graphcore's Poplar libraries.

### File structure

* `classify_images.py` The main Python script.
* `get_images_and_weights.sh` Script to fetch the images and weights.
* `imagenet_categories.py` Dictionary of the ImageNet category strings.
* `README.md` This file.
* `resnet.py` The model definition.
* `utils.py` Helper functions for building the graph.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK following the instructions in the Getting Started guide
   for your IPU system. Make sure to run the enable.sh script and activate a
   Python virtualenv with gc_tensorflow installed.

2) Download the images and weights.

       ./get_images_and_weights.sh

  This will create and populate the `images/` and `weights/` directories.

2) Run the graph. For example to classify a single image:

       python classify_images.py images/zebra.jpg

   To classify all images in a directory you can use

       python classify_images.py images/

   Note that ResNet v1 requires images which are 224x224 pixels.

### Extra information

#### Options
The `classify_images.py` script has a few options. Use the `-h` flag or examine the code to understand them.

To endlessly loop through all the images use `--loop`.

#### Running on IPU Model
To run the inference demo on the Graphcore IPU Model, run with `TF_POPLAR_FLAGS="--use_ipu_model"`:

         TF_POPLAR_FLAGS="--use_ipu_model" python classify_images.py images/zebra.jpg

#### Interactive use
The module can also be used in a Python interactive session. For example:

    >>> from classify_images import ImageClassifier
    >>> ic = ImageClassifier()
        ... wait while graph is compiled ...
    >>> ic.classify_image('images/zebra.jpg')

    Filename : zebra.jpg
    Class 340: zebra 98.3%
    Class 292: tiger, Panthera tigris 0.905%
    Class 353: gazelle 0.186%
    Class 282: tiger cat 0.175%
    Class 352: impala, Aepyceros melampus 0.0706%

### Troubleshooting

If you see an error saying cannot load pywrap_tensorflow.so then TensorFlow can probably
not find the Poplar libraries. You need to have Poplar installed and referenced by
LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
