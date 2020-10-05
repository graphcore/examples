Graphcore
---
## Shakespeare corpus reader

This example learns to predict the next character in the corpus of
William Shakespeare.

It is derived from the example in Google Cloud TPU CoLab.
https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb

### File structure

* `shakespeare.py` The main Python script.
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK. Make sure to run the enable.sh scripts and activate a Python virtualenv with the TensorFlow 2 gc_tensorflow wheel installed.

2) Acquire the training material.

       wget --continue -O shakespeare.txt http://www.gutenberg.org/files/100/100-0.txt

2) Train the graph.

       python shakespeare.py

### Extra information

### Model

The model contains an embedding layer converting from the ASCII character space
into a 128 dimensional embedding vector.  It has a 2 layer stack of LSTMs, with
a final time-series Dense layer to project back into ASCII.

It trains by comparing the predicted character to the known next character.

#### Options
There are no options for this script.
