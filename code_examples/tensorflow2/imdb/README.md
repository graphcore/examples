Graphcore
---
## IMDB Sentiment Classification

This example trains a 2 IPU pipelined model with an embedding layer and an
LSTM to predict the sentiment of an IMDB review.

It is derived from the example in the public Keras repository:
https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py

### File structure

* `imdb.py` The main Python script.
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK. Make sure to run the enable.sh scripts and activate a Python virtualenv with the TensorFlow 2 gc_tensorflow wheel installed.

2) Train the graph.

       python imdb.py

### Extra information

### Model

The model contains an embedding layer with a 20k dictionary, an LSTM layer, and
a projection down to a binary sentiment.

#### Options
There are no options for this script.
