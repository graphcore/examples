Graphcore
---
## IMDB Sentiment Classification

These examples train an IPU model with an embedding layer and an
LSTM to predict the sentiment of an IMDB review.

There are variants covering the use of `ipu.keras.Model` and `ipu.keras.Sequential`
for single IPU execution. Additionally, there are 2 IPU variants for
`ipu.keras.PipelineModel` and `ipu.keras.SequentialPipelineModel`.

They are derived from the example in the public Keras repository:
https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py

### File structure

* `imdb.py` Python script to train the 2 IPU pipelined model.
* `imdb_sequential.py` Python script to train the 2 IPU sequential pipelined model.
* `imdb_single_ipy.py` Python script to train the single IPU model.
* `imdb_single_ipu_sequential.py` Python script to train the single IPU sequential model.
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the Poplar SDK. Make sure to run the enable.sh script and activate a Python virtualenv with the TensorFlow 2 gc_tensorflow wheel installed.

2) Train the graph.

       python imdb.py
   
   Or a variant as above.

### Extra information

### Model

The model contains an embedding layer with a 20k dictionary, an LSTM layer, and
a projection down to a binary sentiment.

#### Options
There are no options for this script.
