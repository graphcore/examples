Graphcore
---
## Basic NMT Example Code Demo

This example trains an NMT model to translate human dates ("4th jan 2018") to computer dates ("2018.01.04"). The aim is to have a very simple example showing TensorFlow targeting IPU devices.

### File structure

* `nmt-tf.py` The main Python script.
* `data_gen/generate.py` Script to generate the data with faker.
* `data_gen/reader.py` Data classes for pre-processing the generated data.
* `seq2seq_edits` Directory containing edits to the tf.contrib.seq2seq library to remove unsupported ops for the IPU
* `utils.py` Helper functions.
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the poplar-sdk following the README provided. Make sure to run the enable.sh scripts and activate a Python virtualenv with gc_tensorflow installed.

2) Generate the data and vocabs.

       pip install faker babel
       python data_gen/generate.py

3) Train the graph.

       python nmt-tf.py

  This will save checkpoints in the `weights` directory.

4) Test the graph.

       python nmt-tf.py --inf

### Extra information

### Model

By default, the demo runs a single layer encoder and decoder, with 128 hidden units and Luong attention.

#### Options
The `nmt-tf.py` script has a few options. Use the `-h` flag or examine the code to understand them.
