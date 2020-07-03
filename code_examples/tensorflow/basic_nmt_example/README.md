Graphcore
---
## Basic NMT Example Code Demo

This example trains an NMT model to translate human dates ("4th jan 2018") to computer dates ("2018.01.04"). The aim is to have a very simple example showing TensorFlow targeting IPU devices.

### File structure

* `nmt-tf.py` The main Python script.
* `seq2seq_edits` Directory containing edits to the tf.contrib.seq2seq library to remove unsupported ops for the IPU
* `utils.py` Helper functions.
* `README.md` This file.

### How to use this demo

1) Prepare the TensorFlow environment.

   Install the poplar-sdk following the README provided. Make sure to run the enable.sh scripts and activate a Python virtualenv with gc_tensorflow installed.


2) Python

Create a virtualenv and install the required packages:

```
virtualenv venv -p python3.6
source venv/bin/activate
pip3 install -r requirements.txt
```

3) Generate the data and vocabs.

   You will need access to internet since it will clone a third party repository.

       ./generate_data.sh

4) Train the graph.

       python nmt-tf.py

  This will save checkpoints in the `weights` directory.

5) Test the graph.

       python nmt-tf.py --infer

### Extra information

### Model

By default, the demo runs a single layer encoder and decoder, with 128 hidden units and Luong attention.

#### Options
The `nmt-tf.py` script has a few options. Use the `-h` flag or examine the code to understand them.
