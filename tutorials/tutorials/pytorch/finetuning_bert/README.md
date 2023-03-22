<!-- Copyright (c) 2021 Graphcore Ltd. All rights reserved. -->
# BERT Fine-tuning on the IPU

This tutorial demonstrates how to fine-tune a pre-trained BERT model with
PyTorch on the Graphcore IPU-POD16 system. It uses a BERT-Large model and
fine-tunes it on the SQuADv1 Question/Answering task. The tutorial is in
[Fine-tuning-BERT.ipynb](./Fine-tuning-BERT.ipynb).

## File structure

- `README.md` This file;
- `Fine-tuning-BERT.ipynb` The tutorial jupyter notebook;
- `Fine-tuning-BERT.py` Python script conversion of the jupyter notebook;
- `squad_preprocessing.py` Contains a number of utility functions to prepare
   the data;
- `requirements.txt` Required packages;
- `tests/test_finetuning_notebook.py` Script for testing this tutorial;
- `tests/requirements.txt` Required packages for the tests;
- `LICENSE` Apache 2.0 license file (applies only to `squad_preprocessing.py`).

## How to use this demo

1. Prepare the PopTorch environment.

   Install the Poplar SDK following the instructions in the Getting Started
   guide for your IPU system. Make sure to run the enable.sh script for Poplar
   and activate a Python virtualenv with a PopTorch wheel from the Poplar SDK
   installed (use the version appropriate to your operating system).

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the jupyter notebook and connect to it with your browser. You may need
   to use an SSH tunnel to tunnel `jupyter` back to your local machine using:
   `ssh -L 8888:localhost:8888 [REMOTE-IPU-MACHINE] -N`.

    ```bash
    jupyter notebook
    ```

## License

The file `squad_preprocessing.py` is based on code from Hugging Face licensed
under Apache 2.0 so is distributed under the same license (see the LICENSE
file in this directory for more information).

The rest of the code in this example is licensed under the MIT license - see
the LICENSE file at the top level of this repository.
