# BERT inference using PopTorch


## Overview

This example shows how to use PopTorch to run inference on a pre-trained BERT model. 
The pre-trained model is downloaded from Hugging Face (https://huggingface.co/transformers/model_doc/bert.html) and compiled to run on the IPU. 
The model has been already been fine tuned on the SQuADv2 corpus and is configured for a question-answering task. 

Two text files are used to provide inputs to the network:
- `--context-file`, by default `context.txt` contains the context sequence  
- `--questions-file`, by default `questions.txt`, contains the sequences of "questions" that are used as inputs by the model and then completed as "answers" as outputs to the model.

## Installation

Install the Python dependencies by running:
```:python
python3 -m pip install -r requirements
```

## Configuration

Ensure the Poplar SDK 1.2.x is installed and configured to run in the current environment. 

## Execution

The example can be run from the command line:
```:bash
python3 bert_inference.py
```


