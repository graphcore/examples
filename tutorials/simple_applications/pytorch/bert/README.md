<!-- Copyright (c) 2020 Graphcore Ltd. All rights reserved. -->
# BERT Inference Using PopTorch

## Overview

This example shows how to use PopTorch to run inference on a pre-trained BERT model.
The pre-trained model is downloaded from [Hugging Face](https://huggingface.co/mrm8488/bert-medium-finetuned-squadv2) and compiled to run on the IPU.
The model has been already been fine tuned on the SQuADv2 corpus and is configured for a question-answering task.

Two text files are used to provide inputs to the network:
- `--context-file`, by default `context.txt` contains the context sequence
- `--questions-file`, by default `questions.txt`, contains the sequences of "questions" that are used as inputs by the model and then completed as "answers" as outputs to the model.

## Requirements

- A Poplar SDK environment enabled, with PopTorch installed (see the [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide for your IPU system)
- Python packages installed with `python -m pip install -r requirements.txt`

## Execution

The example can be run from the command line:
```:bash
python3 bert_inference.py
```
