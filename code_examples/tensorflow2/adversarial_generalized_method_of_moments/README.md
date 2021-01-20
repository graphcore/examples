# Adversarial Generalized Method of Moments

## Overview

This is an efficient reimplementation of the main algorithm 
described in the paper:
[Adversarial Generalized Method of Moments](https://arxiv.org/abs/1803.07164) 
by Greg Lewis and Vasilis Syrgkanis
The model has a variety of applications like balanced regression,
ordinary least squares, instrumental variables regression, 
maximum likelihood estimation, and non-linear least squares.


## Dataset

The dataset is generated while running the script.

## File Structure

The file `logging_utils.py` provides some improved logging capability.

| File                         | Description                                |
| ---------------------------- | ------------------------------------------ |
| `README.md`                  | How to run the model                       |
| `tf2_AdGMoM.py`              | Main algorithm script to run IPU model     |
| `AdGMoM_conf_default.yaml`   | Explains parameters and their defaults     |
| `logging_util.py`            | Logging functionality                      |
| `requirements.txt`           | Required Python packages                   |
| `test_AdGMoM.py`             | Test script. Run using `python -m pytest`  |


## Quick start guide

### 1) Download the Poplar SDK

Install the Poplar SDK following the instructions 
in the Getting Started guide for your IPU system.
Make sure to run the `enable.sh` script for Poplar.

### 2) Prepare the TensorFlow environment
 
Create and activate a Python virtualenv 
with the appropriate version of gc_tensorflow 
for TensorFlow 2 installed.
```
virtualenv venv -p python3.6
source venv/bin/activate
pip install gc_tensorflow-2.X.X+XXXXXX
```
Then install proper versions of numpy and tensorflow-probability by doing:
```
pip install -r requirements.txt
```

### 3) Execution

The main file, can be started using:

```
python tf2_AdGMoM.py
``` 

## Parameters

The main parameters of the algorithm are self-explanatory 
(after reading the paper) and can be changed
in the respective yaml file (`AdGMoM_conf_default.yaml`).
The file also provides some explanation of the parameters.

So far, the code has been tested only for a dimension of one 
for all the variables.
There is no full regularization framework added yet for the critic.
According to the 
[Minimax Estimation of Conditional Moment Models](https://arxiv.org/abs/2006.07201) 
follow up paper, regularization is important
to make the minmax problem behave more stable.

## License

This example is licensed under the MIT license - see the LICENSE file 
at the top-level of this repository.

This directory includes derived work from the following:

Adversarial Generalized Method of Moments, https://github.com/vsyrgkanis/adversarial_gmm

Copyright (c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.