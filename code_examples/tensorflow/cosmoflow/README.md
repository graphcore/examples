Graphcore: CosmoFlow example
===
This README describes how to run a conv3D based model called CosmoFlow on IPU hardware with synthetic data.  

## File structure

* `configs` 	     folder containing the config .yaml file
* `data_gen`		     folder containing code for input data pipeline
* `models`   	     folder containing code defining graph models
* `README.md`	     The file you are reading now
* `requirements.txt`   Python dependencies
* `train.py`	     Main python script to launch and run the example
* `utils`	             folder containing utility functions
* `LICENSE`	     license document
* `test_cosmoflow.py`  unit test for CosmoFlow

## How to use this code example

1. Prepare the IPU environment.

 - Download, install & activate the Poplar SDK following instructions provided in the Getting Started document
	
 - Create python virtual environment, install the tensorflow-1 wheel from the Poplar SDK and active the Python
   environment. This is also documented in the Getting Started guide. This example uses TensorFlow v1, so
   please ensure your Python venv has TFv1 installed. 


2. Start running the example with one of the following commands:

 - run without tensorflow estimator, with 1 IPU:
   `python train.py configs/graphcore.yaml`

 - run without tensorflow estimator, with 2 IPUs:
   The workload is heavily IO bound, so merely increasing IPUs without increasing CPU numa-aware threads to pre-process
   the dataset will show marginal scalability. We use poprun to increase threads involved in processing
   `poprun --num-replicas 2 --num-instances 2 --ipus-per-replica 1 --numa-aware 1 python train.py configs/graphcore.yaml`

 - run with tensorflow estimator, with 1 IPU:
   `python train.py configs/graphcore.yaml --use-estimator`

 - run with tensorflow estimator, with 2 IPUs:
   `poprun --num-replicas 2 --num-instances 2 --ipus-per-replica 1 --numa-aware 1 python train.py configs/graphcore.yaml --use-estimator`

## Licensing

The code in this directory is licensed under the Apache License, Version 2.0. See the LICENSE file in this directory.

This directory includes derived work from https://github.com/sparticlesteve/cosmoflow-benchmark/ which is licensed under the Apache License, Version 2.0.

`models/resnet.py` includes derived work from <https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py> which is licensed under the MIT license. The notice below is reproduced from the keras-applications repository:

Copyright (c) 2016 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.
The initial code of this repository came from https://github.com/keras-team/keras
(the Keras repository), hence, for author information regarding commits
that occured earlier than the first commit in the present repository,
please see the original Keras repository.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


