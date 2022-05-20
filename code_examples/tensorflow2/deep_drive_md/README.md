# Convolutional Variational Autoencoder (CVAE) for DeepDriveMD experiment

This example uses the Convolutional Variational Autoencoder (CVAE) prepared by Cray Labs as a part of adoption of [DeepDriveMD](https://github.com/CrayLabs/smartsim-openmm) experiment to HPE smartsim HPC2AI pipeline.

The CVAE model is an AI surrogate model from DeepDriveMD experiment's HPC+AI pipeline described originally in [DeepDriveMD: Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding](https://arxiv.org/pdf/1909.07817.pdf).

We have adopted the CVAE model to IPU basing on Cray Labs source code.

# Physical background

From the physics perspective the CVAE AI model is involved in protein folding which is one of the important problems of molecular dynamics. It aims to answer how to translate protein’s amino acid sequence (unfolded state) into its final three-dimensional atomic structure (folded state).
In DeepDriveMD experiment deep convolution variational autoencoder (CVAE) has been used to cluster (group conformations sharing similar structural and energetic features) trajectories into a small number of conformational states allowing to determine novel points in way alternative to traditional.

## Usage

Install the Poplar SDK following the the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` script for poplar.  

1. Create the virtual environment using python 3.6 `python3.6 -m venv my_tf2_venv`
2. Activate the virtual environment `source my_tf2_venv/bin/activate`
3. Update pip: `python -m pip install -U pip`
4. Install the Graphcore TensorFlow wheel: `pip install tensorflow-2*`
5. Install the requirements: `pip install -r requirements.txt`
6. Run the example: `python train_cvae.py`

The above script will run the training with randomly generated dataset basing on default parameters.
The default parameters might be overridden with input argument list, e.g:
`python train_cvae.py --batch_size 100 --num_epochs 8  --img_size 20 --dataset_size 10000`

References:
* “[DeepDriveMD: Deep-Learning Driven Adaptive Molecular Simulations for Protein Folding](https://arxiv.org/pdf/1909.07817.pdf)”Hyungro Lee, Heng Ma, Matteo Turilli, Debsindhu Bhowmik, Shantenu Jha, Arvind Ramanathan

## License

This example is licensed under the MIT license - see the LICENSE file at the top-level of this repository.

This directory includes derived work from the following:

https://github.com/CrayLabs/smartsim-openmm

for which the following license applies:

BSD 2-Clause License

Copyright (c) 2021, Cray Labs
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
