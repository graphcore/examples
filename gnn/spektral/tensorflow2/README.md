# Spektral
GNN for molecular heat capacity prediction using the [Spektral](https://github.com/danielegrattarola/spektral) library, optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| TensorFlow 2 | GNNs | Spektral | QM9 |  | <p style="text-align: center;">✅ <br> Min. 1 IPU (POD4) required | <p style="text-align: center;">❌ | [Edge Conditioned Convolutional Networks](https://arxiv.org/abs/1704.02901) |


## Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)

3. Download the MolHIV dataset (See Dataset setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the TensorFlow 2 and IPU TensorFlow add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
For the CPU architecture you are running on

4. Build the custom ops:
```bash
cd static_ops && make
```


More detailed instructions on setting up your TensorFlow 2 environment are available in the [TensorFlow 2 quick start guide](https://docs.graphcore.ai/projects/tensorflow2-quick-start).

## Dataset setup
### QM9
Download from the [QM9 dataset source](https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9), or let the scripts download the dataset automatically when needed.

Disk space required: 283MB

```bash
.
├── dsgdb9nsd_000001.xyz
    .
    .
    .
└── dsgdb9nsd_133885.xyz

133885 files
```

Dataset references:
* L. C. Blum, J.-L. Reymond, [970 Million Druglike Small Molecules for Virtual Screening in the Chemical Universe Database GDB-13](https://pubs.acs.org/doi/10.1021/ja902302h), J. Am. Chem. Soc., 131:8732, 2009
* M. Rupp, A. Tkatchenko, K.-R. Müller, O. A. von Lilienfeld: [Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301), Physical Review Letters, 108(5):058301, 2012


## Usage
Run the example:
```bash
python3 qm9_ipu.py
```


## License

This example is licensed under the MIT license - see the LICENSE file at the top-level of this repository.

This directory includes derived work from the following:

Spektral, https://github.com/danielegrattarola/spektral

MIT License

Copyright (c) 2019 Daniele Grattarola

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
