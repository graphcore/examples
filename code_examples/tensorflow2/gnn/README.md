# Graph Neural Network

This example uses the [Spektral](https://github.com/danielegrattarola/spektral) GNN library to predict the heat capacity of various molecules in the [QM9 dataset](http://quantum-machine.org/datasets/).

In particular this example shows [Edge Conditioned Convolutional Networks](https://arxiv.org/abs/1704.02901) but any dataset or network from the Spektral library should work the same.

## Usage

Install the Poplar SDK following the the instructions in the Getting Started guide for your IPU system. Make sure to source the `enable.sh` script for poplar.  

<<<<<<< HEAD
```python3 -m pip install -U pip```

```pip3 install tensorflow-2*```

```pip3 install -r requirements.txt```

```python3 qm9_ipu.py```
=======
1. Update pip: ```python3 -m pip install -U pip```
2. Install the Graphcore Tensorflow wheel: ```pip3 install tensorflow-2*```
3. Install the Spektral GNN library requirements: ```pip3 install -r requirements.txt```
4. Run the example: ```python3 qm9_ipu.py```
>>>>>>> ss

The above script will download the [QM9 dataset](http://quantum-machine.org/datasets/) automatically which is just about 40MB in size.

Dataset references:
* L. C. Blum, J.-L. Reymond, [970 Million Druglike Small Molecules for Virtual Screening in the Chemical Universe Database GDB-13](https://pubs.acs.org/doi/10.1021/ja902302h), J. Am. Chem. Soc., 131:8732, 2009
* M. Rupp, A. Tkatchenko, K.-R. Müller, O. A. von Lilienfeld: [Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301), Physical Review Letters, 108(5):058301, 2012

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
