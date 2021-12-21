# Reference code for RNN-Transducer loss implementation

The corresponding publication is [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711).

The code in this folder is derived from [A Fast Sequence Transducer Implementation with PyTorch Bindings](https://github.com/awni/transducer).

This code is meant to work with Python 3.6 and PyTorch 1.3.0. 

## Install and Test

First create a virtual environment and install PyTorch: 
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip install torch==1.3.0
```

You need pytorch version 1.3.0. The pytorch from poplar SDK is not compatible with this reference code.

Then from this folder, run

```
python setup.py install
```

And test the reference code with
```
python test.py
```

Note. If you see errors, make sure you have `torch==1.3.0` installed, remove the build directory and rebuild the library.
```
rm -rf build
python setup.py install
```

Note. To build the debug version of RNN-Transducer, set environment variable DEBUG_TRANSDUCER to 1
```
export DEBUG_TRANSDUCER=1
```


