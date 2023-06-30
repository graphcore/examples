# Multi-Horizon Financial Forecasting on IPU using DeepLOB - Training with TensorFlow2

| Framework   | Domain  | Model   | Datasets | Tasks                 | Training | Inference | Reference |
|-------------|---------|---------|----------|-----------------------|----------|-----------|-----------|
| TensorFlow2 | Finance | DeepLOB | FI-2010  | Financial Forecasting | <div style="text-align: center;">✅ | <div style="text-align: center;">✅ | [https://arxiv.org/abs/2105.10430](https://arxiv.org/abs/2105.10430) |

 <!-- ## Paper: ["Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration using Intelligent Processing Units"](https://arxiv.org/abs/2105.10430) -->

### Original Authors: Zihao Zhang and Stefan Zohren
### Oxford-Man Institute of Quantitative Finance, Department of Engineering Science, University of Oxford

Copyright (c) 2021 Oxford Man Institute & University of Oxford. All rights reserved.

Copyright (c) 2023 Graphcore Ltd. All rights reserved.

These Jupyter notebooks have been modified by Graphcore Ltd in order to run with the latest version of the Poplar (TM) SDK

### Methods

This jupyter notebook is used to demonstrate the machine learning methods for multi-horizon forecasting for limit order books, shown in [2], implemented using TensorFlow2. The publicly available FI-2010 [1] dataset is used for model training, validation and inference.

### Data:
The FI-2010 is publicly available and interested readers can check out the paper [1]. The dataset can be downloaded from: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

The notebooks will download the data automatically; alternatively it may be obtained at the following URL:

https://drive.google.com/drive/folders/1Xen3aRid9ZZhFqJRgEMyETNazk02cNmv?usp=sharing.

### References:
[1] Ntakaris A, Magris M, Kanniainen J, Gabbouj M, Iosifidis A. Benchmark dataset for mid‐price forecasting of limit order book data with machine learning methods. Journal of Forecasting. 2018 Dec;37(8):852-66. https://arxiv.org/abs/1705.03233

[2] Zhang Z, Zohren S. Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration using Intelligent Processing Units. https://arxiv.org/abs/2105.10430

### These notebooks demonstrate how to train DeepLOB-Attention and DeepLOB-Seq2Seq on Graphcore IPUs using TensorFlow2.

## Instructions summary

1. Install and enable the Poplar SDK (see [Poplar SDK setup](##poplar-sdk-setup))

2. Install the system and Python requirements (see [Environment setup](##environment-setup))

3. Run the Jupyter notebook (see [Running the notebook](##running-the-notebook))

## Poplar SDK setup

To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is provided, then follow these steps:

1. Download and unpack the Poplar SDK
2. Navigate inside the unpacked Poplar SDK:
```bash
cd poplar_sdk*
```
2. Enable the Poplar SDK with:
```bash
. enable
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup

To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment with your chosen name:
```bash
python3 -m venv <venv name>
source <venv name>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the TensorFlow 2 and IPU TensorFlow add-ons wheels:
```bash
cd <poplar sdk root dir>
pip3 install tensorflow-2.X.X...<OS_arch>...x86_64.whl
pip3 install ipu_tensorflow_addons-2.X.X...any.whl
```
for the CPU architecture you are running on.

4. Install the Keras wheel:
```bash
pip3 install --force-reinstall --no-deps keras-2.X.X...any.whl
```
For further information on Keras on the IPU, see the [documentation](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/keras/keras.html#keras-with-ipus) and the [tutorial](https://github.com/graphcore/examples/tree/master/tutorials/tutorials/tensorflow2/keras).


5. Navigate to this example's root directory, the directory of this readme.

6. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

More detailed instructions on setting up your TensorFlow 2 environment are available in the [TensorFlow 2 quick start guide](https://docs.graphcore.ai/projects/tensorflow2-quick-start).

## Running the notebook

1. In your virtual environment, install the Jupyter notebook server:
```bash
pip3 install jupyter
```

2. Launch a Jupyter Server on a specific port:
```bash
jupyter-notebook --no-browser --port <port number>
```

3. Connect via SSH to your remote machine, forwarding your chosen port
```bash
ssh -NL <port number>:localhost:<port number> <your username>@<remote machine>
```

4. Use a browser to navigate to your Jupyter server and open the `Multi_Horizon_Financial_Forecasting_on_IPU_using_DeepLOBAttention_Training.ipynb` and `Multi_Horizon_Financial_Forecasting_on_IPU_using_DeepLOBSeq2Seq_Training.ipynb` notebooks.

For more details about this process, or if you need troubleshooting, see our
[guide on using IPUs from Jupyter notebooks](https://github.com/graphcore/examples/tree/master/tutorials/tutorials/standard_tools/using_jupyter).

### License

This application is licensed under the MIT license, see the LICENSE file at the top-level of this repository.
