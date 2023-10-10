# Fraud detection on IPU using node classification - Training with PyTorch Geometric

This application demonstrates using PyTorch Geometric on Graphcore IPUs to train a model for fraud detection using the [IEEE-CIS dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data). The approach is inspired by the [AWS Fraud Detection with GNNs](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) project, framing the problem as a node classification task using a heterogeneous graph, where the transaction node types have a label indicating whether they are fraudulent or not.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | GNNs | RGCN | [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
 | Node Classification | <div style="text-align: center;">✅ <br>recommended: 16 (min: 4) | <p style="text-align: center;">❌ | [AWS Fraud Detection with GNNs](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) |

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

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch*x86_64.whl
```

4. Install the PopTorch Geometric wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch_geometric*.whl
```

5. Navigate to this example's root directory, the directory of this readme.

6. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

More detailed instructions on setting up your PyTorch environment are available in the [PyTorch quick start guide](https://docs.graphcore.ai/projects/pytorch-quick-start).

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

4. Preprocess the dataset by running the following command. If you want to understand more about the dataset preprocessing see `1_dataset_preprocessing.ipynb` .
```bash
python3 dataset.py
```

5. Use a browser to navigate to your Jupyter server and open the `2_training.ipynb` notebook.

For more details about this process, or if you need troubleshooting, see our
[guide on using IPUs from Jupyter notebooks](https://github.com/graphcore/examples/tree/master/tutorials/tutorials/standard_tools/using_jupyter).

### License

This application is licensed under the MIT license, see the LICENSE file at the top-level of this repository.

The approach used in this application is inspired from the [AWS Fraud Detection with GNNs](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl) project, licensed with the Apache License 2.0.
