# Graphcore

---


## Deep Interest Network training on IPUs

This readme describes how to run DIN model training and inference on the IPU.


### Deep Interest Network (DIN)

[DIN](https://arxiv.org/abs/1706.06978) is a deep learning model used for recommendation - specifically predicting "click through rate". It includes an attention layer and a fully connected layer. Graphcore uses the Amazon dataset to train and infer on the IPU, and also can use synthetic data to test.


### Running the models

The following files are provided for running the DIN model.

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `din_model.py`               | DIN model definition                                         |
| `din_train.py`               | Main training loop                                           |
| `din_infer.py`               | Main infer loop                                              |

The common files this model need:

| File                            | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `../common/`                    | Common modules used by DIN and potentially other CTR models  |
| `../common/data_generation.py`  | Prepare and separate data for training and infer             |
| `../common/embedding.py`        | Data embedding                                               |
| `../commonn/log.py`             | Print log                                                    |

### Quick start guide

#### Prepare the environment

###### 1) Download the Poplar SDK

Install the Poplar SDK following the Getting Started guide for your IPU system. Make sure to source the `enable.sh`
script for poplar.

###### 2) Python

Create a virtualenv and install the required packages:

```
virtualenv venv -p python3.6
source venv/bin/activate
pip install <path to the tensorflow-1 wheel from the Poplar SDK>
pip install -r requirements.txt
```

#### Prepare dataset

To use the dataset provided by Amazon, use the prepare_dataset.sh script in the common directory to download and process the data.

```
cd common/
sh prepare_data.sh
cd ..
```

This will download the following files into the common directory:

- cat_voc.pkl
- mid_voc.pkl
- uid_voc.pkl
- local_train_splitByUser
- local_test_splitByUser
- reviews-info
- item-info

As an alternative, you can use synthetic data for training and inference with the option '--use-synthetic-data=True'.

#### Training

Run this file:

```
python din_train.py
```

The default configuration is as follows:
`--max-seq-len` = 100

`--attention-size` = 36

`--use-synthetic-data` = False

`--epochs` = 2

`--device-iterations` = 1600

`--micro-batch-size` = 32

`--learning-rate` = 0.1

`--data-type` = float32

##### Model options
`--max-seq-len` : the maximum sequence length

`--attention-size` : the size of attention

##### Dataset options
`--use-synthetic-data` : whether to use synthetic data (defaults to False)

`--device-iterations` : Number of global batches processed on the device in one step

`--epochs` : set epoch number

##### Training options
`--micro-batch-size` : set batch size for training graph

`--seed` : set random seed

`--learning-rate` : set learning rate

`--model-path` : Place to store and restore model

`--data-type` : Choose the data type (support float32 for now)

#### Inference

Run this file:

```
python din_infer.py
```

The default configuration is as follows:
`--max-seq-len` = 100

`--attention-size` = 36

`--use-synthetic-data` = False

`--epochs` = 1

`--device-iterations` = 1600

`--micro-batch-size` = 128

`--data-type` = float32

##### Model options
`--max-seq-len` : the maximum sequence length

##### Dataset options
`--use-synthetic-data` : whether to use synthetic data (defaults to False)

`--device-iterations` : Number of global batches processed on the device in one step

##### Inference options
`--micro-batch-size` : set batch size for inference graph

`--model-path` : Place to store and restore model

`--data-type` : Choose the data type (support float32 for now)

#### Running unit tests

If you want to run the unit tests, use this command line:

```
python -m pytest test
```
