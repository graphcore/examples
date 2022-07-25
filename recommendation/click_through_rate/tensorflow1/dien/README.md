# Graphcore

---


## Deep Interest Evolution Network training on IPUs

This readme describes how to run DIEN model training and inference on the IPU.


### Deep Interest Evolution Network (DIEN)

[DIEN](https://arxiv.org/abs/1809.03672) is a deep learning model used for recommendation - specifically predicting "click through rate". It includes a gru layer, an attention layer, an augru layer and a fully connected layer. Graphcore uses the Amazon dataset to train and infer on the IPU, and also can use synthetic data to test.


### Running the models

The following files are provided for running the DIEN model.

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `dien_model.py`              | DIEN model definition                                        |
| `dien_train.py`              | Main training loop                                           |
| `dien_infer.py`              | Main infer loop                                              |
| `rnn.py`                     | Dynamic RNN Implementation                                   |

The common files this model uses:

| File                            | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `../common/`                    | Common modules used by DIEN and potentially other CTR models |
| `../common/data_generation.py`  | Prepare and separate data for training and infer             |
| `../common/embedding.py`        | Data embedding                                               |
| `../commonn/log.py`             | Print log                                                    |

### Quick start guide

#### Prepare the environment

###### 1) Download the Poplar SDK

Install the Poplar SDK following the Getting Started guide for your IPU system. Poplar SDK 2.0 or later is needed.
Make sure to source the `enable.sh` script for poplar.

###### 2) Python

Create a virtualenv and install the required packages:

```
virtualenv venv -p python3.6
source venv/bin/activate
pip install <path to the tensorflow-1 wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 1 from the Poplar SDK>
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
python dien/dien_train.py
```

The default configuration is as follows:
`--max-seq-len` = 100

`--hidden-size` = 36

`--attention-size` = 36

`--precision` = "32.32"

`--gru-type` = "PopnnGRU"

`--augru-type` = "PopnnAUGRU"

`--epochs` = 2

`--device-iterations` = 1600

`--micro-batch-size` = 32

`--learning-rate` = 0.6

`--optimizer` = SGD

`--use-ipu-emb` = False

##### Model options
`--max-seq-len` : the maximum sequence length

`--hidden-size` : the size of hidden

`--attention-size` : the size of attention

`--gru-type` : where we use a TensorFlow GRU or a Popnn GRU

`--augru-type` : where we use a TensorFlow AUGRU or a Popnn AUGRU

`--precision` : Chosse the precision type (support 32.32 for now)

##### Dataset options
`--use-synthetic-data` : whether to use synthetic data (defaults to False)

`--device-iterations` : Number of global batches processed on the device in one step

`--epochs` : set epoch number

##### Training options
`--micro-batch-size` : set batch size for training graph

`--seed` : set random seed

`--learning-rate` : set learning rate

`--model-path` : Place to store and restore model

`--optimizer` : set optimizer

`--use-ipu-emb` : use ipu embedding or not

#### Inference

Run this file:

```
python dien/dien_infer.py
```

The default configuration is as follows:
`--max-seq-len` = 100

`--hidden-size` = 36

`--attention-size` = 36

`--precision` = "32.32"

`--gru-type` = "PopnnGRU"

`--augru-type` = "PopnnAUGRU"

`--epochs` = 1

`--device-iterations` = 1600

`--micro-batch-size` = 128

`--use-ipu-emb` = False

##### Model options
`--max-seq-len` : the maximum sequence length

`--hidden-size` : the size of hidden

`--attention-size` : the size of attention

`--gru-type` : where we use a TensorFlow GRU or a Popnn GRU

`--augru-type` : where we use a TensorFlow AUGRU or a Popnn AUGRU

`--precision` : Chosse the precision type (support 32.32 for now)

##### Dataset options
`--use-synthetic-data` : whether to use synthetic data (defaults to False)

`--device-iterations` : Number of global batches processed on the device in one step

`--epochs` : set epoch number

##### Inference options
`--micro-batch-size` : set batch size for inference graph

`--model-path` : Place to store and restore model

`--micro-batch-size` : set batch size for training graph

`--seed` : set random seed

`--use-ipu-emb` : use ipu embedding or not

#### Running unit tests

If you want to run the unit tests, use this command line:

```
python -m pytest test
```
