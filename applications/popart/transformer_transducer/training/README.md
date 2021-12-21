# Graphcore

### Transformer Transducer model for Speech Recognition

This PopART application is partly motivated by the Speech Recognition model described in [Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss](https://arxiv.org/abs/2002.02562). Note that the model implemented here is not an exact match with the model from the [Transformer Transducer paper](https://arxiv.org/abs/2002.02562). The model trained with the default config provided here has approximately 14M parameters. With this default config, after 100 epochs of training, one would get a Word Error Rate (WER) of ~7% on the `dev-clean` subset of LibriSpeech. The original idea of sequence transduction and training using the RNN-Transducer loss was introduced in [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711). The code in the folders `common/`, `configs/`, `scripts/`, `rnnt_reference/` and `utils/` are derived from the MLCommons training benchmark for [RNNT Speech Recognition](https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch). Below, we describe how to run the training program for the Transformer-Transducer model.

### Prepare the environment

1. Check if the packages `libsndfile` and `sox` are installed by doing `dpkg -l libsndfile1 sox`. If they are not installed, try installing them by doing: 

```
sudo apt-get install -y libsndfile1 sox
```

2. Source the appropriate `enable.sh` scripts for poplar/popart from the appropriate SDK. 

3. Setup a virtual environment.
 
``` 
virtualenv rnnt_venv -p python3.6
source rnnt_venv/bin/activate
```    

It is highly recommended to upgrade pip to version 19.0 or later (`python3 -m pip install -U pip`).

4. Install the `horovod` software package provided with the poplar SDK:

```
pip install horovod-XXXX-XXXX.whl
```
	
5. Build all the custom operators required for the application.

	Go to the root folder of the application (`transformer_transducer`) and run:

```
make all
```

6. Install the required python packages for the training application:

	Go to the folder `transformer_transducer/training`

	```
	pip install -r requirements.txt
	```
	
	The dataset preparation and the training application scripts should be run from the `transformer_transducer/training` folder.

### Download and preprocess the LibriSpeech dataset

We use the LibriSpeech dataset which is a multi-speaker dataset of approximately 1000 hours of 16kHz English speech. For more details see http://www.openslr.org/12.

Be sure to provide a location to the data-processing scripts where you have write access and ensure there is enough disk space. After preprocessing, the LibriSpeech dataset requires about 120GB. For example, if the system has a disk mounted at `/localdata`, it would be advisable to provide a path like `/localdata/datasets` to download and preprocess the dataset. In the following, we will assume that the location of the dataset is `/localdata/datasets`. 

1. First download the dataset:

	```
	bash scripts/download_librispeech.sh /localdata/datasets
	```

2. Then preprocess the dataset:

	```
	bash scripts/preprocess_librispeech.sh /localdata/datasets
	```

3. Finally, create the sentence pieces to be used as tokens for the training program:

	```
	bash scripts/create_sentencepieces.sh /localdata/datasets
	```

### Launch the training program

Once the dataset is downloaded and prepared, we are ready to launch the training program.

Standard command line options to be provided to the training application for running on a IPU-POD16 are shown below with the two examples.
 
1. Single-instance training (without poprun) on 16 IPUs with a per-device batch-size of 2.
```
python3 transducer_train.py --model-conf-file configs/transducer-1023sp.yaml --model-dir /localdata/transducer_model_checkpoints --data-dir /localdata/datasets/LibriSpeech/ --enable-half-partials --enable-lstm-half-partials --enable-stochastic-rounding
```

2. Four-instance training with poprun on 16 IPUs with a per-device batch-size of 2. Make sure you have the `partition-name` and `vipu-server-host-ip` before doing multi-instance training.
```
poprun --vipu-partition {partition-name} --vipu-server-host {vipu-server-host-ip} --vipu-server-timeout 600 --num-instances=4 --num-replicas=16 --mpi-global-args='--output-filename poprun_output' python3 transducer_train.py --model-conf-file configs/transducer-1023sp.yaml --model-dir /localdata/transducer_model_checkpoints --data-dir /localdata/datasets/LibriSpeech/ --enable-half-partials --enable-lstm-half-partials --enable-stochastic-rounding
```

The model checkpoints will be saved at `/localdata/transducer_model_checkpoints`. Checkpoints after each epoch of training will be created in sub-folders here with names `checkpoint_{epoch_count}`. One can specify a different location to save checkpoints by providing a different location to the command line argument `--model-dir`. 


## Instructions to run the validation program on a transducer model

Once you have a trained model, you can run the validation program by the two sample command lines provided below.

1. Single-instance validation (without poprun) on 16 IPUs with a per-device batch-size of 2.

```
python3 transducer_validation.py --model-conf-file configs/transducer-1023sp.yaml --model-dir /localdata/transducer_model_checkpoints/checkpoint_100 --data-dir /localdata/datasets/LibriSpeech/ --enable-half-partials --enable-lstm-half-partials 
```

2. Sixteen-instance validation with poprun on 16 IPUs with a per-device batch-size of 2. Make sure you have the `partition-name` and `vipu-server-host-ip` before doing multi-instance validation.

```
poprun --vipu-partition {partition-name} --vipu-server-host {vipu-server-host-ip}  --vipu-server-timeout 600 --num-instances=16 --num-replicas=16 --mpi-global-args="--output-filename poprun_output" python3 transducer_validation.py --model-conf-file configs/transducer-1023sp.yaml --model-dir /localdata/transducer_model_checkpoints/checkpoint_100 --data-dir /localdata/datasets/LibriSpeech/ --enable-half-partials --enable-lstm-half-partials 
```

The commands above evaluate the model checkpointed after 100 epochs of training. One can specify a different checkpoint to evaluate by providing a different checkpoint folder to the command line argument `--model-dir`. 

## Run unit-tests

To run unit-tests related to the graph build and training program, do:
```
pytest -v test_transducer.py
```


To run unit-tests related to the audio feature augmentation, do:

```
pytest -v test_data_processor_cpp.py
```

### Options

Use `--help` to show the available options. Here are a few relevant options:

`--replication-factor` - specifies the number of graph replicas to execute for data-parallel training.

`--batch-size` - this is the batch size processed at once by all the devices in the system. The number of samples processed per device will be `batch-size / replication-factor`.

`--gradient-accumulation-factor` - the number of batch iterations over which gradients are accumulated. The global batch size is given as `gradient-accumulation-factor X batch-size`.

`--enable-ema-weights` - whether to enable exponential moving averages of model weights during training.

`--gradient-clipping-norm` - sets the gradient clipping norm for the Lamb optimizer.

`--num-buckets` - this determines the number of buckets for grouping samples by audio duration.

`--num-epochs` the number of epochs to run for training.

`--generated-data` indicates to use random generated data for training benchmarking purposes (does not work with validation).

`--do-validation` indicates to execute validation after every epoch of the training program. 


### License

All the files in this folder are distributed under the MIT license (see the LICENSE file at the top-level of this repository) except for the files in `common/`, `configs/`, `scripts/`, `rnnt_reference/` and `utils/` which are derived from [MLCommons](https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch) and are distributed under the Apache License, Version 2.0.

The LibriSpeech dataset used here for this application is licensed under the Creative Commons Attribution 4.0 International License.
See http://www.openslr.org/12



