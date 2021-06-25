# Graphcore

## Conformer for Speech Recognition 

This PopART application implements a Speech Recognition model using Conformer blocks as described in this paper:

[Conformer: Convolution-augmented Transformer for Speech Recognition.](https://arxiv.org/abs/2005.08100)

Currently, training is based on [Connectionist Temporal Classification (CTC)](https://www.cs.toronto.edu/~graves/icml_2006.pdf). 
For the implementation of the CTC loss, go to `custom_operators/ctc_loss`.


### How to train a conformer model

1.  Prepare the environment.

    Install the `poplar-sdk` following the README provided. Make sure to source
    the `enable.sh` scripts for poplar and popart.

2.  Setup a virtual environment

```
virtualenv venv -p python3.6
source venv/bin/activate
```

3.  Install required packages like torchaudio and librosa.

```
pip install -r requirements.txt
```

4.  Dataset: Currently, we use the LibriSpeech dataset which is a multi-speaker dataset of approximately 1000 hours of 16kHz English speech. For more details see http://www.openslr.org/12

To download the default versions of the training and test sets, go to a suitable location and do:

```
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -zxvf train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -zxvf test-clean.tar.gz
```

More `train`, `dev` and `test` datasets can also be downloaded as listed here: http://www.openslr.org/12 .


5.  Build CTC custom op for training. Go to `custom_operators/ctc_loss`, and run

```
make all
```

6.  Run the training program. Use the `--model-conf-file` option to specify which model configuration to use.
Use the `--data-dir` option to specify the path to the data and use the `--model-dir` option to specify a path to save the trained model.
For e.g., to train the `small` model configuration, do:
	
```
python3 conformer_train.py --model-conf-file  model_configs/small_model_conf_bs4.json --model-dir /path/to/trained/model --data-dir /path/to/librispeech
```

### How to run inference with the conformer model

1. To run inference, one needs to install the [ctcdecode](https://github.com/parlance/ctcdecode) library for CTC beam search decoding.

```
./install_ctcdecode.sh
```

2. Run the inference program. Use the `--model-file` option to specify which trained model checkpoint to use. And use the `--results-dir` option to specify where to save the inference results. For e.g.. to test a trained model of `small` configuration, do:

```
python3 conformer_inference.py --model-conf-file  model_configs/small_model_conf_bs4.json --model-file /path/to/trained/model.onnx --data-dir /path/to/librispeech --results-dir /path/to/inference_results/
```

The ground-truth and model predictions will be saved in a `.txt` file at `/path/to/inference_results/`.


### Run unit-tests

To run unit-tests, simply do:

```
pytest
```


#### Options


Use `--help` to show the available options. Here are a few other options:

`--dataset` the dataset to use. For training, this can be one of `train-clean-100`, `train-clean-360`,`train-other-500`. For testing, this can be one of `test-clean`, `test-other`.

`--num-epochs` the number of epochs to run for training.

`--select-ipu` specifies the ID of the IPU or MultiIPU to use for the session.


#### License

This application is licensed under the MIT license - see the LICENSE file at the top-level of this repository.

The LibriSpeech  dataset used here is licensed under the Creative Commons Attribution 4.0 International License.
See http://www.openslr.org/12

The code for this application uses the [ctcdecode](https://github.com/parlance/ctcdecode) library which is licensed under the MIT license. 
See https://github.com/parlance/ctcdecode/blob/master/LICENSE

