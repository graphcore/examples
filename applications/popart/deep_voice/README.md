# Graphcore

## Deep Voice 3 

This PopART application implements the Deep Voice 3 model for Text-To-Speech
synthesis as described in this paper:

[Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence
Learning.](https://arxiv.org/abs/1710.07654)

We also use the guided attention loss as described in this paper:

[Efficiently Trainable Text-to-Speech System Based on Deep 
Convolutional Networks with Guided Attention.](https://arxiv.org/abs/1710.08969)

### How to train the deep-voice-3 model

1.  Prepare the environment.

    Install the Poplar SDK following the instructions in the Getting Started guide for your IPU system. Make sure to source
    the `enable.sh` scripts for poplar and popart.

2.  Setup a virtual environment

```
virtualenv venv -p python3.6
source venv/bin/activate
```

3.  Install required packages like torchaudio and librosa

```
pip install -r requirements.txt
```
	
4.  Dataset: Currently, we use the VCTK dataset which is a multi-speaker dataset. For more details see:
https://datashare.is.ed.ac.uk/handle/10283/3443

Download and unzip the dataset to a suitable location. The size of the downloaded file is big (around 11GB).

```
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip -d VCTK-Corpus-0.92
```

Also, to download the Carnegie Mellon Pronouncing Dictionary, just run:

```
python3 -c "import nltk; nltk.download('cmudict')"
```
 
5.  Build custom ops to facilitate gradient-clipping 

```
make all
```
If you want to do training or run tests without gradient-clipping, do not build the custom ops or remove it by doing `make clean`. 

6.  Run the training program. Use the `--data_dir` option to specify the path to
    the data and use the `--model_dir` option to specify a path to save the trained model.
	For a default batch-size of 2:
	
```
python3 deep_voice_train.py --data_dir /path/to/dataset/ --model_dir /path/to/trained/model
```

The dataset path provided as input must be the parent folder of the VCTK-Corpus-0.92 folder to which the data was extracted.
To use multiple IPUs in a data-parallel fashion for higher effective batch-sizes, 
one can use the `--num_ipus` and `--replication-factor` options. For e.g. to use
two IPUs for a batch-size of 4, one can do:

```
python3 deep_voice_train.py --data_dir /path/to/dataset/ --model_dir /path/to/trained/model --num_ipus 2 --replication_factor 2 --batch_size 4
```
Here, the batch-size must be a multiple of the replication factor. One can also specifiy the IPU to use with the `--select_ipu` option. 
For e.g. to use the MultiIPU 18 for data-parallel training, use:

```
python3 deep_voice_train.py --data_dir /path/to/dataset/ --model_dir /path/to/trained/model --num_ipus 2 --replication_factor 2 --batch_size 4 --select_ipu 18
```

### Synthesize audio using the forward pass of the training graph

One can synthesize audio from text by calling the forward pass of the training graph multiple times and simulating the auto-regressive process during inference. 
For truly auto-regressive (and efficient) synthesis, a separate graph has to be constructed and scripts for this will be added to this code-base soon. Meanwhile, one can 
synthesize audio by using the `deep_voice_synthesis_ar_sim.py` script and use the `--sentence` argument to specify the text to synthesize.

```
python3 deep_voice_synthesis_ar_sim.py --trained_model_file /path/to/trained/model.onnx --sentence "It is a beautiful day" --results_path /path/to/audio
``` 
Synthesized audio samples for the specified sentence in different speaker voices will be saved in the `.wav` format in the directory specified by `--results_path`. 
In addition, the output linear-scale spectrograms, mel-scale spectrograms and attention maps for each attention block will be saved to the same location in files with the `.npz` format.


### Benchmarking with randomly generated data

To obtain throughput numbers for training, one can use the `--generated_data` option with the training script.

```
python3 deep_voice_train.py --data_dir TEST --model_dir TEST --generated_data --num_epochs 5
```
or
```
python3 deep_voice_train.py --data_dir TEST --model_dir TEST --generated_data --num_epochs 5 --num_ipus 2 --replication_factor 2 --batch_size 4
```

### Run unit-tests for the deep-voice model

To run unit-tests, simply do:

```
pytest test_popart_deep_voice.py
```


#### Options


Use `--help` to show the available options. Here are a few important options:

`--batch-size` the batch size used to build the training graph.

`--num_epochs` the number of epochs to run for training.

`--num_ipus` the number of IPUs for the session.

`--replication_factor` specifies the number of graph replicas to execute in parallel. The number of samples processed on each IPU will be (batch_size / replication_factor). 
Note that the number of IPUs must be equal to or a multiple of the replication factor.

`--select_ipu` specifies the ID of the IPU or MultiIPU to use for the session.


#### License

This application is licensed under the MIT license - see the LICENSE file at the top-level of this repository.

The code for this application is based on the Carnegie Mellon Pronouncing Dictionary. For license information, see here:
http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/00README_FIRST.txt

The VCTK dataset used here is licensed under the Creative Commons License: Attribution 4.0 International.
See https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/license_text

