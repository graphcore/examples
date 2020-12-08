# BERT Training on IPUs using TensorFlow
This directory provides a script and recipe to run BERT models for NLP pre-training and training on Graphcore IPUs.


## Datasets
The Wikipedia dataset contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the Wikipedia dump file is updated all the time.

If full pre-training is required (with the two phases with different sequence lengths) then data will need to be generated separately for the two phases:
- once with --sequence-length 128 --mask-tokens 20 --duplication-factor 5
- once with --sequence-length 384 --mask-tokens 58 --duplication-factor 5

See the `Datasets/README.md` file for more details on how to generate this data.


## File structure

|    File                   |       Description                                                   |
|---------------------------|---------------------------------------------------------------------|
| `run_pretraining.py`      | Main training loop of pre-training task                             |
| `ipu_utils.py`            | IPU specific utilities                                              |
| `optimization.py`         | IPU optimizer                                                       |
| `log.py`                  | Module containing functions for logging results                     |
| `Datasets/`               | Code for using different datasets.<br/>-`data_loader.py`: Dataloader and preprocessing.<br/>-`create_pretraining_data.py`: Script to generate tfrecord files to be loaded from text data. |
| `Models/modeling_ipu.py`  | A Pipeline Model description for the pre-training task on the IPU.  |
| `lr_schedules/`           | Different LR schedules<br/> - `polynomial_decay.py`: A linearily decreasing warmup with option for warmup.<br/>-`natural_exponential.py`: A natural exponential learning rate schedule with optional warmup.<br/>-`custom.py`: A customized learning rate schedule with given `lr_schedule_by_step`. |


## Quick start guide

### Prepare environment
**1) Download the Poplar SDK**

[Download](https://downloads.graphcore.ai/) and install the Poplar SDK following the Getting Started guide for your IPU system. Source the `enable.sh` script for poplar.

**2) Configure Python virtual environment**

Create a virtual environment and install the appropriate Graphcore TensorFlow 1.15 wheel from inside the SDK directory:
```shell
virtualenv --python python3.6 .bert_venv
source .bert_venv/bin/activate
pip install -r requirements.txt
pip install <path to gc_tensorflow1-15 .whl> 
```

### Generate pre-training data (small sample)
As an example we will create data from a small sample: `Datasets/sample.txt`, however the steps are the same for a large corpus of text. As described above, see `Datasets/README.md` for instructions on how to generate pre-training data for the Wikipedia dataset.

**Download the vocab file**

You can download a vocab from the pre-trained model checkpoints at https://github.com/google-research/bert. For this example we are using `Bert-Base, uncased`.

**Create the data**

Create a directory to keep the data.
```shell
mkdir data
```

Download and unzip the files

```shell
cd data
wget link_to_bert_base_uncased
unzip bert_base_uncased.zip
#check that vocab.txt is actually there!
cd ../
```

`Datasets/create_pretraining_data.py` has a few options that can be viewed by running with `-h/--help`.
Data for the sample text is created by running:
```shell
python3 Datasets/create_pretraining_data.py \
  --input-file Datasets/sample.txt \
  --output-file Datasets/sample.tfrecord \
  --vocab-file data/vocab.txt \
  --do-lower-case \
  --sequence-length 128 \
  --mask-tokens 20 \
  --duplication-factor 5
```
**NOTE**: `--input-file/--output-file` can take multiple arguments if you want to split your dataset between files.
When creating data for your own dataset, make sure the text has been preprocessed as specified at https://github.com/google-research/bert. This means with one sentence per line and documents delimited by empty lines.

### Pre-training with BERT on IPU
Now that the data are ready we can start training our BERT tiny on the IPU!
Run this config:
```shell
python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --train-file ./Datasets/sample.tfrecord
```

BERT tiny is a quick model you can use for simple experiments, the config file is the following:
```
{
  "task": "pretraining",
  "attention_probs_dropout_prob": 0.0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 128,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "hidden_layers_per_stage": 2,
  "type_vocab_size": 2,
  "vocab_size": 30528,
  "seq_length": 128,

  "batch_size": 8,
  "batches_per_step": 1,
  "steps": 100,
  "max_predictions_per_seq": 20,
  "lr_schedule": "polynomial_decay",
  "base_learning_rate": 1e-3,
  "warmup": 10,
  "optimiser": "lamb",

  "parallell_io_threads": 16,
  "pipeline_depth": 16,
  "pipeline_schedule": "Grouped",
  "replicas": 1,
  "precision": "16",
  "seed": 1234,
  "steps_per_ckpts": 100,
  "steps_per_logs": 1,
  "weight_decay": 0.0003,
  "disable_graph_outlining": false,
  "restoring":false,
  "no_logs": false,
  "do_validation":false,
  "do_train":true,
  "available_memory_proportion":0.6,
  "embeddings_placement": "same_as_hidden_layers",

  "checkpoint_path": "./checkpoint/phase1",
  "checkpoint_model":"./checkpoint/phase1/model",
  "log_dir": "./logs/"
}
```

As you can see this config has a first part that specifies the model, BERT tiny, the second part is more about the optimisation where we specify the learning rate, the learning rate scheduler the batch size and the optimiser (in this case we are using LAMB but other options can be used like momentum, ADAM, and SGD).

### View the pre-training results in Weights & Biases
Weights and Biases is a tool that helps you tracking different metrics of your machine learning job, for example the loss the accuracy but also the memory utilisation and the compilation time. For more information please see https://www.wandb.com/.
`requirements.txt` will install a version of wandb.
You can login to wandb as you prefer and then simply activate it using the flag --wandb
```shell
python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --train-file ./Dataset/sample.tfrecord --wandb
```

Just before that the run starts you will see a link to your run appearing in your terminal.

#### Pre-training of BERT LARGE on Wikipedia

The steps to follow to run the BERT LARGE model are exactly the same as before.
As first you need to create the Wikipedia pret-raining data using the script in the Dataset folder, see the readme there for the details.
After this you can run BERT LARGE on 16 IPUs at batch size 65k using LAMB with:

```shell
python3 run_pretraining.py --config configs/pretrain_large_128_phase1.json --wandb
```
Remember to adapt the config file inserting the path top your 128 dataset or add the flag `--train-file` as we did before.
At the end of the training, the script will save a checkpoint of the final model, that will be the starting point of Phase 2 training.

Phase 2 is going to be run using batch size 16k, here is the command you can use:
```shell
python3 run_pretraining.py --config configs/pretrain_large_384_phase2.json --start-from-ckpt /path/to/final/ckpt/of/phase1 --wandb
```

Remember to insert in the config files the path to the 384 dataset and be sure that the number of masked tokens in the json matches the ones you used in the creation of the dataset.
Also, it is important to start the training from the very end of phase1, in order to do so you can use the `--start-from-ckpt` and point to the final checkpoint of phase1, the model will be initialised correctly from that point.

## Information about the application

The config files provided demonstrate just a sample of what you can do with this application.
Aspects that can be changed include the optimiser, learning rate schedule and model size/shape.
Use the `-h/--help` option to see the different options that can be used.
The command line options will override the settings contained within a config file.
For example,
```shell
python3 run_pretraining --config configs/pretrain_tiny_128_lamb.json --sequence-length 384
```

will run a job with seq len 384, overriding the 128 present in the config.

## Profiling your applications

The PopVision Graph Analyser and System Analyser are the two main tools to inspect the behaviour of your application on the IPU, they can be downloaded from [Download](https://downloads.graphcore.ai/).
Here, we are going focus on the Graph Analyser and how to use it to profile the execution and memory utilisation of BERT in the IPU. Clearly the same procedure can be applied to any other application.

### Memory Profile

The first thing we are going to see is how we can inspect the hardware utilisation. This will give us insights on how much space left is there on the device, this is important information that we can use to increase the batch size or use a more complex optimiser.

Profiling data can be generated by the compiler by setting the following options:

```shell
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"false", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.format":"v3", "autoReport.directory":"./memory_report"}'
```

the field `"autoReport.directory":"./memory_report"` in this example is pointing to the directory where the memory profile will be found.

When profiling an application it can be useful to inspect the log output from the different layers of the software stack in combination with the information displayed in the PopVision Graph Analyser.
Due to the verbosity of the log level, it is good practice to redirect the output into a file, here is an example:

```shell
POPLAR_LOG_LEVEL=INFO POPLIBS_LOG_LEVEL=INFO TF_CPP_VMODULE='poplar_compiler=1' POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"false", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.format":"v3", "autoReport.directory":"./memory_report"}' python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --compile-only --generated-data > output.log 2>&1 &
```

In the previous command we set the log level for POPLAR, POPLIBS to INFO and we also set TensorFlow log to be 'poplar_compiler=1'.
In this example we are taking advantage of two specific flags of `run_pretraining.py` that can improve the workflow: `--generated-data` and `--compile-only`. 
The first uses random data generated on the host instead of real data.
The second is more interesting and enables the compilation of the application, getting the memory profile, without attaching to any IPUs.
This makes possible for example to run a lot of experiments with different hyperparameters in order to understand which one is going to use the hardware better.
The IPUs can then be used on jobs that require physical hardware such as convergence experiments or to obtain execution profiles, which is the theme of the next section.

### Execution Profiles

The method presented in the previous section will allow you inspect the memory profile, here we show how to use the PopVision Graph Analyser to inspect the execution of your application.As before, this information can be obtained with specific `POPLAR_ENGINE_OPTIONS`:

```shell
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.format":"v3", "autoReport.directory":"./execution_report"}'
```

Where again the folder where the profile output is going to be generated can be chosen to be whatever you prefer. The two `POPLAR_ENGINE_OPTIONS` are the same, the only different is `"autoReport.outputExecutionProfile":"true"`. In this case, it is clearly necessary to attach to the IPU since the code has to be run, so no `--compile-only` flag can be used.

In order to have a easily readable execution trace, it is good practice to modify the execution of your application. We would like for example to execute just a single step, and we would like that batches-per-step is set to 1. For models being executed in a pipeline configuration, we are going also to set the pipeline to the minimal value possible.
The previous suggestions are due to the fact that every step is the same so it is much easier to inspect just one of them, and the second is that an extremely deep pipeline will be difficult to navigate and inspect, and it is anyway the repetition of stages that are the same. We can then set the pipeline to a minimal value.
Let's make an example with a large application, like the phase1 training with LAMB:

```shell
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.format":"v3", "autoReport.directory":"./execution_report"}' python run_pretraining.py --config configs/pretrain_large_128_phase1.json --steps 1 --batches-per-step 1 --pipeline-depth 10
```
In this way the execution profile will be dropped together with the memory profile and we can inspect how many cycles the hardware is spending on each operation and this can give us insight on possible optimisations and improvements.
