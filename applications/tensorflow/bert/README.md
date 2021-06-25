# BERT Training on IPUs using TensorFlow
This directory provides the scripts and recipe to run BERT models for NLP pre-training and SQuAD on Graphcore IPUs.
This README is structured to show the datasets required to train the model, how to quickly start training BERT on Graphcore IPUs, details about how to profile the application, and how to use `popdist` to train BERT at scale on Graphcore IPU-PODs.

# Table of contents
1. [Benchmarking](#benchmarking)
2. [Datasets](#datasets)
3. [File structure](#file_structure)
4. [Quick start guide](#quick_start)
    1. [Prepare environment](#prep_env)
    2. [Generate pre-training data (small sample)](#gen_data)
    3. [Input and output files ](#input_output)
    4. [Remasked datasets ](#remasked_data)
    5. [Pre-training with BERT on IPU](#pretrain_IPU)
    6. [View the pre-training results in Weights & Biases](#wandb)
3. [Pre-training of BERT Large on Wikipedia](#LARGE_WIKI)
    - [Launch BERT Pre-Training Script](#pretrain_script)
3. [Fine-tuining of BERT Large on SQuAD](#LARGE_SQUAD)
    - [Launch BERT Fine-Tuning Script](#finetune_script)
4. [Information about the application](#app_info)
5. [Profiling your applications](#profiling)
    1. [Memory Profile](#memory_profile)
    1. [Execution Profiles](#execution_profile)
3. [Multi-Host training using PopDist](#popdist)
    1. [Utility Script](#popdist_script)
    3. [POD-128 Pre-training](#pod_128)


## Benchmarking <a name="benchmarking"></a>

To reproduce the published Mk2 throughput benchmarks, please follow the setup instructions in this README, and then follow the instructions in [README_Benchmarks.md](README_Benchmarks.md) 

## Datasets <a name="datasets"></a>
The Wikipedia dataset contains approximately 2.5 billion wordpiece tokens. This is only an approximate size since the Wikipedia dump file is updated all the time.

If full pre-training is required (with the two phases with different sequence lengths) then data will need to be generated separately for the two phases:
- once with --sequence-length 128 --mask-tokens 20 --duplication-factor 5
- once with --sequence-length 384 --mask-tokens 56 --duplication-factor 5

See the `bert_data/README.md` file for more details on how to generate this data.


## File structure <a name="file_structure"></a>

|    File                   |       Description                                                   |
|---------------------------|---------------------------------------------------------------------|
| `run_pretraining.py`      | Main training loop of pre-training task                             |
| `ipu_utils.py`            | IPU specific utilities                                              |
| `ipu_optimizer.py`        | IPU optimizer                                                       |
| `log.py`                  | Module containing functions for logging results                     |
| `bert_data/`               | Code for using different datasets.<br/>-`data_loader.py`: Dataloader and preprocessing.<br/>-`create_pretraining_data.py`: Script to generate tfrecord files to be loaded from text data.<br/>-`pretraining.py`: Utility for loading the pre-training data.<br/>-`squad.py`: Utility for loading the SQuAD fine-tuning data. <br/>-`squad_results.py`: Utility for processing SQuAD results. <br/>-`tokenization.py`: Utility for processing tokenization of the data. <br/>-`wiki_processing.py`: Process the Wikipedia data. |
| `modeling.py`  | A Pipeline Model description for the pre-training task on the IPU.  |
| `lr_schedules/`           | Different LR schedules<br/> - `polynomial_decay.py`: A linearily decreasing warmup with option for warmup.<br/>-`natural_exponential.py`: A natural exponential learning rate schedule with optional warmup.<br/>-`custom.py`: A customized learning rate schedule with given `lr_schedule_by_step`. |
| `run_squad.py` | Main training and inference loop for SQuAD fine-tuning. |
| `loss_scaling_schedule.py`| Sets a loss scaling schedule based on config arguments. |
| ` scripts/` | Directory containing a number of utility scripts:<br/>-`create_wikipedia_dataset.sh`: Generate the wikipedia tf_record datafiles for pre-training<br/>-`fine_tune_squad_large.sh`: Fine tune BERT Large from the latest Phase 2 checkpoint on SQuAD.<br/>-`poprun_pretrianing.sh`: Run pre-training of BERT Large on the Graphcore IPU-POD64.<br/>-`pretrain.sh`: ***The main pre-training script.*** Pre-train BERT (large or base) on Graphcore IPUs. This script will run Phase 1, and use the results to train Phase 2 on the Wikipedia dataset.

## Quick start guide <a name="quick_start"></a>

### Prepare environment <a name="prep_env"></a>
**1) Download the Poplar SDK**

[Download](https://downloads.graphcore.ai/) and install the Poplar SDK following the Getting Started guide for your IPU system. Source the `enable.sh` script for poplar.

**2) Configure Python virtual environment**

Create a virtual environment and install the appropriate Graphcore TensorFlow 1.15 wheel from inside the SDK directory:
```shell
virtualenv --python python3.6 .bert_venv
source .bert_venv/bin/activate
pip install -r requirements.txt
pip install <path to the tensorflow-1 wheel from the Poplar SDK>
```

### Generate pre-training data (small sample) <a name="gen_data"></a>
As an example we will create data from a small sample: `bert_data/sample.txt`, however the steps are the same for a large corpus of text. As described above, see `bert_data/README.md` for instructions on how to generate pre-training data for the Wikipedia dataset.

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
wget <path_to_bert_uncased>
unzip bert_base_uncased.zip
#check that vocab.txt is actually there!
cd ../
```

`bert_data/create_pretraining_data.py` has a few options that can be viewed by running with `-h/--help`.
Data for the sample text is created by running:
```shell
python3 bert_data/create_pretraining_data.py \
  --input-file Datasets/sample.txt \
  --output-file Datasets/sample.tfrecord \
  --vocab-file data/vocab.txt \
  --do-lower-case \
  --sequence-length 128 \
  --mask-tokens 20 \
  --duplication-factor 5
```

#### Input and output files <a name="input_output"></a>
`--input-file/--output-file` can take multiple arguments if you want to split your dataset between files.
When creating data for your own dataset, make sure the text has been preprocessed as specified at https://github.com/google-research/bert. This means with one sentence per line and documents delimited by empty lines.


#### Remasked datasets <a name="remasked_data"></a>
The option `--remask` can be used to move the masked elements at the beginning of the sequence. This will improve the inference and training performance.


### Pre-training with BERT on IPU <a name="pretrain_IPU"></a>
Now that the data are ready we can start training our BERT tiny on the IPU!
Run this config:
```shell
python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --train-file ./bert_data/sample.tfrecord
```

The `configs/pretrain_tiny_128_lamb.json` file is a small model that can be used for simple experiments.

This config file has a first part that specifies the model, BERT tiny, the second part is more about the optimisation where we specify the learning rate, the learning rate scheduler the batch size and the optimiser (in this case we are using LAMB but other options can be used like momentum, ADAM, and SGD).

### View the pre-training results in Weights & Biases <a name="wandb"></a>
This project supports Weights & Biases, a platform to keep track of machine learning experiments. A client for Weights&Biases will be installed by default and can be used during training by passing the `--wandb` flag. 
The user will need to manually log in (see the quickstart guide [here](https://docs.wandb.ai/quickstart)) and configure the project name with `--wandb-name`.)
For more information please see https://www.wandb.com/.

Once logged into wandb logging can be activated by running:
```shell
python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --train-file ./Dataset/sample.tfrecord --wandb
```
You can also name you wandb run with the flag `--wandb-name <YOUR RUN NAME HERE>`.


## Pre-training of BERT Large on Wikipedia <a name="LARGE_WIKI"></a>

The steps to follow to run the BERT Large model are exactly the same as before.
As first you need to create the Wikipedia pre-training data using the script in the `bert_data` directory, see the README there for the details.
After this you can run BERT Large on 16 IPUs at batch size 65k using LAMB with:

```shell
python3 run_pretraining.py --config configs/pretrain_large_128_phase1.json 
```
Remember to adapt the config file inserting the path top your 128 dataset or add the flag `--train-file` as we did before.
At the end of the training, the script will save a checkpoint of the final model, that will be the starting point of Phase 2 training.

Phase 2 is going to be run using batch size 16k, here is the command you can use:
```shell
python3 run_pretraining.py --config configs/pretrain_large_384_phase2.json --init-checkpoint /path/to/final/ckpt/of/phase1
```

Remember to insert in the config files the path to the 384 dataset and be sure that the number of masked tokens in the json matches the ones you used in the creation of the dataset.
In phase 2 pretraining the final phase 1 checkpoint is expected to be passed in using `--init-checkpoint` option.

Note that the configuration flag `--static-mask` must be set if the datasets was [generated with the remasking option](#remasked-dataset). A dataset that does not require the `--static-mask` flag may end up using more memory than a dataset that does, like the ones presented in the config files folder. Reducing the `--available-memory-proportion` parameter may be required in this case.

### Launch BERT Pre-Training Script <a name="pretrain_script"></a>
For simplicity in the `scripts/` directory there is a script that manages BERT pre-training from a single command. This script can be used to pre-train BERT on a Graphcore IPU system with 16 IPUs.

To run with the default configuration for `BERT Large` as given in `configs/pretrain_large_128_phase1.json` and `configs/pretrain_large_128_phase2.json` simply run:
```shell
./scripts/pretrain.sh large
```
This will launch a BERT-Large pre-training over 16 IPUs, consisting of 4x replication of a 4 IPU pipelined model. 
You will need to check the paths (`PHASE1_CONFIG_PATH`, 
`PHASE1_TRAIN_FILE`, 
`PHASE2_TRAIN_FILE`, 
`PHASE1_CHECKPOINT`) given in the `pretrain.sh` script match your local paths where you have saved the data / the config you wish to run.

To run pre-training for BERT-Base simply run:
```shell
./scripts/pretrain.sh base
```

## Fine-tuning of BERT-Large on SQuAD <a name="LARGE_SQUAD"></a>
Provided are the scripts to fine tune BERT on the Stanford Question Answering Dataset (SQuAD), a popular question answering benchmark dataset.
To run on SQuAD, you will first need to download the dataset, these are no longer available on the SQuAD website, but the necessary training and evaluation files can be found from the following links:
1.  [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
1.  [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
2.  [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Place these files in a directory `squad/` parallel to the Wikipedia data, the expected path can be found in the `configs/squad_large.json` file. 

To run BERT fine-tuning on SQuAD requires the same set up as for pre-training, follow the steps in [Prepare Environment](#prep_env) to activate the SDK and install the required packages. 

Finally, the fine-tuning for a Large model can be run with the following command:
``` shell
python3 run_squad.py --config configs/squad_large.json --do-training --init-checkpoint /path/to/phase2/large/checkpoint.ckpt
```
where the checkpoint given must be a converged Phase 2 Large model. This will output a fine-tuned checkpoint that can be used for prediction in the following step.
(The same command can be run with the `configs/squad_base.json` configuration if you provide a Phase 2 BERT Base model as an initial checkpoint.)

The prediction can then be run with the following command:
``` shell
python3 run_squad.py --config configs/squad_large.json --do-predict --init-checkpoint /path/to/squad/large/checkpoint.ckpt
```
This will output a set of predictions to the location specified in the SQUAD config file. These predictions can be evaluated as:
```shell
python3 /path/to/evaluate-v1.1.py /path/to/dev-v1.1.json /path/to/predictions.json
```
This will output the Exact Match (EM) and F1 scores of the final fine-tuned BERT model. 

***For simplicity the `run_squad.py` script can be run straight through.***
This means by running the following:
```shell
./run_squad.py --config configs/squad_large.json --do-training --do-predict --do-evaluation --init-checkpoint /path/to/phase2/model.ckpt
```
fine-tuning, prediction, and evaluation can be run with a single command. 
As with pre-training, `run_squad.py` takes an option to log results to Weights and Biases, this functionality can be turned on by adding the command line options ` --wandb --wandb-name <DESIRED NAME>`.


### Launch BERT Fine-Tuning Script <a name="finetune_script"></a>
For simplicity in the `scripts/` directory there is a script that manages BERT fine-tuning from a single command. This script can be used to fine-tune BERT on a Graphcore IPU system with 4 IPUs.

To run with the defulat configuration for `BERT Large` as given in `configs/squad_large.json` simply run:
```shell
./scripts/fine_tune_squad.sh large
```
This will launch a BERT Large fine-tuning over 4 IPUs, on completion the predictions will be made and the official evaluation performed on the SQuAD results. 
The final EM and F1 scores will be displayed.

To run the same fine-tunning, prediction, and evaluation for `BERT Base` simply run:
```shell
./scripts/fine_tune_squad.sh base
```

### Launch BERT End-to-End Script <a name="end_to_end"></a>
Finally a script is provided to train BERT (base or large) end-to-end. 
This script performs pre-training Phase 1, pre-training Phase2, 
fine-tuning on SQuAD, predictions on SQuAD, and evaluates the results
to obtain EM and F1 scores. 
This script is run is the same manner as the previous scripts, ensure the data and environment is set up correctly, then run:
```shell
./scripts/BERT_end_to_end_IPU.sh large
```




## Information about the application  <a name="app_info"></a>

The config files provided demonstrate just a sample of what you can do with this application.
Aspects that can be changed include the optimiser, learning rate schedule and model size/shape.
Use the `-h/--help` option to see the different options that can be used.
The command line options will override the settings contained within a config file.
For example,
```shell
python3 run_pretraining --config configs/pretrain_tiny_128_lamb.json --sequence-length 384
```

will run a job with seq len 384, overriding the 128 present in the config.

## Profiling your applications <a name="profiling"></a>

The PopVision Graph Analyser and System Analyser are the two main tools to inspect the behaviour of your application on the IPU, they can be downloaded from [Download](https://downloads.graphcore.ai/).
Here, we are going focus on the Graph Analyser and how to use it to profile the execution and memory utilisation of BERT in the IPU. Clearly the same procedure can be applied to any other application.

### Memory Profile <a name="memory_profile"></a>

The first thing we are going to see is how we can inspect the hardware utilisation. This will give us insights on how much space left is there on the device, this is important information that we can use to increase the batch size or use a more complex optimiser.

Profiling data can be generated by the compiler by setting the following options:

```shell
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"false", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "autoReport.directory":"./memory_report"}'
```

the field `"autoReport.directory":"./memory_report"` in this example is pointing to the directory where the memory profile will be found.

When profiling an application it can be useful to inspect the log output from the different layers of the software stack in combination with the information displayed in the PopVision Graph Analyser.
Due to the verbosity of the log level, it is good practice to redirect the output into a file, here is an example:

```shell
POPLAR_LOG_LEVEL=INFO POPLIBS_LOG_LEVEL=INFO TF_CPP_VMODULE='poplar_compiler=1' POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"false", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "autoReport.directory":"./memory_report"}' python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --compile-only --generated-data > output.log 2>&1 &
```

In the previous command we set the log level for POPLAR, POPLIBS to INFO and we also set TensorFlow log to be 'poplar_compiler=1'.
In this example we are taking advantage of two specific flags of `run_pretraining.py` that can improve the workflow: `--generated-data` and `--compile-only`.
The first uses random data generated on the host instead of real data.
The second is more interesting and enables the compilation of the application, getting the memory profile, without attaching to any IPUs.
This makes possible for example to run a lot of experiments with different hyperparameters in order to understand which one is going to use the hardware better.
The IPUs can then be used on jobs that require physical hardware such as convergence experiments or to obtain execution profiles, which is the theme of the next section.

### Execution Profiles <a name="execution_profile"></a>

The method presented in the previous section will allow you inspect the memory profile, here we show how to use the PopVision Graph Analyser to inspect the execution of your application. As before, this information can be obtained with specific `POPLAR_ENGINE_OPTIONS`:

```shell
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "autoReport.directory":"./execution_report"}'
```
This option is very similar to the previous example to obtain a memory profile, the only difference is `"autoReport.outputExecutionProfile":"true"`. 
In this case, it is necessary to attach to the IPU since the code has to be run, so no `--compile-only` flag can be used.

In order to have a easily readable execution trace, it is good practice to modify the execution of your application. We would like for example to execute just a single step, and we would like that batches-per-step is set to 1. For models being executed in a pipeline configuration, we are going also to set the pipeline to the minimal value possible.
The previous suggestions are due to the fact that every step is the same so it is much easier to inspect just one of them, and the second is that an extremely deep pipeline will be difficult to navigate and inspect, and it is anyway the repetition of stages that are the same. We can then set the pipeline to a minimal value.
Let's make an example with a large application, like the phase1 training with LAMB:

```shell
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "autoReport.directory":"./execution_report"}' python run_pretraining.py --config configs/pretrain_large_128_phase1.json --steps 1 --batches-per-step 1 --gradient-accumulation-count 10
```
In this way the execution profile will be dropped together with the memory profile and we can inspect how many cycles the hardware is spending on each operation and this can give us insight on possible optimisations and improvements.

# Multi-Host training using PopDist <a name="popdist"></a>

PopDist is the tool shipped with the SDK that allows to run multiple SDK instances at the same time on the same or different hosts.
The config  files ending with '_POD64' and '_POD128' have been specifically designed to be trained using PopDist.

Before launching this we need to set up the V-IPU cluster and eventually a V-IPU partition, the procedure for which can be found in the [V-IPU user-guide](https://docs.graphcore.ai/projects/vipu-user/en/latest/). 
We need then to set up the dataset and the SDKs, it is important that these components are found on each host in the same global path. The same is valid for the virtual environment and for the `run_pretraining.py` script.

Further details on how to set up `poprun`, and the arguemnts used, can be found in the [docs]( https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/index.html).
The relevant set up is provided in the script given in `scripts/pretrain_distributed.sh`. This will be detailed in the following section.

## Script to train BERT Large on Graphcore IPU-POD64 <a name="popdist_script"></a>
We provide a utility script to run Phase 1 and Phase 2 pre-training on an IPU-POD64 machine. This script manages the config and checkpoints required for both phases of pre-training, as well as the necessary arguments to launch poprun on the IPU-POD64. 
This can be executed as:
``` shell
./scripts/poprun_pretraining.sh <host_address_1>
```
Inside this script the ssh keys will be copied across the hosts, as will the files in the BERT directory, as well as SDKs.
The default MPI options are set in the script, however you will need to check that the environment variables below are set correctly for your machine:
`PHASE1_CONFIG_PATH` : The path to the Phase 1 config file - this should not need changing  
`PHASE1_TRAIN_FILE` : The path to the Phase 1 (sequence length 128) wikipedia data 
`VIPU_PARTITION_NAME` : The name of the partition of the POD 64
`PHASE2_CONFIG_PATH` : The path to the Phase 2 config file - this should not need changing   
`PHASE2_TRAIN_FILE` : The path to the Phase 2 (sequence length 384) wikipdeia data

## IPU-POD128 Pre-training <a name="pod_128"></a>
PopDist is fundamental if we want to run on a IPU-POD128, an example of this is the IPU-POD128 config file in the confing folder.
As before, make sure that you configured the V-IPU cluster correctly, that the same folder structure for the python script, the SDK, the virtual environment and the dataset is present in all the hosts involved in the process.

We can then run the job using this simple command:

```shell
poprun -vv --host host_address_1,host_address_2 --num-ilds 2 --num-instances 2 --num-replicas 32 --ipus-per-replica 4 --numa-aware=yes --vipu-server-host=host_address_1 --reset-partition=no  --vipu-partition=pod128 --vipu-server-timeout=600 --mpi-global-args="--tag-output" python run_pretraining.py --config configs/pretrain_large_128_lamb_POD128.json --train-file '/path/to/tokenised/wikipedia/*.tfrecord'
```

As for the main application, there is the possibility to run PopDist in a `compile-only` mode.
If we want to perform this operation on a single server we need to first set the following environmental variable:

```shell
export POPLAR_TARGET_OPTIONS='{"ipuLinkDomainSize":"64"}'
```

We can then execute this command to trigger the execution:

```shell
 poprun --offline-mode=on  -vv --num-ilds 2 --num-instances 2 --num-replicas 32 --ipus-per-replica 4 --numa-aware=yes --mpi-global-args="--tag-output" python run_pretraining.py --config configs/pretrain_large_128_lamb_128ipus.json --generated-data --compile-only
```

With this command we are able to compile the previous IPU-POD128 job but on a much smaller machine. The only modifications that have to be applied are: --compile-only for the application and --offline-mode=on for poprun ensure that this will not attach to any IPU and the job will compile as if it was on a larger machine.
