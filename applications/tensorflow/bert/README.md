# BERT Training on IPUs using TensorFlow
This directory provides a script and recipe to run BERT models for NLP pre-training and training on Graphcore IPUs.

## Benchmarking

To reproduce the published Mk2 throughput benchmarks, please follow the setup instructions in this README, and then follow the instructions in [README_Benchmarks.md](README_Benchmarks.md) 

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
| `ipu_optimizer.py`        | IPU optimizer                                                       |
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
pip install <path to the tensorflow-1 wheel from the Poplar SDK>
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

#### Input and output files
`--input-file/--output-file` can take multiple arguments if you want to split your dataset between files.
When creating data for your own dataset, make sure the text has been preprocessed as specified at https://github.com/google-research/bert. This means with one sentence per line and documents delimited by empty lines.


#### Remasked datasets
The option `--remask` can be used to move the masked elements at the beginning of the sequence. This will improve the inference and training performance.

### Pre-training with BERT on IPU
Now that the data are ready we can start training our BERT tiny on the IPU!
Run this config:
```shell
python3 run_pretraining.py --config configs/pretrain_tiny_128_lamb.json --train-file ./Datasets/sample.tfrecord
```

The `configs/pretrain_tiny_128_lamb.json` file is a small model that can be used for simple experiments.

This config file has a first part that specifies the model, BERT tiny, the second part is more about the optimisation where we specify the learning rate, the learning rate scheduler the batch size and the optimiser (in this case we are using LAMB but other options can be used like momentum, ADAM, and SGD).

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
As first you need to create the Wikipedia pre-training data using the script in the Dataset folder, see the README there for the details.
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

Note that the configuration flag `--static-mask` must be set if the datasets was [generated with the remasking option](#remasked-dataset). A dataset that does not require the `--static-mask` flag may end up using more memory than a dataset that does, like the ones presented in the config files folder. Reducing the `--available-memory-proportion` parameter may be required in this case.

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
POPLAR_ENGINE_OPTIONS='{"autoReport.outputExecutionProfile":"true", "debug.allowOutOfMemory": "true", "debug.outputAllSymbols": "true", "autoReport.all": "true", "profiler.format":"v3", "autoReport.directory":"./execution_report"}' python run_pretraining.py --config configs/pretrain_large_128_phase1.json --steps 1 --batches-per-step 1 --gradient-accumulation-count 10
```
In this way the execution profile will be dropped together with the memory profile and we can inspect how many cycles the hardware is spending on each operation and this can give us insight on possible optimisations and improvements.

# Multi-Host training using PopDist

PopDist is the tool shipped with the SDK that allows to run multiple SDK instances at the same time on the same or different hosts.
In the config folder we added two config files, the ones ending with '_POD64' and '_POD128', that have been specifically designed to be trained using PopDist.
The POD64 configuration file can be run using a single host but in this case we suggest to perform a hyperparameter search in particular for the loss scaling parameter.

The first command we are intersted in allows us to run on a POD64 using not just a single host but 4, each of them running 4 SDK instances, one for each replica.
Before to launch this we need to set up the V-IPU cluster and eventually a V-IPU partition, procedure can be found in the [V-IPU user-guide](https://docs.graphcore.ai/projects/vipu-user/en/latest/). We need then to set up the dataset and the SDKs, it is important that these components are found into each host in the same global path. The same is valid for the virtual environment and also for the `run_pretraining.py` script.
After all the previous setup steps have been taken you can run the training using the following command:

```shell
poprun -vv --host host_address_1, host_address_2, host_address_3, host_address_4 --num-instances 16 --num-replicas 16 --ipus-per-replica 4 --numa-aware=yes --vipu-server-host=host_address_1 --vipu-partition=p_1-16 --mpi-global-args="--tag-output" --mpi-local-args="TF_POPLAR_FLAGS=--executable_cache_path=/path/to/cache/folder" python /path/to/run_pretraining.py --config configs/pretrain_large_128_phase1_POD64.json  --train-file '/path/to/tokenised/wikipedia/*.tfrecord'
```

This command will execute the entire training procedure on the Wikipedia dataset on the POD64, any extra logging may be added to the `--mpi-local-args` flag.
Some details that may be interesting are: 
- `--host` this contains the IP addresses of the hosts taking part to the training.
- `--vipu-partition` in this case we used a pre-existent partition. This is not mandatory since PopDist can create a partition for you on the fly, the only strict requirement is the presence of a V-IPU cluster.
- `--vipu-server-host` location of the V-IPU server.
For more information about the V-IPU, please refer to the docs.
The poprun command needs also to receive some information about the model itself:
- `--ipus-per-replicas` the number of IPUs required for each replica, 4 in the case of BERT LARGE.
- `--num-replicas` total number of replicas in total (in the POD64 example the total replicas are 16, 4 for each host)
- `--num-instances` number of instances created, in our case 16 so 4 for each host.

PopDist is fundamental if we want to run on a POD128, an example of this is the POD128 config file in the confing folder.
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

With this command we are able to compile the previous POD128 job but on a much smaller machine. The only modifications that have to be applied are: --compile-only for the application and --offline-mode=on for poprun ensure that this will not attach to any IPU and the job will compile as if it was on a larger machine.
