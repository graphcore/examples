# TensorFlow 2 BERT on Graphcore IPUs

## Table of contents

1. [Introduction](#intro)
2. [Datasets](#datasets)
3. [Quick start guide](#quick_start)
    * [Prepare environment](#prep_env)
    * [Pre-training with BERT on IPU](#pretrain_IPU)
    * [View the pre-training results in Weights & Biases](#wandb)
4. [Pre-training BERT on Wikipedia](#large_wiki)
5. [Fine-tuining BERT on SQuAD](#large_squad)
6. [Detailed overview of the config file format](#config)
7. [Notes for optimization approaches](#optimization)

## Introduction <a name='intro' ></a>

This directory demonstrates how to run the natural language model BERT (https://arxiv.org/pdf/1810.04805.pdf) on Graphcore IPUs utilising the [huggingface transformers library](https://huggingface.co/docs/transformers/index). 

There are two examples:

1. BERT for pre-training on masked Wikipedia data - `run_pretraining.py`
2. BERT for SQuAD (Stanford Question Answering Dataset) fine-tuning - `run_squad.py`

The BERT model in these examples is taken from the Huggingface transformers library and is converted into an IPU optimised format. 
This includes dynamically replacing layers of the model with their IPU specific counterparts, outlining repeated blocks of the model, recomputation, and pipelining the model to efficiently over several Graphcore IPUs.
The pretraining implementation of BERT uses the LAMB optimiser to capitalise on a large batch-size of 65k sequences in pre-training, and the SQuAD fine-tuning demonstrates converting a pre-existing Huggingface checkpoint. 

## Datasets <a name="datasets"></a>

The Wikipedia dataset contains approximately 2.5 billion word-piece tokens. This is only an approximate size since the Wikipedia dump file is updated all the time.

If full pre-training is required (with the two phases with different sequence lengths) then data will need to be generated separately for the two phases:

* once with --sequence-length 128 --mask-tokens 20
* once with --sequence-length 384 --mask-tokens 56

For further details on generating the datasets look at the PyTorch implementation [here](https://github.com/graphcore/examples/blob/v2.3.0/applications/pytorch/bert/README.md)

An example sample text and a corresponding `tfrecord` file can be found in the `data_utils/wikipedia/` directory to show the correct format.

## Quick start guide <a name="quick_start"></a>

### Prepare environment <a name="prep_env"></a>

**1) Download the Poplar SDK**

[Download](https://downloads.graphcore.ai/) and install the Poplar SDK following the Getting Started guide for your IPU system. Source the `enable.sh` script for poplar.

**2) Configure Python virtual environment**

Create a virtual environment and install the appropriate Graphcore TensorFlow 2.4 wheels from inside the SDK directory:

```shell
virtualenv --python python3.6 .bert_venv
source .bert_venv/bin/activate
pip install -r requirements.txt
pip install <path to the TensorFlow-2 wheel from the Poplar SDK>
pip install --force-reinstall --no-deps <path to the Keras wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 2 from the Poplar SDK>
```

**3) (Optional) Download Huggingface checkpoint**
To quickly run finetuning you can download a pre-trained model from the Huggingface [repository](https://huggingface.co/models) and run finetuning. 
The path to this checkpoint can be given in the command-line for `run_squad.py --pretrained-ckpt-path <PATH TO THE HUGGINGFACE CHECKPOINT>`.

## Running and benchmarking

To run a tested and optimised configuration and to reproduce the performance shown on our [performance results page](https://www.graphcore.ai/performance-results), please follow the setup instructions in this README to setup the environment, and then use the `examples_utils` module (installed automatically as part of the environment setup) to run one or more benchmarks. For example:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file>
```

Or to run a specific benchmark in the `benchmarks.yml` file provided:

```python
python3 -m examples_utils benchmark --spec <path to benchmarks.yml file> --benchmark <name of benchmark>
```

For more information on using the examples-utils benchmarking module, please refer to [the README](https://github.com/graphcore/examples-utils/blob/master/examples_utils/benchmarks/README.md).

### Pre-training with BERT on IPU <a name="pretrain_IPU"></a>

To validate that the machine and repository are set up correctly the BERT tiny model and sample text can be used. 

The `tests/pretrain_tiny_test.json` file is a small model that can be used for simple experiments. Note that if you don't specify the directory where the sample text file is stored, this will default to using the whole wikipedia dataset. 

```shell
python run_pretraining.py --config tests/pretrain_tiny_test.json --dataset-dir data_utils/wikipedia/
```

### View the pre-training results in Weights & Biases <a name="wandb"></a>

This project supports Weights & Biases, a platform to keep track of machine learning experiments. A client for Weights&Biases will be installed by default and can be used during training bypassing the `--wandb` flag. 
The user will need to manually login (see the quickstart guide [here](https://docs.wandb.ai/quickstart) and configure the project name with `--wandb-name`.)
For more information please see https://www.wandb.com/.

Once logged into wandb logging can be activated by toggling the `log_to_wandb` option in the config file. 
You can also name your wandb run with the flag `--name <YOUR RUN NAME HERE>`.

# Run Pre-Training Phase 1 <a name="large_wiki"></a>

Pre-Training BERT Phase 1 uses sequence length 128, and uses masked Wikipedia data to learn word and position embeddings - task specific performance is later tuned with finetuning.
This can be thought of as training the body of the model while the finetuning provides performance for specific heads.

The pre-training is managed by the `run_pretraining.py` script and the model configuration by the config files in `configs/`.
Provided configs are for `BASE` and `LARGE`, tuned to run on a Graphcore IPU POD system with 16 IPUs.

To run pre-training for BERT Base use the following command:

```shell
python run_pretraining.py --config configs/pretrain_base_128_phase1.json
```

Swapping the config for `pretrain_large_128.phase1.json` will train the BERT Large model. 
The path to the dataset is specified in the config, so ensure the path is correct for your machine, or give the path directly in the command-line with `--dataset-dir /localdata/datasets/wikipedia/128/`.

The results of this run can be logged to Weights and Biases. 

The resulting MLM loss curve will look like the following figure. You should see the double descent, characteristic of BERT pre-training, over the first 2000 steps before the loss plateaus to a value of approximately `1.4`. The exact result will vary depending on the Wikipedia dump and stochasticity.

<img src=./figs/MLM_loss_phase1.png width=80% height=80%>

# Run Pre-Training Phase 2  <a name="large_wiki_2"></a>

Phase 2 of pre-training is done using a sequence length of 384 with the masked Wikipedia dataset, this second phase is run using the same `run_pretraining.py` script as for phase 1.

To run pre-training phase 2 starting from a phase 1 checkpoint for BERT Base use the following command:

```shell
python run_pretraining.py --config configs/pretrain_base_384_phase2.json --pretrained-ckpt-path <PATH TO PHASE 1 CHECKPOINT>
```

# Fine-Tuning BERT for Question Answering with SQuAD 1.1 <a name="large_squad"></a>

Provided are the scripts to fine-tune BERT on the Stanford Question Answering Dataset (SQuAD 1.1), a popular question answering benchmark dataset. In version 1.1 there are no unanswerable questions.

To run on SQuAD, you will first need to download the dataset. The necessary training data can be found at the following link:
[train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)

Place the data in a directory `squad/` parallel to the Wikipedia data, the expected path can be found in the `configs/squad_base.json` file. 

To run BERT fine-tuning on SQuAD requires the same set-up as for pre-training, follow the steps in [Prepare Environment](#prep_env) to activate the SDK and install the required packages. 

The fine-tuning for a Large model on SQuAD 1.1 can be run with the following command:

``` shell
python3 run_squad.py configs/squad_base.json 
```

After fine-tuning the predictions file will be generated and the results evaluated.
You should expect to see results for BERT Base approximately as:
`{"f1": 87.97, "exact_match": 80.60}`.

# Detailed overview of the config file format <a name="config"></a>

The config files are how the bulk of the interaction with the model is conducted. These config files have two sections; firstly parameters that describe the model architecture (hidden layer size, number of attention heads, etc.) and differentiate between different BERT models; secondly, the parameters that describe the optimisation of the model for the IPU are given (such as batch size, loss scaling, outlining, and the pipeline configuration.)

To see the available configurations for both SQuAD and pre-training see the `JSON` files in the `config/` directory, and
To see the available options available to use in the command line interface use the `--help` argument:

```console
python3 run_pretraining.py --help
# or
python3 run_squad.py --help
```

For advanced users wanting to customise the models, you may wish to update the config.

Key parameters to consider  if customising the model are:

* `replicas` - the number of times to replicate the model, e.g., for a 4 IPU pipeline, 4x replication will use all 16 
   IPUs on an IPU-POD 16. Replicating a model is known as _data parallelism_, since each replica process a part of the 
   batch of samples. We call _micro batch_, the number of samples processed by each replica at a time, so they have to 
   fit in memory during the forward and backward passes; and we call _global batch_ the total number of samples  used 
   to estimate the gradient, which can include multiple micro batches per replica. 
* `micro_batch_size` - the size of the micro-batch that is seen by each replica
* `grad_acc_steps_per_replica` - the number of micro-batches that are accumulated on each replica before performing the 
   gradient update. 
* `global batch size` - this is not a direct parameter in the config file, but is derived from the product: 
   `global_batch_size = micro_batch_size * replicas * grad_acc_steps_per_replica`. In other words, this is the total 
   number of samples used to compute the single gradient update step. By accumulating over multiple micro batches and 
   over multiple replicas, we can use very large batch sizes, even if they don't fit in memory, which has been proved 
   to speed up training (see, e.g., https://arxiv.org/pdf/1904.00962.pdf). 
* `loss_scaling` - scaling factor used to scale the losses down to improve stability when training with large global  
   batch sizes and partial precision (float16) computations. 
* `matmul_available_memory_proportion_per_pipeline_stage` - the available memory proportion per IPU that `Poplar` 
   reserves for temporary values, or intermediate sums, in operations such as matrix multiplications or convolutions. 
   Reducing the memory proportion, reduces the memory footprint which allows to fit a larger micro batch in memory; but 
   it also constraints the `Poplar` planner, which can lead to lower throughput. 
* `pipeline_stages` - a nested list describing which model blocks are placed on which IPUs. The name abbreviations are 
   found in `model/pipeline_stage_names.py` and can be customised with different layers.
* `device_mapping` - a list mapping each of the pipeline stages onto each physical IPUs. An example of how this can be 
   customised can be seen in the BERT configurations where the _pooler_ layer and the heads are placed on IPU 0 with 
   the embeddings layer to improve performance.

# Multi-Host training using PopDist <a name="popdist"></a>

As explained in section [Detailed overview of the config file format](#config) above, the `replicas` parameter allows 
for _data parallelism_, replicating the model multiple times, such that each replica processes a part of the batch size. 

BERT Large with 8 samples per replica (micro batch) fits in 4 IPUs. Hence, we can use 4 replicas in a POD16 or 16 
replicas in a POD64. Both POD16 and POD64 are usually built in a rack system that includes a single host, hence they
can be run by simply setting the `replica` parameter to 4 or 16, respectively. In addition to increasing the value of the 
`replicas` parameter, we can use PopDist. PopDist is the tool shipped with the SDK that allows to run multiple SDK instances
at the same time on the same or different hosts. 

The script in `scripts/pretrain_distributed.sh` uses `poprun` to train BERT for pretraining with 16 replicas in a POD64 
(recall this can be done without `poprun` too).

The config  files ending with '_POD64' have been specifically designed to be trained using PopDist.

Before launching this we need to set up the V-IPU cluster and eventually a V-IPU partition, the procedure for which can 
be found in the [V-IPU user-guide](https://docs.graphcore.ai/projects/vipu-user/en/latest/). 
We need then to set up the dataset and the SDKs, it is important that these components are found on each host in the 
same global path. The same is valid for the virtual environment and for the `run_pretraining.py` script.

Further details on how to set up `poprun`, and the arguemnts used, can be found in the 
[docs]( https://docs.graphcore.ai/projects/poprun-user-guide/en/latest/index.html).
The relevant set up is provided in the script given in `scripts/pretrain_distributed.sh`. This will be detailed in the 
following section.

## Script to train BERT Large on Graphcore IPU-POD64 <a name="popdist_script"></a>

We provide a utility script to run Phase 1 and Phase 2 pre-training on an IPU-POD64 machine. This script manages the 
config and checkpoints required for both phases of pre-training.
This can be executed as:

``` shell
./scripts/pretrain_distributed.sh <MODEL> <CONFIG> <VIPU_CLUSTER_NAME> <VIPU_PARTITION_NAME> <VIPU_HOST> <HOSTS>
```

`MODEL`: One of 'base' or 'large'.

`CONFIG`: 'POD64'.

`VIPU_CLUSTER_NAME`: The name of the cluster in the POD. It can be obtained with `$ vipu list partition`, 
once logged in the POD.

`VIPU_PARTITION_NAME` : The name of the partition of the POD. It can be obtained with `$ vipu list partition`, 
once logged in the POD.

`VIPU_HOST`: IP address of VIPU host. Once logged in the POD, the host name can be obtained with 
`$ vipu --server-version`; then the IP can be obtained with `host <hostname>`. 

`HOSTS`: Space separated list of IP addresses where the instances will be run.

Inside this script the ssh keys will be copied across the hosts, as will the files in the BERT directory, as well as 
SDKs. Ensure your directory structure aligns with that used in the script, including the path to the wikipedia dataset.

# Notes for optimization approaches <a name="optimization"></a>
In order to reach optimized performance for the BERT model, the following optimization methods have been adopted:
* The Keras precision policy was set to float16 in `run_pretraining.py`, `run_squad.py` and `run_seq_classification.py`:
   ```
   policy = tf.keras.mixed_precision.Policy("float16")
   tf.keras.mixed_precision.set_global_policy(policy)
   ```
   Both compute and variable dtypes are set to float16. But note that the optimizer states, optimizer update and the loss functions are still in float32 for convergence purpose.
* To simulate larger batch sizes we use gradient accumulation, which accumulates gradients across multiple micro-batches together and then performs the weight update with the accumulated gradients. We use the [running mean](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/tensorflow/api.html?highlight=running_mean#tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod) accumulation method which performs a more stable running mean of the gradients.
* The model was pipelined over 4 IPUs. More information about pipelining can be found [here](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/optimising-performance.html#pipeline-execution-scheme).
* When pipelining, place the heads on the same IPU with the embeddings (typically IPU0) so that the weights for embeddings can be shared.
For example, for BERT large in the config file:
   ```
   "pipeline_stages": [
      ["emb", "hid", "hid", "hid"],
      ["hid", "hid", "hid", "hid", "hid", "hid", "hid"],
      ["hid", "hid", "hid", "hid", "hid", "hid", "hid"],
      ["hid", "hid", "hid", "hid", "hid", "hid", "hid"],
      ["enc_out", "pool", "heads"]
   ],
   "pipeline_device_mapping": [0, 1, 2, 3, 0]
   ```
   The pipeline stage "emb" was placed on IPU0 along with the "heads".
* Serialised embedding matmul: in `utilities/options.py`
   ```
   embedding_serialization_factor: int = 2
   ```
   The embedding matmul is serialized with the same factor as the matmul used in the IpuTFBertLMPredictionHead class in `model/ipu_lm_prediction_head.py` as it's a tied embedding. The ipu specific math operation `serialized_matmul` is used.
* Compute the MLM loss for the masked tokens only instead of over all tokens, as the calculation of loss on full sequence length requires more memory. This is done in `model/ipu_pretraining_model.py`. The number of masked tokens should be fixed and specified in the config files: 
   ```
   "max_predictions_per_seq": 20
   ```
   for phase 1 and
   ```
   "max_predictions_per_seq": 56
   ```
   for phase 2.
* Enable recomputation with checkpoints after each hidden layer. This will limit significantly the amount of not-always-live memory required.
The recompuation checkpoint is added with the class
   ```
   ModelAddRecomputationCheckpoints
   ```
   in `keras_extentions/model_transformations.py`. More information about [recomputation checkpoint](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/common-memory-optimisations.html#recomputation-checkpoints). 
* Offload the optimiser state. Guide can be found [here](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/common-memory-optimisations.html#variable-offloading).
* Replace the upstream GeLu activation with the IPU specific GeLu when calling the bert_config in `run_pretraining.py`, `run_squad.py` and `run_seq_classification.py`.
   ```
   bert_config = BertConfig(**config.bert_config.dict(), hidden_act=ipu.nn_ops.gelu)
   ```
* Replace upstream Dropout and LayerNorm layers with the IPU specific versions.
In `model/convert_bert_mode.py`:
   ```
   {Dropout: {"new_class": IpuDropoutCustom}},
   {LayerNormalization: {
      "new_class": IpuLayerNormalization,
      "copy_weights": True
   }
   ```
