# TensorFlow 2 BERT on Graphcore IPUs

# Table of contents
1. [Introduction](#intro)
2. [Datasets](#datasets)
3. [Quick start guide](#quick_start)
    * [Prepare environment](#prep_env)
    * [Pre-training with BERT on IPU](#pretrain_IPU)
    * [View the pre-training results in Weights & Biases](#wandb)
4. [Pre-training BERT on Wikipedia](#large_wiki)
5. [Fine-tuining BERT on SQuAD](#large_squad)
6. [Detailed overview of the config file format](#config)

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
- once with --sequence-length 128 --mask-tokens 20
- once with --sequence-length 384 --mask-tokens 56
For further details on generating the datasets look at the PyTorch implementation [here](https://github.com/graphcore/examples/blob/v2.3.0/applications/pytorch/bert/README.md)

An example sample text and a corresponding `tfrecord` file can be found in the `data_utils/wikipedia/` directory to show the correct format.

## Quick start guide <a name="quick_start"></a>

### Prepare environment <a name="prep_env"></a>
**1) Download the Poplar SDK**

[Download](https://downloads.graphcore.ai/) and install the Poplar SDK following the Getting Started guide for your IPU system. Source the `enable.sh` script for poplar.

**2) Configure Python virtual environment**

Create a virtual environment and install the appropriate Graphcore TensorFlow 2.4 wheel from inside the SDK directory:
```shell
virtualenv --python python3.6 .bert_venv
source .bert_venv/bin/activate
pip install -r requirements.txt
pip install <path to the TensorFlow-2 wheel from the Poplar SDK>
pip install <path to the ipu_tensorflow_addons wheel for TensorFlow 2 from the Poplar SDK>
```

**3) (Optional) Download Huggingface checkpoint**
To quickly run finetuning you can download a pre-trained model from the Huggingface [repository](https://huggingface.co/models) and run finetuning. 
The path to this checkpoint can be given in the command-line for `run_squad.py --pretrained-ckpt-path <PATH TO THE HUGGINGFACE CHECKPOINT>`.



### Pre-training with BERT on IPU <a name="pretrain_IPU"></a>
To validate that the machine and repository are set up correctly the BERT tiny model and sample text can be used. 

The `tests/pretrain_tiny_test.json` file is a small model that can be used for simple experiments. Note that if you don't specify the directory where the sample text file is stored, this will default to using the whole wikipedia dataset. 


```shell
python run_pretraining.py tests/pretrain_tiny_test.json --dataset-dir data_utils/wikipedia/
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
python run_pretraining.py configs/pretrain_base_128_phase1.json
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
python run_pretraining.py configs/pretrain_base_384_phase2.json --pretrained-ckpt-path <PATH TO PHASE 1 CHECKPOINT>
```


# Fine-Tuning BERT for Question Answereing with SQuAD 1.1 <a name="large_squad"></a>
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


# Detailed overview of the config file format. <a name="config"></a>
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
* `replicas` - the number of times to replicate the model. e.g for a 4 IPU pipeline, 4x replication will use all 16 IPUs on an IPU-POD 16
* `micro_batch_size` - the size of the micro-batch that is seen by each replica
    * `global batch size` - this is not a direct parameter in the config file, but is derived from the product `micro_batch_size * replicas * grad_acc_steps_per_replica`. This is the total number of sequences that will be processed in each batch, the loss is accumilated over these samples, and is used to compute the single gradient update step. By accumilating over multiple micro batches and over replicas it allows a much larger functional batch size to be used. 
* `grad_acc_steps_per_replica` - the number of micro-batches that are accumulated on each replica before performing the gradient update. This allows the use of extremely large batch sizes that facilitates extremely fast training. (https://arxiv.org/pdf/1904.00962.pdf)
* `loss_scaling` - a factor used to scale the losses and improve stability over replicated training.
* `matmul_available_memory_proportion_per_pipeline_stage` - the available memory proportion per IPU that that `Poplar` reserves for temporary values, or intermediate sums, in operations such as matrix multiplications or convolutions. This can help the model layout and improve performance. 
* `pipeline_stages` - a nested list describing which model blocks are placed on which IPUs. The name abbreviations are found in `model/pipeline_stage_names.py` and can be customised with different layers.
* `device_mapping` - a list mapping each of the pipeline stages onto each physical IPUs. An example of how this can be customised can be seen in the BERT configurations where the pooler layer and the heads are placed on IPU 0 with the embeddings layer to improve performance.


