

# GPT training on IPUs using PopXL

This README describes how to run GPT models for NLP pre-training on Graphcore IPUs using the PopXL library. A combination of phased execution, tensor model parallelism, data parallelism, remote tensor sharding and code loading are utilised to train the models.

**Note:** The model runs on a POD64. It has been tested to show that the training loss decreases as expected, so is “trainable” but it has not been trained from beginning to end and we do not provide a training dataset or a fully trained checkpoint alongside the model. Because the model has only been tested as trainable, it is possible that it contains a bug that we have not identified or has convergence problems, but we are not aware of any such difficulties.

# Table of contents
1. [File structure](#file_structure)
2. [Quick start guide](#quick_start)
    1. [Prepare environment](#prep_env)
    2. [Pre-training with GPT on IPU](#pretrain_IPU)
    3. [View the pre-training results in Weights & Biases](#wandb)
3. [Configure your GPT runs](#configs)
4. [Prepare datasets](#datasets)
5. [Scale GPT on IPU](#scale_GPT)
    1. [Phased execution and RTS](#pe)
    2. [Data parallel](#dp)
    3. [Tensor model parallel](#tp)
6. [Pre-training code details](#code_details)
7. [Layer execution diagrams](#execution)

## File structure <a name="file_structure"></a>

|    File                   |       Description                                                   |
|---------------------------|---------------------------------------------------------------------|
| `config/`                 | Contains configuration options for pre-training GPT.|
| `data/`            | Data preprocessing for pre-training |
| `demo/`            | Contains GPT pre-training on Wikipedia dataset scripts (`pretraining.py`).          |
| `execution/`       | Contains the execution scheme implementation (`pretraining.py`) which uses phased execution, tensor model parallelism, data parallelism, remote tensor sharding and code loading - all key features to scale up the GPT model size.. They are used in the corresponding `/demo`. The dataset used in this folder is synthetic for benchmarking and testing purposes.                  |
| `modelling/`       | Implements layers in GPT model and the models for different tasks for inference and training.<br/> -`embedding.py`, `attention.py`, and `feed_forward.py` present the implementations of embedding layer, self-attention, and  feed forward networks respectively. <br/> -`gpt_lm.py` present the implementation of the language model heads.<br/>                    |
| `tests/`           | Includes integration tests and unit tests.|
| `utils/`  | Helper functions to set up GPT configs and parse arguments.  |

## Quick start guide <a name="quick_start"></a>

### Prepare environment <a name="prep_env"></a>
**1) Download the Poplar SDK**

Download and install the Poplar SDK following the Getting Started guide for your IPU system. Source the `enable.sh` scripts for both Poplar and PopART, see more details in
[SDK installation](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#sdk-installation).

**2) Configure Python virtual environment**

Create a virtual environment, install the required packages, and add PopXL addons in PYTHONPATH:
```shell
virtualenv --python python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pre-training with GPT on IPU <a name="pretrain_IPU"></a>
You can run pre-training for GPT with settings defined in `pretraining.yml` by using the script below. You need to provide the data files to `--input_files`.
```shell
python3 demo/pretraining.py --input_files {path to your wikipedia data}/*.tfrecord
```
The default model size in demo pre-training is GPT-3 2.7B on POD64 (named `gpt3_2.7B_pod64`). You can change it to other sizes that are avaible in the 
configuration file `confif/pretraining.yml` using the `--config` CLI parameter like so. 
```
python3 demo/pretraining.py --config gpt3_2.7B_pod64 --input_files {path to your wikipedia data}/*.tfrecord
```

You can run these scripts for benchmarking with generated data by changing the directory name from `demo` to `execution`. For instance, the command below runs the benchmarking for GPT pre-training.
```shell
python3 execution/pretraining.py
```

When running the application, it is possible to save/load executables to/from a cache store. This allows for reusing a saved executable instead of re-compiling the model when re-running identical model configurations. To enable saving/loading from the cache store, use the environment variable `POPXL_CACHE_DIR=<PATH/TO/CACHE>` when running the application.

### View the pre-training results in Weights & Biases <a name="wandb"></a>
This project supports Weights & Biases, a platform to keep track of machine learning experiments. A client for Weights&Biases will be installed by default and can be used during training by passing the `--wandb` flag. You will need to manually log in (see the quickstart guide [here](https://docs.wandb.ai/quickstart)) and configure the project name with `--wandb-name`.) For more information please see https://www.wandb.com/.

The trainings in demo are logged in wandb under project `popxl-gpt`. Each run has loss, learning rate and throughput logged. The version for `addons` and PopXL are also logged together with the configuration settings.

## Configure your GPT runs <a name="configs"></a>

You can find configuration options for GPT in class `GPTConfig` in the file `config/config.py`. It contains configurations for these aspects:

* Models

    You can set the parameters used in the GPT model.
    - general parameters: 
        1. `layers` the number of decoder layers in the model, 
        2. `hidden_size` the hidden size of the layers,
        3. `sequence_length` number of tokens in a sample,
        4. `eval` to enable the model to be built for inference or validation which will disable dropout and optimisation,
        5. `dropout_prob` the dropout probability,
        6. `precision` to set the precision used in the model parameters, for instance, `popxl.float32` and `popxl.float16`.
        7. `seed` the random seed used by the model and data generation.
    - parameters for `embedding` layers: vocabulary size `vocab_size` and maximum number of positions to support in the embeddings `max_positional_length`.
    - parameters for `attention` layer: `heads` the number of attention heads.

* Training

    You can configure the training options that have impact on training. 
    - `steps`: number of steps, 
    - `epochs`: number of epochs, 
    - `global_batch_size`: the number of samples that contribute to an optimizer step,
    - `stochastic_rounding`: a flag to enable stochastic rounding,
    - `optimizer`: an optimizer with the following settings.
        - `name`: name of the optimizer, by default, AdamW.
        - `learning_rate`: to set up the learning rate including `function` used in scheduler, `maximum` learning rate, and `warmup_proportion` to set the proportion of the warmup step,
        - `beta1`: by default 0.9, 
        - `beta2`: by default 0.999, 
        - `weight_decay`: weight decay factor by default 0.0.

* Data
    - `input_files`: the path to input data files

* Execution

    It allows you to change how to execute a GPT run on IPU.
    - `micro_batch_size`: the number of samples that contribute to a gradient accumulation step,
    - `data_parallel`: the number of model replicas to use for data parallelism,
    - `tensor_parallel`: the number of IPUs used for tensor model parallel axis.
    - `device_iterations`: the number of times the training loop is executed before relinquishing control and reporting to the host,
    - `io_tiles`: the number of tiles dedicated to streaming data,
    - `available_memory_proportion`: the available memory proportion for any op that supports this option,
    - `loss_scaling`: the scaling factor to apply to gradients, by default 1,
    - `pipeline`: the pipeline layers distribution,

Note that the `gradient_accumulation` size is automatically computed from the `global_batch_size`, the `micro_batch_size` and `data_parallel`.

* Checkpoint

    You can set the path to load and save checkpoints respectively by `load` and `save`.

## Prepare datasets <a name="datasets"></a>
Instruction on how to obtain and prepare the dataset can be found in the [data folder README](data)

## Scale GPT on IPU <a name="scale"></a>
Here we introduce some techniques that were required to scale up the GPT model for the required capacity and throughput.

### Phased Execution and RTS <a name="pe"></a>
For compute graphs that have memory requirements greater than the available on-chip memory, we can partition it into a series of smaller sub-graphs and execute them in series on the IPU, using remote memory to store input and output tensors between calls. This is called phased execution. We recommend the tutorial of this concept in [Phased Execution in MNIST example](https://github.com/graphcore/popxl-addons/tree/master/examples/mnist/5_Phased_Execution).

In the GPT application we demonstrate this concept on a full sized model. Recomputation and replicated tensor sharding ([RTS](https://github.com/graphcore/popxl-addons/tree/master/examples/mnist/4_Remote_Variables#replicated-tensor-sharding)) are also used to improve the performance.

### Tensor Model Parallel <a name="tp"></a>
Tensor-parallel training involves breaking the layers into shards, which are each allocated to a different devices. Communication is required within a layer between the different devices to rematerialise the same numerical result if tensor parallelism sharding wasn't used. For the embedding layer one all-reduces communication operations are required for the forwards and backwards pass (not included recomputation). For the GPT layers, four all-reduce operations are required for the forwards and backwards pass. For the pre-training head four all-reduce operations are required for the forwards and backwards pass.

### Data Parallel <a name="dp"></a>
Data-parallel training involves breaking the training dataset up into multiple parts, which are each consumed by a model replica. At each optimization step, the gradients are mean-reduced across all replicas so that the weight update and model state are the same across all replicas. You can find more details about how to use data parallel in PopXL addons in [MNIST example](https://github.com/graphcore/popxl-addons/tree/master/examples/mnist/2_Data_Parallelism).

## Pre-training code details <a name="code_details"></a>

### Constructing computational graphs for each phase

First of all, we build the training graphs for each phase, represented in the class `Graphs`. A phase can include one layer or consecutive layers. The execution of a phase can be for the forward graph, gradient graph, optimizer graph or a combination of them. We need to build the graphs used in each phase before we define the phases in [Build the main computational graph](#main).

The graphs required for each phase can be represented in class `Graphs`. 
* The `fwd` and `bwd` are respectively the forward and backward pass graphs. The `bwd` graph is obtained directly by using `autodiff_with_accumulation` from the forward graph `fwd`.
* The `facts` has the required variable factories in the forward graph and optimizer graph. The `grad_facts` has the required variable factories for the backward graph. 
* The `optim` contains the optimizer graphs for each variable.
* The `buffers` are remote buffers used to handle the loading and offloading of the activations, trainable weights, and optimiser states. 
* To handle the remote load and store for the remote buffers, we also need the:      
    - graph `_fwd_load` that loads variables from `fwd` buffers and returns `_fwd_load_names`,
    - graph `_optim_fwd_load` that loads all forward and optimiser state from buffers
    - graph `_optim_fwd_store` that stores all forward and optimiser state to buffers
    - graph `_grad_store` that stores to `bwd` buffers. It is only used in pre-training GPT layer and task head layer.
* To handle collectives for replica all gather and reduce replica for RTS variables, we also created the graphs:
    - graph `_fwd_all_gather` that does AllGather across replicas for forward RTS variables and returns `_fwd_all_gather_names`,
    - graph `_grad_reduce` that reduces across replicas for gradient RTS variables and returns `_grad_reduce_names`.

We created these graphs:
* `embeddings` by calling the method `create_embeddings_graph` for the embedding layer. Note that the optimizer step for embedding layer happens straight after the backward pass on device, so there is no need to store the gradient in a buffer.
* `layer` by calling the method `create_decoder_block_graph` for each GPT decoder layer. Its buffer contains the forward tensors and gradient tensors. Since each GPT decoder layer has identical input and output data type and shape, we stack the buffers for each layer together. Hence, the number of entries in the buffers is the same as the number of decoder layers.
* `head` by calling the method `create_task_head_graph` for the task head layer. There are some slight differences in the implementation from the above two instances.
    * Its gradient graph is combined with the forward graph by using `GPTPretrainingLossAndGrad`. The calculation of gradients happens just after the forward graph calculation in the same phase. Hence, the `fwd` graph includes both the graph for forward pass and the calculation of its gradients.
    * Tied embedding is used. The linear layer in LM task head reuses the inputs embedding weights. As shown in the diagram below, in the forward pass the LM weights are loaded from the embedding layer weights buffer `embedding.buffers.fwd.word.weight`. In the backward pass, the gradient of the tied embedding weights is stored in a separate remote buffer `tied_weight_grad_buffer`.


#### Apply transformations on graphs

We then apply transformations to the graphs built:
* **recomputation**: to reduce memory consumption in backward pass for embedding gradients and decoder gradients. You can transform the gradient graphs by using `popxl_addons.recompute_graph` method.

* **batch serialisation**: to avoid the frequent loading and offloading of the variables and graphs in different layers for each batch, we use batch serialisation. It repeats the same graph with different data for each partition of the model for `steps` times. You can find the transformed graphs in `embeddings_batch_serialise`, `decoder_block_batch_serialise` and `head_batch_serialise` respectively. Each batch serialization produces the forward and gradient graphs and the activations. You can get the transformed graphs for the embedding and decoder layers by using the `popxl_addons.transforms.batch_serialisation.batch_serialise_fwd_and_grad` directly. As for head layer that has a combined forward and gradient graph, it uses `popxl_addons.transforms.batch_serialisation.batch_serialise`.

For batch serialisation, we also need to create remote buffers to load the inputs and store outputs for each partition by using `popxl_addons.batch_serial_buffer`. In this application, we use the remote buffers `x_buffer` and `dx_buffer` respectively to handle the intermediate outputs of each partition in the forward pass and backward pass. The two buffers for this application are illustrated in the following diagram. Each row handles `config.gradient_accumulation` elements.

![Buffers `x_buffer` and `dx_buffer`](imgs/bert_bs_buffers.png)

For instance, in `x_buffer`, row 0 stores the output of the embedding layer in forward pass. The output of each GPT decoder layer is stored from row 1 to `config.model.layers+1`. Note that the rows in the two buffers are filled up in the opposite directions.

## Execution of layers  <a name="execution"></a>

Below are diagrams demonstrating how each layer is executed during the forward, backward and optimiser steps.

### Forward layers

![Forward layer execution and communication](imgs/forward.png)

### Backwards layers

![Backward layer execution and communication](imgs/backward.png)

### Optimiser layers

![Optimiser layer execution and communication](imgs/optimiser.png)

Note that the optimiser layer operates directly on the RTS sharded gradient accumulators, optimiser state and variables.
