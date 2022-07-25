# BERT training on IPUs using PopXL

This README describes how to run BERT models for NLP pretraining and fine-tuning tasks (SQuAD) on Graphcore IPUs using the PopXL library.

## Table of contents

1. [File structure](#file_structure)
2. [Quick start guide](#quick_start)
    1. [Prepare environment](#prep_env)
    2. [Pretraining with BERT on IPU](#pretrain_IPU)
    3. [Fine-tuning and inference with BERT for SQuAD on IPU](#squad)
    4. [View the pretraining results in Weights & Biases](#wandb)
3. [Configure your BERT runs](#configs)
4. [Prepare datasets](#datasets)
5. [Scale BERT on IPU](#scale_BERT)
    1. [Phased execution](#pe)
    2. [Data parallel](#dp)
6. [Avoid recompilation: caching executables](#cache)
7. [Profile your applications](#profiling)

## File structure <a name="file_structure"></a>

|    Directory       |       Description                                                   |
|--------------------|---------------------------------------------------------------------|
| `config/`          | Contains configuration options for running BERT.<br/> - `config.py`: Definition of configuration options for BERT.<br/> - `pretraining.yml` provides available parameter settings for three different sizes of BERT: large, base and tiny, and how to execute them for pretraining on IPU.<br/> - `squad_inference.yml` provides available parameter settings for the three different BERT sizes and how to execute them for SQuAD inference.<br/> - `squad_training.yml` provides parameter settings for three different BERT sizes and how to execute them for SQuAD fine-tuning.|
| `data/`            | Scripts for data preprocessing for pretraining and fine-tuning SQuAD, respectively, in `pretraining_data.py` and `squad_data.py`. |
| `demo/`            | Contains BERT pretraining MLM + NSP tasks on the Wikipedia dataset and fine-tuning on the SQuAD dataset.<br/> - `/pretraining/phased.py` presents how to run the pretraining with phased execution that is the key to scaling up the BERT model size.<br/> - `squad/` includes inference and fine-tuning training on the SQuAD dataset task with phased execution.            |
| `execution/`       | Contains the execution scheme implementations for phased execution. They are used in the corresponding `/demo`. The dataset used in this folder is synthetic for benchmarking purpose.                  |
| `modelling/`       | Implements layers in the BERT model and the models for different tasks for inference and training.<br/> - `embedding.py`, `attention.py`, and `feed_forward.py` present the implementations of embedding layers, self-attention, and  feed-forward networks respectively. <br/> - `mlm.py` and `nsp.py` present the implementation of the masked language model (MLM) task layer and the next sentence prediction (NSP) task layer respectively.<br/> - `bert_model.py` presents the implementation of a transformer layer as in `BertLayer` and MLM + NSP task head layer as in `BertPretrainingLossAndGrad`. <br/> - `squad.py` applies the BERT model for SQuAD.                    |
| `tests/`           | Includes integration tests and unit tests.|
| `utils/`           | Helper functions to set up BERT configs and parse arguments.  |

## Quick start guide <a name="quick_start"></a>

### Prepare environment <a name="prep_env"></a>

**1) Download the Poplar SDK**

Download and install the Poplar SDK following the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU system. Source the `enable.sh` scripts for both Poplar and PopART. See more details in
[SDK installation](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html#sdk-installation), for example.

**2) Configure Python virtual environment**

Create a virtual environment, install the required packages, and add PopXL addons in `PYTHONPATH`:

```shell
$ virtualenv --python python3.6 .bert_venv
$ source .bert_venv/bin/activate
$ pip install -r requirements.txt
```

### Pretraining with BERT on IPU <a name="pretrain_IPU"></a>

You can run pretraining for BERT base with the settings defined in `pretraining.yml` by using the script below. You need to provide the data files with `--input_files`.

```shell
$ python3 demo/pretraining/phased.py --input_files {path to your wikipedia data}/*.tfrecord
```

The default model size in demo pretraining is BERT base. You can change it to the BERT large with the command below.

```shell
$ python3 demo/pretraining/phased.py --config large --input_files {path to your wikipedia data}/*.tfrecord
```

You can run the scripts for benchmarking with generated data by replacing the directory name `demo` with `execution`. All the scripts in `execution` are for BERT large by default. For instance, the following command will run benchmark for BERT large pretraining. You can change it by adding `--config`. For instance, the command below runs benchmarking for BERT base pretraining.

```shell
$ python3 execution/pretraining/phased.py --config base
```

### Fine-tuning and inference with BERT for SQuAD on IPU <a name="squad"></a>

You can run fine-tuning on SQuAD for BERT large with the settings defined in `squad_training.yml` by using the command below. It will first load a pretrained checkpoint from Hugging Face.

```shell
$ python3 demo/squad/training_phased.py
```

You can also run inference on the trained SQuAD model with the settings defined in `squad_inference.yml` by using the command below.

```shell
$ python3 demo/squad/inference_phased.py
```

This outputs the context and questions for the BERT question-and-answer model, as well as the comparison of inference results from PopXL and Hugging Face.

You can benchmark the fine-tuning and inference by replacing the directory name `demo` with `execution`. All the scripts in `execution` run BERT large by default. You can change the model size by using `--config`. For instance, the script below will give benchmark results for fine-tuning on SQuAD with BERT base.

```shell
$ python3 execution/squad/training_phased.py --config base
```

### View the pretraining results in Weights & Biases <a name="wandb"></a>

This project supports Weights & Biases, a platform to keep track of machine learning experiments. A client for Weights & Biases will be installed by default and can be used during training by passing the `--wandb` flag. You will need to manually log in (see the [quickstart guide](https://docs.wandb.ai/quickstart)) and configure the project name with `--wandb-name`.) For more information see https://www.wandb.com/.

The trainings in demo are logged in wandb under project `popxl-bert`. Each run has loss, learning rate and throughput logged. The version for `addons` and PopXL are also logged together with the configuration settings.

## Configure your BERT runs <a name="configs"></a>

You can find configuration options for BERT in class `BertConfig` in the file `config/config.py`. It contains configurations for these aspects:

* Models

    You can set the parameters used in the BERT model.
    - general parameters:
        1. `layers`: the number of encoder layers in the model,
        2. `hidden_size`: the hidden size of the layers,
        3. `sequence_length`: number of tokens in a sample,
        4. `eval`: to enable the model to be built for inference or validation which will disable dropout and optimisation,
        5. `dropout_prob`: the dropout probability,
        6. `precision`: to set the precision used in the model parameters, for instance, `popxl.float32` and `popxl.float16`.
        7. `seed`: the random seed used by the model and data generation.
    - parameters for `emdedding` layers: vocabulary size, `vocab_size`, and maximum number of positions to support in the embeddings, `max_positional_length`.
    - parameters for `attention` layer: the number of attention heads, `heads`.
    - parameters for MLM task `mlm`: the maximum number of masked tokens in a sequence, `mask_tokens`.

* Training

    You can configure the training options that have an impact on training.
    - `steps`: number of steps,
    - `epochs`: number of epochs,
    - `global_batch_size`: the number of samples that contribute to an optimizer step,
    - `stochastic_rounding`: a flag to enable stochastic rounding,
    - `optimizer`: an optimizer with the following settings.
        - `name`: name of the optimizer, by default, AdamW.
        - `learning_rate`: to set up the learning rate including `function` used in scheduler, `maximum` learning rate, and `warmup_proportion` to set the proportion of the warmup step,
        - `beta1`: by default 0.9,
        - `beta2`: by default 0.999,
        - `weight_decay`: weight decay factor, by default 0.0.

* Data
    - `input_files`: the path to input data files.

* Execution

    This allows you to change how to execute a BERT run on IPU.
    - `micro_batch_size`: the number of samples that contribute to a gradient accumulation step,
    - `data_parallel`: the number of model replicas to use for data parallelism,
    - `device_iterations`: the number of times the training loop is executed before relinquishing control and reporting to the host,
    - `io_tiles`: the number of tiles dedicated to streaming data,
    - `available_memory_proportion`: the available memory proportion for any op that supports this option,
    - `loss_scaling`: the scaling factor to apply to gradients, by default 1.

* Checkpoint

    You can set the path to load and save checkpoints, respectively, with `load` and `save`.

## Prepare datasets <a name="datasets"></a>

We prepare the dataset for the pretraining task in `/data/pretraining_data.py` from TFRecord files. Note that we don't actually generate the datasets, just load/preprocess/postprocess them. To generate the datasets you should follow the [BERT in PyTorch](https://github.com/graphcore/examples/tree/master/nlp/bert/pytorch/) instructions.

The SQuAD dataset for the fine-tuning task is handled in `/data/squad_data.py`. This dataset will automatically be downloaded from Hugging Face.

## Scale BERT on IPU <a name="scale"></a>

Here we introduce some techniques we used to scale up the BERT model on IPUs in terms of memory consumption and training speed.

### Phased Execution <a name="pe"></a>

For compute graphs that have memory requirements greater than the available on-chip memory, we can partition them into a series of smaller sub-graphs and execute them in series on the IPU, using remote buffers in Streaming Memory to store input and output tensors between calls. This is called phased execution.

In the BERT application, we demonstrate this concept on a full sized model. Recomputation and replicated tensor sharding ([RTS](https://github.com/graphcore/popxl-addons/tree/master/examples/mnist/4_Remote_Variables#replicated-tensor-sharding)) are also used to improve the performance. Since most parts of the implementation of the phased execution in pretraining and fine-tuning are similar, in this README, we focus on the implementation of phased execution for pretraining in `execution/pretraining/phased.py`, and will show you the difference from SQuAD fine-tuning in `execution/squad/training_phased.py`.

Recall that we need to build an [IR in PopXL](https://docs.pages.gitlab.sourcevertex.net/docs/docs/PopART/popxl-user-guide/2.5.0/concepts.html#irs). In its main graph, we first define the input and output data streams. Then we build the computation graphs. As phased execution involves loading and offloading each partition in sequence, much use is made of remote buffers and RTS in the graph construction.

#### Constructing computation graphs for each phase

First of all, we build the training graphs for each phase, represented in the class `Graphs`. A phase can include one layer or consecutive layers. The execution of a phase can be for the forward graph, gradient graph, optimizer graph or a combination of them. We need to build the graphs used in each phase before we define the phases in [Build the main computation graph](#main).

The graphs required for each phase can be represented in class `Graphs`.

* `fwd` and `grad` are respectively the forward and backward pass graphs. The `grad` graph is obtained directly by using `autodiff_with_accumulation` from the forward graph `fwd`.
* `args` has the required arguments in the forward graph and optimizer graph. The `grad_args` has the required arguments for the backward graph.
* `optim` contains the optimizer graphs for each variable.
* The `buffers` are remote buffers used to handle the loading and offloading of the activations, trainable weights, and optimizer states.
* To handle the remote load and store for the remote buffers, we also need the graphs:
    - `_fwd_load` that loads variables from `fwd` buffers and returns `_fwd_load_names`,
    - `_optim_load` that load variables from `optim` buffers and returns `_optim_load_names`,
    - `_optim_store` that stores variables to `optim` buffers.
    - `_grad_store` that stores to `grad` buffers. It is only used in pretraining BERT layer and task head layer.
* To handle collectives for replica AllGather and reduce replica for RTS variables, we also created the graphs:
    - `_fwd_all_gather` that does AllGather across replicas for forward RTS variables and returns `_fwd_all_gather_names`,
    - `_grad_reduce` that reduces across replicas for gradient RTS variables and returns `_grad_reduce_names`.

In this BERT model, there are three types of layers:

* the embedding layer,
* each BERT encoder transformer layer, and
* the task head layer.

We created the following graphs for these:

* `embeddings`, by calling the method `create_embeddings_graph` for the embedding layer. Note that, the optimizer step for embedding layer happens straight after the backward pass on the IPU, so there is no need to store the gradient in a buffer.
* `layer`, by calling the method `create_layer_graph` for each BERT encoder layer. Its buffer contains the forward tensors and gradient tensors. Since each BERT encoder layer has identical input and output data type and shape, we stack the buffers for each layer together. Hence, the number of entries in the buffers is the same as the number of encoder layers.
* `head`, by calling the method `create_task_head_graph` for the task head layer. There are some slight differences in the implementation from the above two instances.
    * Its gradient graph is combined with the forward graph by using `BertPretrainingLossAndGrad`. The calculation of gradients happens just after the forward graph calculation in the same phase. Hence, the `fwd` graph includes both the graph for the forward pass and the calculation of its gradients.
    * Tied embedding is used. The linear layer in MLM task head reuses the inputs' embedding weights. As shown in the diagram below, in the forward pass the MLM weights are loaded from the embedding layer weights buffer `embedding.buffers.fwd.word.weight`. In the backward pass, the gradient of the tied embedding weights is stored in a separate remote buffer `tied_weight_grad_buffer`.

    ![Tied embedding](imgs/tied_embedding.png)

For SQuAD fine-tuning, the graphs for the SQuAD task head is created in `create_squad_graph`. Its gradient graph is combined with the forward graph from `BertSquadLossAndGrad`. No tied embedding is used.

#### Apply transformations on graphs

We then apply transformations to the graphs built:

* **recomputation**: to reduce memory consumption in the backward pass for embedding gradients and encoder gradients. You can transform the gradient graphs by using `popxl_addons.recompute_graph` method.

* **batch serialisation**: to avoid the frequent loading and offloading of the variables and graphs in different layers for each batch, we use batch serialisation. This repeats the same graph with different data for each partition of the model, for `steps` iterations. You can find the transformed graphs in `embeddings_batch_serialise`, `layer_batch_serialise` and `head_batch_serialise`, respectively. Each batch serialization produces the forward and gradient graphs and the activations. You can get the transformed graphs for the embedding and encoder layers by using the `popxl_addons.transforms.batch_serialisation.batch_serialise_fwd_and_grad` directly. As for the head layer that has a combined forward and gradient graph, it uses `popxl_addons.transforms.batch_serialisation.batch_serialise`.

For batch serialisation, we also need to create remote buffers to load the inputs and store outputs for each partition by using `popxl_addons.batch_serial_buffer`. In this application, we use the remote buffers `x_buffer` and `dx_buffer` respectively to handle the intermediate outputs of each partition in the forward pass and backward pass. The two buffers for this application are illustrated in the following diagram. Each row handles `config.gradient_accumulation` elements.

![Buffers `x_buffer` and `dx_buffer`](imgs/bert_bs_buffers.png)

For instance, in `x_buffer`, row 0 stores the output of the embedding layer in forward pass. The output of each BERT encoder layer is stored from row 1 to `config.model.layers+1`. Note that the rows in the two buffers are filled up in the opposite directions.

#### Build the main computation graph <a name="main"></a>

Once we initialize the required variables, we can build the main computation graph within the context of `popxl.in_sequence()`.

**Forward**

- load data from `input_streams`:
    1. learning rate to `lr`,
    2. random `seed`.
    3. masks. Note that masks are loaded and then stored in the remote buffer `mask_buffer` by using `fill_buffer_from_host`.
- Forward embedding layer phase in `embedding_fwd_phase`:
    1. load the embedding layer variables used in forward graph,
    2. propagate the seed to the following layer by using `split_random_seed`,
    3. call the embedding graph.
- Forward BERT encoder layers phases: load BERT encoder layers in a loop by using `ops.repeat`. This calls the graph once for each BERT encoder layer, in sequential order.

```python
            def single_bert_layer_fwd_phase(n: popxl.Tensor, seed: popxl.Tensor):
                # Load Encoder layers
                layer_vars = layer.fwd_load(n)
                layer_vars = layer.fwd_all_gather(layer_vars)
                # Forward
                seed, layer_seed = ops.split_random_seed(seed)
                layer.fwd.bind(layer_vars).call(n, layer_seed)
                return n + 1, seed

            i = popxl.constant(0, name="layer_index")
            bwd_graph = ir.create_graph(single_bert_layer_fwd_phase, i, seed)
            ops.repeat(bwd_graph, config.model.layers, i, seed)
```

- Forward and backward task head phase, `task_head_fwd_grad_phase`. The forward and backward graphs are combined in the same phase.
    1. load the optimizer state variables from remote buffer, forward variables from remote buffer then do an AllGather, and initialize the gradient variables.
    2. transpose the tied embedding weights to calculate the forward pass graph and the gradients.
    3. store the reduced tied embedding gradients in the remote buffer, `tied_weight_grad_buffer`.

Note that gradient clipping is used in pretraining but not in SQuAD fine-tuning. The order of applying the optimizer step is slightly different for the two. SQuAD fine-tuning does the optimizer step as soon as possible. Whereas pretraining does gradient clipping which requires the optimizer step to be *after* the global norm has been calculated for all the layers. Therefore, you can see the gradient calculation is followed immediately by `optimizer_step` and `optim_store`  in the backward phase of each layer in the `squad_training_phased`.

**Backward**

- Backward BERT encoder layers phase: repeatedly call the encoder layer's backward graph for `config.model.layers` times. Each calculation graph in `single_bert_layer_grad_phase`, calculates the gradients and the global norm. The graphs are called in the reverse order of the forward pass.

```python
            def single_bert_layer_grad_phase(n: popxl.Tensor, grad_norm: popxl.TensorByRef):
                # Load layer
                layer_vars = layer.fwd_load(n)
                layer_vars = layer.fwd_all_gather(layer_vars)
                # Gradient
                grads = layer.grad_args.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(layer_vars)
                layer.grad.bind(bwd_vars).call(n)
                # Data parallel reduce
                reduced_grads = layer.grad_reduce(grads)
                # Global Norm calculation
                global_norm_reduce(config, grad_norm, reduced_grads)
                # Store gradient
                layer.grad_store(reduced_grads, n)
                return n - 1

            i = popxl.constant(config.model.layers - 1, name="layer_index")
            bwd_graph = ir.create_graph(single_bert_layer_grad_phase, i, grad_norm)
            ops.repeat(bwd_graph, config.model.layers, i, grad_norm)
```

- Backward embedding phase in `embedding_grad_optimizer_phase`:
    1. load the optimizer states.
    2. call the gradient graph to calculate gradients.
    3. add the gradients calculated in task head `tied_weight_grad_buffer` to the embedding weights gradients
    4. finish calculating global norm for gradient clipping.
    5. apply optimizer step and store the updated weights.

**Optimizer**

- Apply optimizer step to BERT layers by repeating the `layer_optim`.

```python
            # Optimizer Step for Layers
            def layer_optim(n: popxl.Tensor, lr: popxl.Tensor, grad_norm: popxl.Tensor):
                layer_vars = layer.optim_load(n)
                optimizer_step(layer.optim, layer_vars, lr, grad_norm)
                layer.optim_store(layer_vars, n)
                return n + 1

            i = popxl.constant(0, name="layer_index")
            optim_graph = ir.create_graph(layer_optim, i, lr, grad_norm)
            ops.repeat(optim_graph, config.model.layers, i, lr, grad_norm)

```

- Apply optimizer step to task head layer.

#### Execution of the session

The main graph is repeated `config.execution.device_iterations` times. A training session is then created and returned in the `pretraining_phased` method. It is run in the `main()` method.

### Data Parallel <a name="dp"></a>

Data-parallel training involves breaking the training dataset up into multiple parts, which are each consumed by a model replica. At each optimization step, the gradients are mean-reduced across all replicas so that the weight update and model state are the same across all replicas. You can find more details about how to use data parallelism with PopXL addons in [MNIST example](https://github.com/graphcore/popxl-addons/tree/master/examples/mnist/2_Data_Parallelism).

## Avoid recompilation: caching executables <a name="cache"></a>

When running the application, it is possible to save and load executables in a cache store. This allows the reuse of a saved executable instead of re-compiling the model when re-running identical model configurations. To enable saving and loading from the cache store, use `POPART_CACHE_DIR <relative/path/to/cache/store>` when running the application.

## Benchmarking

To reproduce the benchmarks, please follow the setup instructions in this README to setup the environment, and then from this dir, use the `examples_utils` module to run one or more benchmarks. For example:
```
python3 -m examples_utils benchmark --spec benchmarks.yml
```

or to run a specific benchmark in the `benchmarks.yml` file provided:
```
python3 -m examples_utils benchmark --spec benchmarks.yml --benchmark <benchmark_name>
```

For more information on how to use the examples_utils benchmark functionality, please see the <a>benchmarking readme<a href=<https://github.com/graphcore/examples-utils/tree/master/examples_utils/benchmarks>


## Profile your applications <a name="profiling"></a>

You can generate the profiling files and visualize them in [PopVision](https://docs.graphcore.ai/projects/graph-analyser-userguide/). For instance, the profiling files for phased execution in the pretraining benchmark can be generated by using the following command:

```shell
$ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"pretrain_pe","profiler.replicaToProfile":"0"}' python3 execution/pretraining/phased.py
```

In the execution tab, you can find the execution trace of each phase. Below is a screenshot of the execution of one step of training.

![One training step execution trace.](imgs/bert_large_execution.png)
