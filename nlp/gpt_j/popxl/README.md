# GPT-J
GPT-J for NLP pre-training and text generation, optimised for Graphcore's IPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference |
|-----------|--------|-------|----------|-------|----------|-----------|
| popXL | NLP | GPT-J | MNLI | Next sentence prediction, Question/Answering | <p style="text-align: center;">✅ <br> Min. 16 IPUs (POD16) required | <p style="text-align: center;">✅ <br> Min. 16 IPU (POD16) required |

# Instructions summary

1. Install and enable the Poplar SDK (see Poplar SDK setup)

2. Install the system and Python requirements (see Environment setup)


## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash 
cd poplar-<OS version>-<SDK version>-<hash>
. enable.sh
```

3. Additionally, enable PopArt with:
```bash 
cd popart-<OS version>-<SDK version>-<hash>
. enable.sh
```

More detailed instructions on setting up your environment are available in the [poplar quick start guide](https://docs.graphcore.ai/projects/graphcloud-poplar-quick-start/en/latest/).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to this example's root directory

3. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```


## Dataset setup
This dataset is downloaded automatically when requireed by the example itself, there is no requirement to download it manually.


## Custom training/inference and other features

### Mnli finetuning <a name="finetuning"></a>

We present a fine-tuning example of GPT-J on [mnli dataset](https://huggingface.co/datasets/glue).
Mnli dataset consists of pairs of sentences, a *premise* and a *hypothesis*.
The task is to predict the relation between the premise and the hypothesis, which can be:
- `entailment`: hypothesis follows from the premise,
- `contradiction`: hypothesis contradicts the premise,
- `neutral`: hypothesis and premise are unrelated.


The default model size for fine-tuning is GPT-J 6B on POD64 (named `gptj_6B_1024_pod64`). You can
change it to other configurations that are available in the configuration file `config/finetuning.yml` using the `--config` CLI parameter.
In particular, you can run fine-tuning on a POD16 using
```bash
python3 run_finetuning_mnli.py --config gptj_6B_1024_pod16
```

When running the application, it is possible to save/load executables to/from a cache store. This allows for reusing a saved executable instead of re-compiling the model when re-running identical model configurations. To enable this, use the environment variable `POPXL_CACHE_DIR=<PATH/TO/CACHE>` when running the application:
```bash
POPXL_CACHE_DIR=<PATH/TO/CACHE> python3 run_finetuning_mnli.py
```

We finetune the model as a Causal Language Model (CLM): given a sequence of tokens, the task is to predict the next token.
Hence, we preprocess the `mnli` training dataset by forming input prompts with the format
```bash
mnli hypothesis: {hypothesis} premise: {premise} target: {class_label} <|endoftext|>
```
For example:
```
mnli hypothesis: Your contributions were of no help with our students' education. premise: Your contribution helped make it possible for us to provide our students with a quality education. target: contradiction <|endoftext|>
```

The tokenizer is GPT2 Tokenizer with some extra tokens.
Indeed, GPT-J embedding `vocab_size` is 50400 but GPT2 Tokenizer works with `50257` tokens. Therefore, remaining tokens are mapped to  `<|extratoken_1|>` ... `<|extratoken_143|>` (see also the [HF model doc](https://huggingface.co/docs/transformers/model_doc/gptj)).

Prompt sentences are tokenized and packed together to form 1024 token sequences, following [HF packing algorithm](https://github.com/huggingface/transformers/blob/v4.20.1/examples/pytorch/language-modeling/run_clm.py). No padding is used.
Since the model is trained to predict the next token, labels are simply the input sequence shifted by one token.
Given the training format, no extra care is needed to account for different sequences: the model does not need to know which sentence a token belongs to.

### Mnli validation <a name="validation"></a>
Generative inference is performed using a `greedy` heuristic: the next token is chosen based on the highest logits. No beam search or top-k/top-p techniques are
employed.
We run validation on Hugging Face `mnli` `validation_mismatched` and we measure accuracy using the corresponding metric.
The validation dataset is preprocessed to obtain input sentences in the prompt format
```bash
mnli hypothesis: {hypothesis} premise: {premise} target:
```
and target labels (one between `entailment`, `contradiction` and `neutral`).
After tokenization, the maximum length of sequences is computed.
Each sequence is right-padded to `max_len` + `output_len`, where `output_len` is the maximum number of new tokens we ask the model to generate. We set the `output_len` to 5 to accommodate all class labels and the `<|endoftext|>` token.
We use right padding so that the causal mask automatically accounts for padding.
GPTJTokenizer has no native padding token. However, we can safetly use the first `<|extratoken_1|>`.
To increase efficiency, we perform inference of micro batches.
Note that in a micro-batch each sequence has a different padding.
Since next token logits are located at the last non-padded token, we need to provide these indices to the batch inference algorithm.

Finally, we retrieve literal labels detokenizing the predictions and we compute the accuracy comparing the result with the expected one.

To run validation using a finetuned model, run
```bash
python3 run_mnli_validation.py --load {path_to_finetuned_checkpoint}
```
This script runs validation on the full dataset, producing the resulting accuracy.

If you just want to have a look at the outputs of a fine-tuned model, you can use the `run_inference.py` script instead:
```bash
python3 run_inference.py
```
Weights are taken from our Hugging Face checkpoint [Graphcore/gptj-mnli](https://huggingface.co/Graphcore/gptj-mnli).
The script runs inference on a single batch of the `mnli` `validation_mismatched` dataset and compares the output with the one produced by an HF model with the same weights.
To control the number of sentences, use the `micro_batch_size` parameter (default is 16):
```bash
python3 run_inference.py --micro_batch_size 4
```

### Benchmarking <a name="benchmarking"></a>
You can run execution scripts `inference.py` `finetuning_mnli.py` directly for benchmarking.
In that case, generated data will be used.
For instance, the command below runs the benchmarking for GPT-J mnli finetuning.
```bash
python3 finetuning_mnli.py
```

### View the pre-training results in Weights & Biases <a name="wandb"></a>
This project supports Weights & Biases, a platform to keep track of machine learning experiments. A client for Weights & Biases will be installed by default and can be used during training by passing the `--wandb` flag. You will need to manually log in (see the quickstart guide [here](https://docs.wandb.ai/quickstart)) and configure the project name with `--wandb-name`.) For more information please see https: // www.wandb.com/.

The trainings in demo are logged in wandb under project `popxl-gptj`. Each run has loss, learning rate and throughput logged. The version for `addons` and PopXL are also logged together with the configuration settings.

### Configure your GPT-J runs <a name="configs"></a>

You can find configuration options for GPT-J in class `GPTJConfig` in the file `config/config.py`. It contains configurations for these aspects:

* Models

    You can set the parameters used in the GPT-J model.
    - General parameters:
        1. `layers` the number of decoder layers in the model,
        2. `hidden_size` the hidden size of the layers,
        3. `sequence_length` number of tokens in a sample,
        4. `eval` to enable the model to be built for inference or validation which will disable dropout and optimisation,
        5. `dropout_prob` the dropout probability,
        6. `precision` to set the precision used in the model parameters, for instance, `popxl.float32` and `popxl.float16`.
        7. `seed` the random seed used by the model and data generation.
    - Parameters for `embedding` layers: vocabulary size `vocab_size`.
    - Parameters for `attention` layer: `heads` the number of attention heads, `rotary_dim` number of dimensions that rotary positional embedding is applied to,
    `rotary_positional_embeddings_base` base used for rotary embedding rotation

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

* Execution

    It allows you to change how to execute a GPT-J run on IPU.
    - `micro_batch_size`: the number of samples that contribute to a gradient accumulation step,
    - `data_parallel`: the number of model replicas to use for data parallelism,
    - `tensor_parallel`: the number of IPUs used for tensor model parallel axis.
    - `device_iterations`: the number of times the training loop is executed before relinquishing control and reporting to the host,
    - `io_tiles`: the number of tiles dedicated to streaming data,
    - `available_memory_proportion`: the available memory proportion for any op that supports this option,
    - `loss_scaling`: the scaling factor to apply to gradients, by default 1,

Note that the `gradient_accumulation` size is automatically computed from the `global_batch_size`, the `micro_batch_size` and `data_parallel`.

* Checkpoint

    You can set the path to load and save checkpoints respectively by `load` and `save`.
    ```bash
    python3 run_finetuning_mnli.py --save {path to your checkpoint file}
    ```

    ```bash
    python3 run_finetuning_mnli.py --load {path to your checkpoint file}
    ```

    ```bash
    python3 run_mnli_validation.py --load {path_to_finetuned_checkpoint}
    ```


### Scale GPT-J on IPU <a name="scale"></a>
Here we introduce some techniques that were required to scale up the GPT-J model for the required capacity and throughput.

### Combining data parallelism, tensor parallelism and RTS <a name="tp_dp"></a>
The model is executed using multiple IPUs to implement data parallelism and tensor parallelism via replication.

**Data parallelism** means that the same program(which can span over multiple devices) is duplicated on different sets of devices, and each copy is feeded with different data. At each optimization step, the gradients are mean-reduced so that the weight update and model state are the same across all replicas. You can find more details about it in the [data parallelism tutorial](https://github.com/graphcore/tutorials/tree/master/tutorials/popxl/3_data_parallelism).

<figure >
<img src="imgs/data_parallelism.png" width="800" alt="Data parallelism"/>
<figcaption> <em > <b > Fig 1: </b> Data parallelism. The model (which can span across multiple devices) is duplicated on several device sets. All copies have same program and same variables but are feeded with different data. </em> </figcaption>
</figure>

By itself, data parallelism it's just a way to increase the throughput and provides no memory gain.
Its real benefit in terms of memory comes when combined with **replicated tensor sharding** (see the tutorial about [remote variables and RTS](https://github.com/graphcore/tutorials/tree/master/tutorials/popxl/5_remote_variables_and_rts#replicated-tensor-sharding)).
Since each replica has the same variables we can shard them over the data parallel dimension, so that each replica only has
`num_elements/data_parallel` elements of the tensor.
![RTS](imgs/rts.png)
<figcaption><em><b> Fig 2: </b> Replicated tensor sharding. </em></figcaption>
</figure>


**Tensor model parallelism** is instead a type of model parallelism: certain computations in the layers are not performed with full-size tensors, but instead with sliced versions, and each device works with different shards.

![Tensor parallelism](imgs/tp.jpg)
<figcaption><em><b> Fig 3: </b> Tensor model parallelism: some variables are sharded and different devices have different shards and perform sharded computations. Collectives are needed to rematerialise the same numerical result if tensor parallelism wasn't used.</em></figcaption>


Communication is required within a layer between the different devices to rematerialise the same numerical result if tensor parallelism wasn't used.

<figure>
<img src="imgs/tensor_parallelism.png" width="800" alt="Tensor parallelism"/>
<figcaption><em><b> Fig 4: </b> Layers' computations are sharded across multiple devices. </em></figcaption>
</figure>

In the layers implementation, you will notice the use of the `addons` custom collectives `replicated_all_reduce_identical_inputs` `replicated_all_reduce_identical_grad_inputs`.
Operations happening between these functions are sharded and give different results on each device. You can really see these collectives as opening and closing blocks for sharded computations.
- `replicated_all_reduce_identical_inputs` is an identity in the forward graph, while its corresponding gradient op is an `all_reduce`, needed to rematerialise identical tensors when backpropagating from sharded computations.
- `replicated_all_reduce_identical_grad_inputs` is an `all_reduce` in the forward graph, needed to rematerialise identical tensors when coming from sharded computations, while its corresponding gradient op is an identity.

For example, below is the implementation of `feed_forward` layer:
```python
def build(self, x: popxl.Tensor, seed: Optional[popxl.Tensor]=None) -> List[popxl.Tensor]:
    # ----- Identical computation -----
    z=replicated_all_reduce_identical_inputs(
        x, group=self.replica_grouping.transpose())

    # ----- Sharded computation -----
    z=self.intermediate(z)
    z=ops.gelu(z)
    z=self.output(z)

    z=replicated_all_reduce_identical_grad_inputs(
        z, group=self.replica_grouping.transpose())

    # ----- Identical computation -----

    self.bias=self.add_variable_input(
        'bias', lambda: np.zeros(z.shape[-1]), z.dtype)
    z=z + self.bias

    if not self.config.model.eval and self.config.model.dropout_prob != 0.0:
        assert seed is not None, "A seed Tensor must be provided when creating a non-eval model."
        z=ops.dropout(z, seed, p=self.config.model.dropout_prob)
    return z
```

During training, these are the only tp related collectives we need, because we compute the cross entropy loss on sharded logits using `popxl_addons.ops.cross_entropy_sharded_loss`, so we don't need to rematerialise full logits by gathering them.
For the embedding layer one all-reduce communication operation is required for the forwards and backwards pass (not including recomputation). For the decoder layers, four all-reduce operations are required for the forwards and backwards pass . For the language model head four all-reduce operations are required for the forwards and backwards pass .

During inference we need to gather the sharded logits to retrieve the full set of probabilities. That is done in the `generate_greedy_tp` function in `modelling/gptj_lm.py`

When tensor parallelism and data parallelism are combined, it's important to understand that the `data_parallel` replication dimension is orthogonal to the `tensor_parallel` replication dimension.
The total number of IPUs required, the `replication_factor` of the `ir`, is the product between the two: our tensor parallel program spans across `tensor_parallel` IPUs, and this device set is replicated `data_parallel` times.

It should be clear from the above discussion that different communication is required in different for tensor parallel related collectives and data parallel related collectives. The diagram below illustrates the different device sets where communication happens.

<figure>
<img src="imgs/dp_tp.png" width="800" alt="Tensor parallelism and data parallelism" />
<figcaption><em><b> Fig 5: </b> When replication is used to implement data parallelism and tensor parallelism, communication related to the two techniques happens in different set of devices. </em></figcaption>
</figure>

Popxl's concept of `ReplicaGroup` allow us to handle all communication cases.
- The replica group of a variable represents the set of devices where the variable is the same.
- The replica group provided when a collective operation is created represents the set of devices that must communicate.

Let's call `tp=tensor_parallel` and `dp=data_parallel` and give some examples.
-  A tp-sharded variable (that is , a variable involved in sharded computations in a layer) is identical only on corresponding devices across the data parallel dimension, because in the tensor parallel dimension each device has a different shard of the variable. Hence, its replica group has a `group_size = dp` and a `stride = tp`.
-  A non-sharded variable (that is , a variable involved in identical computations in a layer) is equal on all devices. Hence, its replica group has a `group_size = dp*tp` and a `stride = 1`. This is the default replica group setting, identifying all replicas.
- In tensor parallel collectives we want to communicate along tp dimension. Hence, we use a replica group with `stride=1` and `group_size=tp`, which is the replica group transpose of the sharded variables group.
- In data parallel collectives (gradients reduction) we always want to communicate along dp dimension. Hence, the reduce group is always a replica group with `group_size=dp` and `stride=tp`.

Now let's have a look at RTS collectives.
If a variable is equal on X devices, regardeless how they are used, that variable can be sharded across all of them.
Therefore, the replica group of a variable defines the largest replica group for RTS, used in RTS collectives (gather of variables, scatter/slicing after reduction). It follows that in tensor parallel layers, tp-sharded variables have a different rts group (the dp-group) from identical variables (all replicas).

![Different rts groups in tensor parallel + data parallel scenario](imgs/tp_dp_rts.png)
<figcaption><em><b> Fig 6: </b> The replica group of a variable defines the largest set of devices for replicated tensor sharding. Therefore, variables that are identical on all devices (non tp-sharded) can be sharded on all replicas. Instead, tp-sharded variables are different along the tensor parallel axis and can be sharded only in the dp-group.</em></figcaption>

### Phased Execution and Batch serialisation <a name="pe"></a>
If a model requires greater memory than the available on-chip memory, we can partition it into a series of smaller graphs and execute them in series on the IPU, using remote memory to store variables and input/output tensors between calls (activations).
This is called phased execution. We recommend going through the tutorial [Phased Execution in MNIST example](https://github.com/graphcore/tutorials/tree/master/tutorials/popxl/6_phased_execution).
In the GPT-J application we demonstrate this concept on a full sized model.
As explained in the tutorial, **batch serialisation** is required to get the best performance from phased execution. It rearranges the gradient accumulation loop so that variables stored in remote memory are loaded just one time before the loop, while inputs are loaded inside the loop.
Hence, we apply the batch serialisation transform to our phases graphs.

### Code loading <a name="code_loading"></a>
By default, the code for each phase is always live on the IPU.
The code for each phase can instead be saved in remote memory and loaded to the IPU only when the phase needs to be executed.
Enable the `code_load` flag in the configs to use this optimisation.

### RTS on activations  <a name="activations_rts"></a>
When using phased execution, intermediate outputs (activations) need to be passed between phases.
With batch serialisation, each phase is executed N times and activations are saved in a remote buffer that stores the N outputs.
Since we are using replication to implement tensor parallelism, we can exploit the extra IPUs to shard activations, so that each replica just holds a slice of the tensor.
This saves remote memory and makes remote transfer faster, since less data has to be moved between DDR and IPUs.
After each remote load, sharded activations need to be gathered: this communication happens via IPU links.
As explained in the [RTS](https://github.com/graphcore/tutorials/tree/master/tutorials/popxl/5_remote_variables_and_rts#replicated-tensor-sharding) tutorial, using RTS
is a way of increasing the effective bandwidth because it performs part of the data transfer via IPU links, which have better bandwidth.
### Summary of execution scheme <a name="execution"></a>

Below is a diagram demonstrating how each layer is executed during the forward, backward and optimiser steps.

![Full execution](imgs/execution.jpg)
<figcaption><em><b> Fig 7: </b> Execution scheme for a layer during forward, backward and optimiser steps.
The layer is batch-serialised: computations included in the dashed repeat box are repeated for gradient accumulation times, but variables are loaded just once before the loop starts.
Since we apply RTS on the activations stored in x and dx buffers, gather and slice operations are inserted after loading and before storing from / to the buffers respectively.
TP related collectives happen during the layer execution in each gradient accumulation step.
Instead, collectives to gather variables and to reduce gradients happen only once per weight update.
The backward layer executes both forward and backward because recomputation is used.
The optimiser layer operates directly on the RTS sharded gradient accumulators, optimiser state and variables.
</em></figcaption>

#### Attention serialisation <a name="attention_serialisation"></a>
Computations in the transformers' attention layers are sequence-length dependent.
Recall that three linear projections Q, K, V are applied to obtain query, key and value vectors respectively.
- Q[batch, num_heads, seq_length, hidden_size]
- K[batch, num_heads, seq_length, hidden_size]
- V[batch, num_heads, seq_length, hidden_size]

The computation is then:

```python
attn_weights=query @ key.T    # [batch, num_heads, seq_length, seq_length]
# ... scaling and causal mask
attn_scores=ops.softmax(attn_weights, axis=-1)
# ... dropout
attn_output=attn_scores @ value   # [batch, num_heads, seq_length, hidden_size]
```

For big sequences these activations are big, and we need to store or recompute them during the backward phase.
The solution is attention serialisation: we take a slice of `Q` of size `f=seq_length/attention_serialisation` and serialise the calculation.
The pseudocode is :

```python
for q in split(Q):
    # q shape: [batch, num_heads, f, hidden_size]
    # key.T shape: [batch, num_heads, hidden_size, seq_length]
    attn_weights=q @ key.T    # [batch, num_heads, f, seq_length]
    # ... scaling and causal mask
    attn_scores=ops.softmax(attn_weights, axis=-1)
    # ... dropout
    attn_output=attn_scores @ value   # [batch, num_heads, f, hidden_size]

concat slices
```

This way, intermediate tensors and activations are smaller.
To build the backward graph, we autodiff the serialised step and we apply the recomputation transform to each slice independently,
so that full activations does not need to be live at the same time.
To achieve this the autodiff `called_graph_grad_info` parameter is used.
For this technique to reach optimal performance, it should be employed with `popxl.transforms.decompose_sum` transform.
Using this transform, gradients produced in a loop are progressively accumulated instead of being saved and then summed all at the end.
In this way, memory is saved because only the accumulated result needs to be alive for the whole loop duration.

You can look at the `create_decoder_block_graph` function in `finetuning_mnli.py` and to the attention layer in `modelling/attention.py` to understand how this is implemented in practice.

### Finetuning code details <a name="code_details"></a>

#### Constructing computational graphs for each phase

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
    - graph `_grad_store` that stores to `bwd` buffers. It is only used in pre-training GPTJ layer and task head layer.
* To handle collectives for replica all gather and reduce replica for RTS variables, we also created the graphs:
    - graph `_fwd_all_gather` that does AllGather across replicas for forward RTS variables and returns `_fwd_all_gather_names`,
    - graph `_grad_reduce` that reduces across replicas for gradient RTS variables and returns `_grad_reduce_names`.

We created these graphs:
* `embeddings` by calling the method `create_embeddings_graph` for the embedding layer. Note that the optimizer step for embedding layer happens straight after the backward pass on device, so there is no need to store the gradient in a buffer.
* `layer` by calling the method `create_decoder_block_graph` for each GPTJ decoder layer. Its buffer contains the forward tensors and gradient tensors. Since each GPTJ decoder layer has identical input and output data type and shape, we stack the buffers for each layer together. Hence, the number of entries in the buffers is the same as the number of decoder layers.
* `head` by calling the method `create_task_head_graph` for the task head layer. There are some slight differences in the implementation from the above two instances.
    * Its gradient graph is combined with the forward graph by using `GPTJPretrainingLossAndGrad`. The calculation of gradients happens just after the forward graph calculation in the same phase. Hence, the `fwd` graph includes both the graph for forward pass and the calculation of its gradients.
    * Unlike in gpt, gpt-j does not use of tied embedding.


#### Apply transformations on graphs

We then apply transformations to the graphs built:
* **recomputation**: to reduce memory consumption in backward pass for embedding gradients and decoder gradients. You can transform the gradient graphs by using `popxl_addons.recompute_graph` method.

* **batch serialisation**: to avoid the frequent loading and offloading of the variables and graphs in different layers for each batch, we use batch serialisation. It repeats the same graph with different data for each partition of the model for `steps` times. You can find the transformed graphs in `embeddings_batch_serialise`, `decoder_block_batch_serialise` and `head_batch_serialise` respectively. Each batch serialization produces the forward and gradient graphs and the activations. You can get the transformed graphs for the embedding and decoder layers by using the `popxl_addons.transforms.batch_serialisation.batch_serialise_fwd_and_grad` directly. As for head layer that has a combined forward and gradient graph, it uses `popxl_addons.transforms.batch_serialisation.batch_serialise`.

For batch serialisation, we also need to create remote buffers to load the inputs and store outputs for each partition by using `popxl_addons.batch_serial_buffer`. In this application, we use the remote buffers `x_buffer` and `dx_buffer` respectively to handle the intermediate outputs of each partition in the forward pass and backward pass . The two buffers for this application are illustrated in the following diagram. Each row handles `config.gradient_accumulation` elements. Since we use RTS over activations, these buffers are sharded.

![Buffers `x_buffer` and `dx_buffer`](imgs/bs_buffers.png)

For instance, in `x_buffer`, row 0 stores the output of the embedding layer in forward pass . The output of each GPT-J decoder layer is stored from row 1 to `config.model.layers+1`. Note that the rows in the two buffers are filled up in the opposite directions.
