# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import time
import numpy as np
from typing import Dict, List, Union, Optional
from functools import partial

import popdist
import popxl
from popxl import ops
import popxl_addons as addons
from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons import TaskSession
from popxl_addons.patterns import apply_pre_alias_patterns

from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, fill_none_group, is_cross_instance, \
    get_instance_replica_grouping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.transforms.batch_serialisation import batch_serialise_fwd_and_grad, batch_serial_buffer, \
    batch_serialise
from popxl_addons.rts import (all_gather_replica_sharded_graph, replica_sharded_spec, reduce_replica_sharded_graph,
                              reduce_replica_sharded_tensor)
from popxl_addons.remote import (named_variable_buffers, load_remote_graph, store_remote_graph, NamedRemoteBuffers,
                                 create_remote_buffer)
from popxl_addons.ops.grad_reduce_square_add import grad_reduce_square_add

from config import GPTConfig, CONFIG_DIR
from utils.setup import gpt_config_setup
from modelling.embedding import GPTEmbeddingsTP
from modelling.decoder import GPTDecoderBlockTP
from modelling.gpt_lm import GPTLMHeadLossAndGradTP

__all__ = ["pretraining"]

OptimGraphs = Dict[str, GraphWithNamedArgs]
RTS_THRESHOLD = 0


def get_rts_groups(facts: NamedVariableFactories,
                   rts_group_if_none: Optional[popxl.ReplicaGrouping] = None) -> NamedReplicaGrouping:
    if rts_group_if_none:
        if is_cross_instance(rts_group_if_none):
            raise ValueError("rts group should span across a single instance. Use get_instance_replica_grouping")

    ir = popxl.gcg().ir
    none_value = rts_group_if_none or ir.replica_grouping(group_size=ir.instance_replication_factor,
                                                          stride=1)  # defaults to all replicas in a single instance

    def fill_none_and_get_instance(rg: popxl.ReplicaGrouping):
        rg = rg or none_value
        rg = get_instance_replica_grouping(rg)
        return rg

    rts_groups = facts.replica_groupings.map(fill_none_and_get_instance)
    return rts_groups


def get_variable_groups(facts: NamedVariableFactories,
                        var_group_if_none: Optional[popxl.ReplicaGrouping] = None) -> NamedReplicaGrouping:
    default_for_none = var_group_if_none or popxl.gcg().ir.replica_grouping()  # defaults to all replicas
    func_to_apply = partial(fill_none_group, none_value=default_for_none)
    return facts.replica_groupings.map(func_to_apply)


def requires_weight_decay(t: popxl.Tensor):
    return not any(map(lambda exclude: exclude in t.name, ["norm", "bias"]))


def optimizer_graphs(config: GPTConfig, optimizer: addons.Module, variables: NamedTensors,
                     replica_groups: NamedReplicaGrouping):
    optim_facts = {}
    optim_graphs = {}
    replica_groups = replica_groups.to_dict()
    for name, var in variables.to_dict().items():
        # currently assumes grads have the same replica group as their var. Can change this if needed.
        optim_facts[name], optim_graphs[name] = optimizer.create_graph(
            replica_sharded_spec(var, threshold=RTS_THRESHOLD, replica_grouping=replica_groups[name]),
            replica_sharded_spec(var, threshold=RTS_THRESHOLD, replica_grouping=replica_groups[name]),
            lr=popxl.TensorSpec((), popxl.float32),
            replica_grouping=replica_groups[name],
            weight_decay=config.training.optimizer.weight_decay if requires_weight_decay(var) else 0.0,
            beta1=config.training.optimizer.beta1,
            beta2=config.training.optimizer.beta2,
            eps=1e-6,
            bias_correction=True,
            first_order_dtype=popxl.float32,
            loss_scaling=config.execution.loss_scaling,
            global_norm=popxl.TensorSpec((), popxl.float32),
            global_norm_max=config.training.optimizer.gradient_clipping)
    return NamedVariableFactories.from_dict(optim_facts), optim_graphs


class Graphs:
    def __init__(self):
        self.fwd: GraphWithNamedArgs
        self.bwd: GraphWithNamedArgs
        self.optim: OptimGraphs
        self.facts: NamedVariableFactories
        self.grad_facts: NamedVariableFactories
        self.buffers: NamedRemoteBuffers
        self._fwd_load: GraphWithNamedArgs
        self._fwd_load_names: List[str]
        self._grad_store: GraphWithNamedArgs
        self._optim_fwd_load: GraphWithNamedArgs
        self._optim_fwd_load_names: List[str]
        self._optim_fwd_store: GraphWithNamedArgs
        self._fwd_all_gather: GraphWithNamedArgs
        self._fwd_all_gather_names: List[str]
        self._grad_reduce: GraphWithNamedArgs
        self._grad_reduce_names: List[str]

    def fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._fwd_load_names, self._fwd_load.call(i))

    def grad_store(self, args: NamedTensors, i: Union[float, popxl.Tensor]):
        return self._grad_store.bind(args).call(i)

    def optim_fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._optim_fwd_load_names, self._optim_fwd_load.call(i))

    def optim_fwd_store(self, args: NamedTensors, i: Union[int, popxl.Tensor]):
        return self._optim_fwd_store.bind(args).call(i)

    def fwd_all_gather(self, args: NamedTensors):
        return NamedTensors.pack(self._fwd_all_gather_names, self._fwd_all_gather.bind(args).call())

    def grad_reduce(self, args: NamedTensors):
        return NamedTensors.pack(self._grad_reduce_names, self._grad_reduce.bind(args).call())


def create_embeddings_graph(config: GPTConfig, optimizer: addons.Module, *args, **kwargs):
    ir = popxl.gcg().ir
    embeddings = Graphs()

    # Create Graphs for computing forward, gradient and optimizer
    fwd_facts, embeddings.fwd = GPTEmbeddingsTP(config).create_graph(*args, **kwargs)

    # where the variables are equal. If a variable has None as rg, it is assumed equal on all replicas.
    fwd_var_groups = get_variable_groups(fwd_facts)
    dp_group = popxl.gcg().ir.replica_grouping(stride=config.execution.tensor_parallel,
                                               group_size=config.execution.data_parallel)

    # Embedding needs no onward gradients
    required_grads = ()
    grad_facts, embeddings.bwd = addons.autodiff_with_accumulation(embeddings.fwd,
                                                                   embeddings.fwd.args.tensors,
                                                                   required_grads,
                                                                   replica_groupings=fwd_var_groups)

    optim_facts, embeddings.optim = optimizer_graphs(config,
                                                     optimizer,
                                                     embeddings.fwd.args,
                                                     replica_groups=fwd_var_groups)
    # Variables required
    embeddings.facts = NamedVariableFactories(fwd=fwd_facts, optim=optim_facts)
    embeddings.grad_facts = grad_facts

    # Create remote buffers using only forward facts
    # Embedding optimizer step happens straight after the bwd: no need to store the gradient in a buffer.
    rts_fwd_optim_groups = get_rts_groups(embeddings.facts)
    embeddings.buffers = named_variable_buffers(embeddings.facts,
                                                sharded_threshold=RTS_THRESHOLD,
                                                shard_groups=rts_fwd_optim_groups)

    # Create Graphs for loading/gathering/storing/reducing
    embeddings._optim_fwd_load, embeddings._optim_fwd_load_names = load_remote_graph(embeddings.buffers)
    embeddings._optim_fwd_store = store_remote_graph(embeddings.buffers)
    embeddings._fwd_load, embeddings._fwd_load_names = load_remote_graph(embeddings.buffers.fwd)

    embeddings._fwd_all_gather, embeddings._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(embeddings._fwd_load_names, embeddings._fwd_load.graph.outputs),
        replica_groups=rts_fwd_optim_groups.fwd)

    grad_accums = embeddings.bwd.args.copy()
    grad_accums.pop("mean_accum_counter")
    rts_bwd_group = NamedReplicaGrouping(accum=rts_fwd_optim_groups.fwd.copy())
    embeddings._grad_reduce, embeddings._grad_reduce_names = reduce_replica_sharded_graph(grad_accums,
                                                                                          'mean',
                                                                                          threshold=RTS_THRESHOLD,
                                                                                          shard_groups=rts_bwd_group,
                                                                                          replica_group=dp_group)
    return embeddings


def create_decoder_block_graph(config: GPTConfig, optimizer: addons.Module, *args, **kwargs):
    layer = Graphs()
    ir = popxl.gcg().ir

    # Create Graphs for computing forward, gradient and optimizer
    fwd_facts, layer.fwd = GPTDecoderBlockTP(config).create_graph(*args, **kwargs)
    required_grads = (layer.fwd.graph.inputs[0], )

    fwd_var_groups = get_variable_groups(fwd_facts)
    dp_group = ir.replica_grouping(stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel)

    grad_facts, layer.bwd = addons.autodiff_with_accumulation(layer.fwd,
                                                              layer.fwd.args.tensors,
                                                              required_grads,
                                                              replica_groupings=fwd_var_groups)

    optim_args, layer.optim = optimizer_graphs(config, optimizer, layer.fwd.args, replica_groups=fwd_var_groups)

    # Variables required
    layer.facts = NamedVariableFactories(fwd=fwd_facts, optim=optim_args)
    layer.grad_facts = grad_facts

    # Create remote buffers
    entries = config.model.layers
    buffer_facts = layer.facts.copy()
    buffer_facts.insert("bwd", grad_facts.copy())
    buffer_facts.bwd.pop("mean_accum_counter")

    rts_fwd_bwd_group = get_rts_groups(buffer_facts)
    layer.buffers = named_variable_buffers(buffer_facts,
                                           entries,
                                           sharded_threshold=RTS_THRESHOLD,
                                           shard_groups=rts_fwd_bwd_group)

    # Create Graphs for loading/gathering/storing/reducing
    # Load fwd, bwd and optim
    layer._optim_fwd_load, layer._optim_fwd_load_names = load_remote_graph(layer.buffers, entries)

    buffers = layer.buffers.copy()
    buffers_grad = buffers.pop("bwd")
    # Store fwd and optim
    layer._optim_fwd_store = store_remote_graph(buffers, entries)
    # Store bwd
    layer._grad_store = store_remote_graph(buffers_grad, entries)
    layer._fwd_load, layer._fwd_load_names = load_remote_graph(layer.buffers.fwd, entries)

    layer._fwd_all_gather, layer._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(layer._fwd_load_names, layer._fwd_load.graph.outputs), replica_groups=rts_fwd_bwd_group.fwd)

    grad_accums = layer.bwd.args.copy()
    grad_accums.pop("mean_accum_counter")

    layer._grad_reduce, layer._grad_reduce_names = reduce_replica_sharded_graph(grad_accums,
                                                                                'mean',
                                                                                threshold=RTS_THRESHOLD,
                                                                                shard_groups=rts_fwd_bwd_group.bwd,
                                                                                replica_group=dp_group)

    return layer


def create_task_head_graph(config: GPTConfig, optimizer: addons.Module, embeddings: Graphs, *args, **kwargs):
    """Combines the LM forward (which includes an initial layer norm, normally at the end of the gpt decoder stack),
     loss and bwd into a single Module."""
    head = Graphs()
    ir = popxl.gcg().ir

    # the head weight is tied to word embedding. Head variables are only layer norm
    facts, graph = GPTLMHeadLossAndGradTP(config).create_graph(*args, **kwargs)

    fwd_bwd_var_groups = get_variable_groups(facts)
    dp_group = ir.replica_grouping(stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel)

    optim_ts = graph.args.fwd.copy()
    optim_facts, optim_graphs = optimizer_graphs(config, optimizer, optim_ts, replica_groups=fwd_bwd_var_groups.fwd)

    facts.insert("optim", optim_facts)
    head.fwd = graph
    head.optim = optim_graphs
    head.facts = facts
    head.grad_facts = facts.pop("bwd")

    # Create remote buffers for layer norm weights
    buffer_facts = head.facts.copy()
    buffer_facts.insert("bwd", head.grad_facts.copy())

    rts_fwd_bwd_groups = get_rts_groups(buffer_facts)
    head.buffers = named_variable_buffers(buffer_facts,
                                          sharded_threshold=RTS_THRESHOLD,
                                          shard_groups=rts_fwd_bwd_groups)
    # Create graphs to load/store layer norm weights and related
    head._optim_fwd_load, head._optim_fwd_load_names = load_remote_graph(head.buffers)

    buffers = head.buffers.copy()
    buffers_bwd = buffers.pop("bwd")

    # don't include the tied weights, store handled separately
    head._optim_fwd_store = store_remote_graph(head.buffers)
    head._grad_store = store_remote_graph(buffers_bwd)
    # Add the tied weight buffer to the buffers for the fwd load, this
    # this way the embedding weight will be loaded together with the layer norm vars
    rts_fwd_bwd_groups.fwd.insert("weight",
                                  get_instance_replica_grouping(embeddings.facts.fwd.word.weight.replica_grouping))
    head.buffers.fwd.insert("weight", embeddings.buffers.fwd.word.weight)

    head._fwd_load, head._fwd_load_names = load_remote_graph(head.buffers.fwd)
    head._fwd_all_gather, head._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(head._fwd_load_names, head._fwd_load.graph.outputs), replica_groups=rts_fwd_bwd_groups.fwd)

    # tied weight handled elsewhere
    head._grad_reduce, head._grad_reduce_names = reduce_replica_sharded_graph(graph.args.bwd,
                                                                              'mean',
                                                                              threshold=RTS_THRESHOLD,
                                                                              shard_groups=rts_fwd_bwd_groups.bwd,
                                                                              replica_group=dp_group)
    return head


def embeddings_batch_serialise(config: GPTConfig, embeddings: Graphs, input_streams: addons.InputStreams,
                               x_buffer: popxl.RemoteBuffer, dx_buffer: popxl.RemoteBuffer):
    fwd, bwd = batch_serialise_fwd_and_grad(embeddings.fwd,
                                            embeddings.bwd,
                                            embeddings.fwd.args,
                                            config.gradient_accumulation,
                                            load_handles={
                                                embeddings.fwd.graph.inputs[0]: input_streams.words,
                                                embeddings.bwd.graph.inputs[0]: (dx_buffer, 0)
                                            },
                                            store_streams={},
                                            store_buffers={embeddings.fwd.graph.outputs[0]: (x_buffer, 0)},
                                            seed_input=embeddings.fwd.graph.inputs[2],
                                            rows=1,
                                            io_mode='io')
    embeddings.fwd = fwd.graph
    embeddings.bwd = bwd.graph


def decoder_block_batch_serialise(config: GPTConfig, layer: Graphs, x_buffer: popxl.RemoteBuffer,
                                  dx_buffer: popxl.RemoteBuffer):
    fwd, bwd = batch_serialise_fwd_and_grad(layer.fwd,
                                            layer.bwd,
                                            layer.fwd.args,
                                            config.gradient_accumulation,
                                            load_handles={
                                                layer.fwd.graph.inputs[0]: (x_buffer, 0),
                                                layer.bwd.graph.inputs[0]: (dx_buffer, 1)
                                            },
                                            store_streams={},
                                            store_buffers={
                                                layer.fwd.graph.outputs[0]: (x_buffer, 1),
                                                layer.bwd.graph.outputs[0]: (dx_buffer, 0)
                                            },
                                            seed_input=layer.fwd.graph.inputs[1],
                                            rows=config.model.layers,
                                            io_mode='io')
    layer.fwd = fwd.graph
    layer.bwd = bwd.graph


def head_batch_serialise(config: GPTConfig, head_graph: GraphWithNamedArgs, input_streams: addons.InputStreams,
                         output_streams: addons.OutputStreams, x_buffer: popxl.RemoteBuffer,
                         dx_buffer: popxl.RemoteBuffer) -> GraphWithNamedArgs:
    bs_head = batch_serialise(head_graph,
                              config.gradient_accumulation,
                              load_handles={
                                  head_graph.graph.inputs[0]: (x_buffer, config.model.layers),
                                  head_graph.graph.inputs[1]: input_streams.labels,
                              },
                              store_streams={head_graph.graph.outputs[0]: output_streams.loss},
                              store_buffers={head_graph.graph.outputs[1]: (dx_buffer, config.model.layers)},
                              io_mode='io')
    return bs_head.graph


def get_optimizer_state(name: str, state: NamedTensors) -> NamedTensors:
    attrs = name.split(".")
    for attr in attrs:
        state = getattr(state, attr)
    return state


def optimizer_step(optim_graphs: OptimGraphs, ts: NamedTensors, lr: popxl.Tensor, global_norm: popxl.Tensor):
    _variables = ts.fwd.to_dict()
    _state = ts.optim
    _grads = ts.bwd.accum.to_dict()
    for name, graph in optim_graphs.items():
        graph.bind(get_optimizer_state(name, _state)).call(_variables[name], _grads[name], lr, global_norm)


def task_head_optimizer_step(optim_graphs: OptimGraphs, ts: NamedTensors, lr: popxl.Tensor, global_norm: popxl.Tensor):
    _variables = ts.fwd.to_dict()
    _state = ts.optim
    _grads = {name.replace("accum.", ""): t for name, t in ts.bwd.to_dict().items()}
    for name, graph in optim_graphs.items():
        graph.bind(get_optimizer_state(name, _state)).call(_variables[name], _grads[name], lr, global_norm)


def global_norm_reduce(config: GPTConfig, grad_norm: popxl.Tensor, grads: NamedTensors):
    for g in grads.tensors:
        ops.add_(grad_norm, grad_reduce_square_add(g, config.execution.loss_scaling))


def pretraining(config: GPTConfig) -> TaskSession:
    replicas = config.execution.data_parallel * config.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas
    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"

    t = time.time()
    main = ir.main_graph
    dp_group = ir.replica_grouping(stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel)

    with main:
        # -----  Define input and output streams -----
        input_shape = (config.execution.micro_batch_size * config.model.sequence_length, )
        input_streams = addons.InputStreams(words=(input_shape, popxl.int32),
                                            labels=(input_shape, popxl.int32),
                                            seed=((2, ), popxl.uint32),
                                            lr=((), popxl.float32))
        output_streams = addons.OutputStreams(loss=((), config.model.dtype), grad_norm=((), popxl.float32))

        # Create a "constant" tensor for positions
        positions_offseted_np = GPTEmbeddingsTP.offset_inputs(config)
        positions = popxl.variable(positions_offseted_np, popxl.int32, name="positions", replica_grouping=dp_group)

        # Create a "constant" tensor for word offset
        word_offsets_data, _ = GPTEmbeddingsTP.get_offsets(config)
        word_offset = popxl.variable(word_offsets_data, popxl.int32, 'word_offset', replica_grouping=dp_group)

        # ----- Build compute graphs -----
        optimizer = AdamOptimizerStep()

        embeddings = create_embeddings_graph(config,
                                             optimizer,
                                             input_streams.words.spec,
                                             positions.spec,
                                             seed=input_streams.seed.spec)

        decoder_block = create_decoder_block_graph(config,
                                                   optimizer,
                                                   embeddings.fwd.graph.outputs[0],
                                                   seed=input_streams.seed.spec)

        tied_weight_spec = popxl.TensorSpec(
            (embeddings.fwd.args.word.weight.shape[1], embeddings.fwd.args.word.weight.shape[0]),
            dtype=config.model.dtype)
        head = create_task_head_graph(config, optimizer, embeddings, decoder_block.fwd.graph.outputs[0],
                                      input_streams.labels.spec, tied_weight_spec, tied_weight_spec, word_offset.spec)

        # ---- Transform graphs ----

        # Recomputation
        embeddings.bwd = addons.recompute_graph(embeddings.bwd)
        decoder_block.bwd = addons.recompute_graph(decoder_block.bwd)

        # Batch Serialisation
        #   Buffers
        x_buffer = batch_serial_buffer(embeddings.fwd.graph.outputs[0],
                                       steps=config.gradient_accumulation,
                                       rows=config.model.layers + 1)
        dx_buffer = batch_serial_buffer(embeddings.bwd.graph.inputs[0],
                                        steps=config.gradient_accumulation,
                                        rows=config.model.layers + 1)

        #   Graphs
        embeddings_batch_serialise(config, embeddings, input_streams, x_buffer, dx_buffer)
        decoder_block_batch_serialise(config, decoder_block, x_buffer, dx_buffer)
        head.fwd = head_batch_serialise(config, head.fwd, input_streams, output_streams, x_buffer, dx_buffer)

        # Available Memory Proportion

        addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)
        # set specific phases
        # memory_proportion = (0.05, )
        # set_graph_available_memory_proportion_by_ipu(head.fwd.graph,
        #                                          memory_proportion)

        # ----- Create Variables -----

        variables = NamedTensors()
        variables.insert("embeddings", embeddings.facts.init_remote(embeddings.buffers, 0, "embeddings"))
        variables.insert(
            "layer",
            NamedTensors.from_dict({
                n: decoder_block.facts.init_remote(decoder_block.buffers, n, f"layer.{n}")
                for n in range(config.model.layers)
            }))
        variables.insert("head", head.facts.init_remote(head.buffers, 0, "head"))

        # ---- Execute ----

        with popxl.in_sequence():
            # Load current learning rate
            with popxl.transforms.merge_exchange(), popxl.in_sequence(False):
                seed = ops.host_load(input_streams.seed)
                lr = ops.host_load(input_streams.lr)

            def embedding_fwd_phase(seed, positions):
                # Load Embedding layer
                embeddings_vars = embeddings.fwd_load(0)
                embeddings_vars = embeddings.fwd_all_gather(embeddings_vars)
                # Forward
                seed, embed_seed = ops.split_random_seed(seed)
                embeddings.fwd.bind(embeddings_vars).call(0, embed_seed, positions)
                return seed

            embed_fwd_graph = ir.create_graph(embedding_fwd_phase, seed, positions)
            if config.execution.code_load:
                ops.remote_code_load(embed_fwd_graph, "executable")
            seed, = ops.call(embed_fwd_graph, seed, positions)

            def single_decoder_block_fwd_phase(n: popxl.Tensor, seed: popxl.Tensor):
                # Load decoder block
                layer_vars = decoder_block.fwd_load(n)
                layer_vars = decoder_block.fwd_all_gather(layer_vars)
                # Forward
                seed, layer_seed = ops.split_random_seed(seed)
                decoder_block.fwd.bind(layer_vars).call(n, layer_seed)
                return n + 1, seed

            i = popxl.constant(0, name="layer_index")
            fwd_graph = ir.create_graph(single_decoder_block_fwd_phase, i, seed)
            if config.execution.code_load:
                ops.remote_code_load(fwd_graph, "executable")
            ops.repeat(fwd_graph, config.model.layers, i, seed)

            # Buffer to be used later
            tied_weight_grad_buffer = None

            def task_head_fwd_grad_phase(word_offset):
                nonlocal tied_weight_grad_buffer
                # Load task head layer
                head_vars = head.fwd_load(0)
                head_vars = NamedTensors(fwd=head.fwd_all_gather(head_vars), bwd=head.grad_facts.init_zero())
                # Tied weight
                tied_weight_t = ops.transpose_(head_vars.fwd.pop('weight'))
                tied_weight_grad_t = ops.init(tied_weight_t.shape, tied_weight_t.dtype, 'word_embedding_grad_t', "zero")
                # Forward + Gradient
                head.fwd.bind(head_vars).call(0, tied_weight_t, tied_weight_grad_t, word_offset)
                # Data parallel reduce
                reduced_grads = head.grad_reduce(head_vars.bwd)

                # Global Norm calculation
                grad_norm = ops.init((), popxl.float32, name='grad_norm', init_type='zero')
                global_norm_reduce(config, grad_norm, reduced_grads)
                # Store Gradients
                head.grad_store(reduced_grads, 0)

                # Reduce and Store the tied gradient
                grad_t = reduce_replica_sharded_tensor(ops.transpose_(tied_weight_grad_t),
                                                       'mean',
                                                       threshold=RTS_THRESHOLD,
                                                       replica_group=dp_group,
                                                       shard_group=get_instance_replica_grouping(dp_group))

                tied_weight_grad_buffer = create_remote_buffer(grad_t.spec,
                                                               sharded_threshold=RTS_THRESHOLD,
                                                               replica_group=dp_group,
                                                               shard_group=get_instance_replica_grouping(dp_group))

                ops.remote_store(tied_weight_grad_buffer, 0, grad_t)
                return grad_norm

            task_graph = ir.create_graph(task_head_fwd_grad_phase, word_offset)
            if config.execution.code_load:
                ops.remote_code_load(task_graph, "executable")
            grad_norm, = ops.call(task_graph, word_offset)

            def single_decoder_block_grad_phase(n: popxl.Tensor, grad_norm: popxl.TensorByRef):
                # Load layer
                layer_vars = decoder_block.fwd_load(n)
                layer_vars = decoder_block.fwd_all_gather(layer_vars)
                # Gradient
                grads = decoder_block.grad_facts.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(layer_vars)
                decoder_block.bwd.bind(bwd_vars).call(n)
                # Data parallel reduce
                reduced_grads = decoder_block.grad_reduce(grads)
                # Global Norm calculation
                global_norm_reduce(config, grad_norm, reduced_grads)
                # Store gradient
                decoder_block.grad_store(reduced_grads, n)
                return n - 1

            i = popxl.constant(config.model.layers - 1, name="layer_index")
            bwd_graph = ir.create_graph(single_decoder_block_grad_phase, i, grad_norm)
            if config.execution.code_load:
                ops.remote_code_load(bwd_graph, "executable")
            ops.repeat(bwd_graph, config.model.layers, i, grad_norm)

            def embedding_grad_optimizer_phase(lr: popxl.Tensor, grad_norm: popxl.TensorByRef):
                nonlocal tied_weight_grad_buffer
                # Load Embeddings layer
                embeddings_vars = embeddings.optim_fwd_load(0)
                embeddings_fwd_vars = embeddings.fwd_all_gather(embeddings_vars.fwd)
                # Gradient
                grads = embeddings.grad_facts.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(embeddings_fwd_vars)
                embeddings.bwd.bind(bwd_vars).call(0)
                # Data parallel reduce
                reduced_grads = embeddings.grad_reduce(grads)

                # Add the tied gradient from the projection
                tied_weight_grad = ops.remote_load(tied_weight_grad_buffer, 0)
                ops.add_(reduced_grads.accum.word.weight, tied_weight_grad)

                # Global Norm calculation
                global_norm_reduce(config, grad_norm, reduced_grads)
                # Finalise global bwd norm with an all reduce and sqrt
                grad_norm = ops.sqrt(ops.collectives.replicated_all_reduce(grad_norm, op='add'))
                ops.host_store(output_streams.grad_norm, grad_norm)

                # Optimizer Step for Embeddings.
                # Note: No need to store then load the gradient.. just use it directly
                embeddings_vars.insert("bwd", reduced_grads)
                optimizer_step(embeddings.optim, embeddings_vars, lr, grad_norm)
                # Store
                embeddings.optim_fwd_store(embeddings_vars, 0)
                return grad_norm

            embed_bwd_graph = ir.create_graph(embedding_grad_optimizer_phase, lr, grad_norm)
            if config.execution.code_load:
                ops.remote_code_load(embed_bwd_graph, "executable")
            grad_norm, = ops.call(embed_bwd_graph, lr, grad_norm)

            # Optimizer Step for Layers
            def layer_optim(n: popxl.Tensor, lr: popxl.Tensor, grad_norm: popxl.Tensor):
                layer_vars = decoder_block.optim_fwd_load(n)
                optimizer_step(decoder_block.optim, layer_vars, lr, grad_norm)
                decoder_block.optim_fwd_store(layer_vars, n)
                return n + 1

            i = popxl.constant(0, name="layer_index")
            layer_optim_graph = ir.create_graph(layer_optim, i, lr, grad_norm)
            if config.execution.code_load:
                ops.remote_code_load(layer_optim_graph, "executable")
            ops.repeat(layer_optim_graph, config.model.layers, i, lr, grad_norm)

            def head_optim(lr: popxl.Tensor, grad_norm: popxl.Tensor):
                # Optimizer Step for Task Head - Only layer norm, tied weights handled by embedding
                head_vars = head.optim_fwd_load(0)
                task_head_optimizer_step(head.optim, head_vars, lr, grad_norm)
                # Store
                head.optim_fwd_store(head_vars, 0)

            head_optim_graph = ir.create_graph(head_optim, lr, grad_norm)
            if config.execution.code_load:
                ops.remote_code_load(head_optim_graph, "executable")
            ops.call(head_optim_graph, lr, grad_norm)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')

    repeat_graph(main, config.execution.device_iterations)

    fwd_vars = NamedTensors(embeddings=variables.embeddings.fwd,
                            layer=NamedTensors.from_dict(
                                {i: variables.layer[i].fwd
                                 for i in range(config.model.layers)}),
                            head=variables.head.fwd)

    logging.info(f"popxl IR construction duration: {(time.time() - t) / 60:.2f} mins")

    ir.num_host_transfers = config.execution.device_iterations * \
        config.gradient_accumulation

    session = TaskSession(input_streams, output_streams, fwd_vars, ir, "ipu_hw")

    return session


def main():
    """Run a benchmark configuration"""
    config, _ = gpt_config_setup(CONFIG_DIR / "pretraining.yml", "release", "gpt3_2.7B_pod64")
    session = pretraining(config)
    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    with session:
        # Skip one result
        session.run(inputs)

        durations = []
        for _ in range(5):
            start = time.perf_counter()
            session.run(inputs)
            durations.append(time.perf_counter() - start)
    duration = np.mean(durations)

    samples_per_step = config.execution.device_iterations * \
        config.training.global_batch_size
    result_str = \
        f"Duration: {duration} s " \
        f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    main()
