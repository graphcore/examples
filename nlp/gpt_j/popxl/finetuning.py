# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import logging
import time
import numpy as np
from typing import Dict, List, Union

import popdist
import popxl
from popxl import ops
import popxl_addons as addons
from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons import TaskSession
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.utils import timer

from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, get_ild_replica_grouping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.transforms.batch_serialisation import (
    batch_serialise_fwd_and_grad,
    batch_serial_buffer,
    batch_serialise,
    RemoteHandle,
)
from popxl_addons.rts import all_gather_replica_sharded_graph, replica_sharded_spec, reduce_replica_sharded_graph
from popxl_addons.remote import named_variable_buffers, load_remote_graph, store_remote_graph, NamedRemoteBuffers
from popxl_addons.ops.grad_reduce_square_add import grad_reduce_square_add

from config import GPTJConfig, CONFIG_DIR
from utils.setup import gptj_config_setup
from modelling.embedding import GPTJEmbeddingsTP
from modelling.decoder import GPTJDecoderBlockTP
from modelling.gptj_lm import GPTJLMHeadLossAndGradTP

__all__ = ["finetuning"]

OptimGraphs = Dict[str, GraphWithNamedArgs]
RTS_THRESHOLD = 0
RTS_ACTIVATIONS_THRESHOLD = 0
use_io_tiles = False


def get_activ_shard_group(a: popxl.Tensor, shard_group: popxl.ReplicaGrouping):
    return shard_group if a.nelms >= RTS_ACTIVATIONS_THRESHOLD else popxl.gcg().ir.replica_grouping(group_size=1)


def get_rts_groups(facts: NamedVariableFactories) -> NamedReplicaGrouping:
    ir = popxl.gcg().ir

    rts_groups = {}
    for k, f in facts.to_dict().items():
        size = np.prod(f.shape)
        rg = f.replica_grouping
        # Limit RTS to within an ILD
        rg = get_ild_replica_grouping(rg)
        if size % rg.group_size == 0 and size >= RTS_THRESHOLD:
            rts_groups[k] = rg
        else:
            rts_groups[k] = ir.replica_grouping(group_size=1)
    return NamedReplicaGrouping.from_dict(rts_groups)


def requires_weight_decay(t: popxl.Tensor):
    return not any(map(lambda exclude: exclude in t.name, ["norm", "bias"]))


def optimizer_graphs(
    config: GPTJConfig,
    optimizer: addons.Module,
    variables: NamedTensors,
    replica_groups: NamedReplicaGrouping,
    shard_groups: NamedReplicaGrouping,
):
    optim_facts = {}
    optim_graphs = {}
    replica_groups = replica_groups.to_dict()
    shard_groups = shard_groups.to_dict()
    for name, var in variables.to_dict().items():
        # Currently assumes grads have the same replica group as their var
        input_spec = replica_sharded_spec(var, shard_over=shard_groups[name])
        optim_facts[name], optim_graphs[name] = optimizer.create_graph(
            input_spec,
            input_spec,
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
            global_norm_max=config.training.optimizer.gradient_clipping,
        )
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


def create_embeddings_graph(config: GPTJConfig, optimizer: addons.Module, *args, **kwargs):
    embeddings = Graphs()

    # Create Graphs for computing forward, gradient and optimizer
    fwd_facts, embeddings.fwd = GPTJEmbeddingsTP(config).create_graph(*args, **kwargs)

    # where the variables are equal. If a variable has None as rg, it is assumed equal on all replicas.
    dp_group = popxl.gcg().ir.replica_grouping(
        stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
    )

    # Embedding needs no onward gradients
    required_grads = ()
    grad_facts, embeddings.bwd = addons.autodiff_with_accumulation(
        embeddings.fwd,
        embeddings.fwd.args.tensors,
        grads_required=required_grads,
        replica_groupings=fwd_facts.replica_groupings,
    )

    optim_facts, embeddings.optim = optimizer_graphs(
        config,
        optimizer,
        embeddings.fwd.args,
        replica_groups=fwd_facts.replica_groupings,
        shard_groups=get_rts_groups(fwd_facts),
    )
    # Variables required
    embeddings.facts = NamedVariableFactories(fwd=fwd_facts, optim=optim_facts)
    embeddings.grad_facts = grad_facts

    # Create remote buffers using only forward facts
    # Embedding optimizer step happens straight after the bwd: no need to store the gradient in a buffer.
    rts_fwd_optim_groups = get_rts_groups(embeddings.facts)
    shard_over = {k: rg.group_size for k, rg in rts_fwd_optim_groups.to_dict().items()}
    embeddings.buffers = named_variable_buffers(embeddings.facts, shard_over_dict=shard_over)

    # Create Graphs for loading/gathering/storing/reducing
    embeddings._optim_fwd_load, embeddings._optim_fwd_load_names = load_remote_graph(embeddings.buffers)
    embeddings._optim_fwd_store = store_remote_graph(embeddings.buffers)
    embeddings._fwd_load, embeddings._fwd_load_names = load_remote_graph(embeddings.buffers.fwd)

    embeddings._fwd_all_gather, embeddings._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(embeddings._fwd_load_names, embeddings._fwd_load.graph.outputs),
        replica_groups=rts_fwd_optim_groups.fwd,
        use_io_tiles=use_io_tiles,
    )

    grad_accums = embeddings.bwd.args.copy()
    grad_accums.pop("mean_accum_counter")
    rts_bwd_group = NamedReplicaGrouping(accum=rts_fwd_optim_groups.fwd.copy())
    embeddings._grad_reduce, embeddings._grad_reduce_names = reduce_replica_sharded_graph(
        grad_accums, "mean", shard_groups=rts_bwd_group, replica_group=dp_group, use_io_tiles=use_io_tiles
    )
    return embeddings


def create_decoder_block_graph(config: GPTJConfig, optimizer: addons.Module, *args, **kwargs):
    layer = Graphs()

    # Create Graphs for computing forward, gradient and optimizer
    fwd_facts, layer.fwd = GPTJDecoderBlockTP(config).create_graph(*args, **kwargs)
    required_grads = (layer.fwd.graph.inputs[0],)

    dp_group = popxl.gcg().ir.replica_grouping(
        stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
    )

    called_graphs_grad_info = {}
    if config.execution.attention_serialisation > 1:
        # Optimisation to recompute each blk separately
        assert len(layer.fwd.graph.called_graphs) == 1, "expected exactly 1 called graph by decoder layer fwd"
        blk_graph = GraphWithNamedArgs(layer.fwd.graph.called_graphs[0])
        grad_blk_graph = addons.transforms.autodiff(blk_graph, grads_required=blk_graph.graph.inputs[:-2])
        grad_blk_graph = addons.transforms.recompute_graph(grad_blk_graph)
        called_graphs_grad_info[blk_graph.graph] = grad_blk_graph.grad_graph_info

    grad_facts, layer.bwd = addons.autodiff_with_accumulation(
        layer.fwd,
        layer.fwd.args.tensors,
        grads_required=required_grads,
        called_graphs_grad_info=called_graphs_grad_info,
        replica_groupings=fwd_facts.replica_groupings,
    )

    popxl.transforms.decompose_sum(layer.bwd.graph)

    optim_args, layer.optim = optimizer_graphs(
        config,
        optimizer,
        layer.fwd.args,
        replica_groups=fwd_facts.replica_groupings,
        shard_groups=get_rts_groups(fwd_facts),
    )

    # Variables required
    layer.facts = NamedVariableFactories(fwd=fwd_facts, optim=optim_args)
    layer.grad_facts = grad_facts

    # Create remote buffers
    entries = config.model.layers
    buffer_facts = layer.facts.copy()
    buffer_facts.insert("bwd", grad_facts.copy())
    buffer_facts.bwd.pop("mean_accum_counter")

    rts_fwd_bwd_groups = get_rts_groups(buffer_facts)
    shard_over = {k: rg.group_size for k, rg in rts_fwd_bwd_groups.to_dict().items()}
    layer.buffers = named_variable_buffers(buffer_facts, entries, shard_over_dict=shard_over)

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
        NamedTensors.pack(layer._fwd_load_names, layer._fwd_load.graph.outputs),
        replica_groups=rts_fwd_bwd_groups.fwd,
        use_io_tiles=use_io_tiles,
    )

    grad_accums = layer.bwd.args.copy()
    grad_accums.pop("mean_accum_counter")

    layer._grad_reduce, layer._grad_reduce_names = reduce_replica_sharded_graph(
        grad_accums, "mean", shard_groups=rts_fwd_bwd_groups.bwd, replica_group=dp_group, use_io_tiles=use_io_tiles
    )

    return layer


def create_task_head_graph(config: GPTJConfig, optimizer: addons.Module, *args, **kwargs):
    """Combines the LM forward (which includes an initial layer norm, normally at the end of the gpt decoder stack),
    loss and bwd into a single Module."""
    head = Graphs()

    facts, graph = GPTJLMHeadLossAndGradTP(config).create_graph(*args, **kwargs)

    dp_group = popxl.gcg().ir.replica_grouping(
        stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
    )

    optim_ts = graph.args.fwd.copy()
    optim_facts, optim_graphs = optimizer_graphs(
        config, optimizer, optim_ts, replica_groups=facts.fwd.replica_groupings, shard_groups=get_rts_groups(facts.fwd)
    )

    facts.insert("optim", optim_facts)
    head.fwd = graph
    head.optim = optim_graphs
    head.facts = facts
    head.grad_facts = facts.pop("bwd")

    # Create remote buffers
    buffer_facts = head.facts.copy()
    buffer_facts.insert("bwd", head.grad_facts.copy())

    rts_fwd_bwd_groups = get_rts_groups(buffer_facts)
    shard_over = {k: rg.group_size for k, rg in rts_fwd_bwd_groups.to_dict().items()}
    head.buffers = named_variable_buffers(buffer_facts, shard_over_dict=shard_over)

    # Create Graphs for loading/gathering/storing/reducing
    head._optim_fwd_load, head._optim_fwd_load_names = load_remote_graph(head.buffers)

    buffers = head.buffers.copy()
    buffers_bwd = buffers.pop("bwd")

    # Store fwd and optim
    head._optim_fwd_store = store_remote_graph(head.buffers)
    # Store bwd
    head._grad_store = store_remote_graph(buffers_bwd)

    head._fwd_load, head._fwd_load_names = load_remote_graph(head.buffers.fwd)
    head._fwd_all_gather, head._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(head._fwd_load_names, head._fwd_load.graph.outputs),
        use_io_tiles=use_io_tiles,
        replica_groups=rts_fwd_bwd_groups.fwd,
    )

    head._grad_reduce, head._grad_reduce_names = reduce_replica_sharded_graph(
        graph.args.bwd, "mean", shard_groups=rts_fwd_bwd_groups.bwd, replica_group=dp_group, use_io_tiles=use_io_tiles
    )
    return head


def embeddings_batch_serialise(
    config: GPTJConfig,
    embeddings: Graphs,
    input_streams: addons.InputStreams,
    x_buffer: popxl.RemoteBuffer,
    dx_buffer: popxl.RemoteBuffer,
):
    tp = config.execution.tensor_parallel
    tp_group = popxl.gcg().ir.replica_grouping(stride=1, group_size=tp)
    x_shard_group = tp_group if x_buffer.meta_shape else popxl.gcg().ir.replica_grouping(group_size=1)
    dx_shard_group = tp_group if dx_buffer.meta_shape else popxl.gcg().ir.replica_grouping(group_size=1)

    fwd, bwd = batch_serialise_fwd_and_grad(
        embeddings.fwd,
        embeddings.bwd,
        embeddings.fwd.args,
        config.gradient_accumulation,
        load_handles={
            embeddings.fwd.graph.inputs[0]: input_streams.words,
            embeddings.bwd.graph.inputs[0]: RemoteHandle(dx_buffer, 0, dx_shard_group),
        },
        store_streams={},
        store_buffers={embeddings.fwd.graph.outputs[0]: RemoteHandle(x_buffer, 0, x_shard_group)},
        seed_input=embeddings.fwd.graph.inputs[1],
        rows=1,
        io_mode="io",
    )
    embeddings.fwd = fwd.graph
    embeddings.bwd = bwd.graph


def decoder_block_batch_serialise(
    config: GPTJConfig, layer: Graphs, x_buffer: popxl.RemoteBuffer, dx_buffer: popxl.RemoteBuffer
):
    tp = config.execution.tensor_parallel
    tp_group = popxl.gcg().ir.replica_grouping(stride=1, group_size=tp)
    x_shard_group = tp_group if x_buffer.meta_shape else popxl.gcg().ir.replica_grouping(group_size=1)
    dx_shard_group = tp_group if dx_buffer.meta_shape else popxl.gcg().ir.replica_grouping(group_size=1)

    fwd, bwd = batch_serialise_fwd_and_grad(
        layer.fwd,
        layer.bwd,
        layer.fwd.args,
        config.gradient_accumulation,
        load_handles={
            layer.fwd.graph.inputs[0]: RemoteHandle(x_buffer, 0, x_shard_group),
            layer.bwd.graph.inputs[0]: RemoteHandle(dx_buffer, 1, dx_shard_group),
        },
        store_streams={},
        store_buffers={
            layer.fwd.graph.outputs[0]: RemoteHandle(x_buffer, 1, x_shard_group),
            layer.bwd.graph.outputs[0]: RemoteHandle(dx_buffer, 0, dx_shard_group),
        },
        seed_input=layer.fwd.graph.inputs[1],
        rows=config.model.layers,
        io_mode="io",
    )
    layer.fwd = fwd.graph
    layer.bwd = bwd.graph


def head_batch_serialise(
    config: GPTJConfig,
    head_graph: GraphWithNamedArgs,
    input_streams: addons.InputStreams,
    output_streams: addons.OutputStreams,
    x_buffer: popxl.RemoteBuffer,
    dx_buffer: popxl.RemoteBuffer,
) -> GraphWithNamedArgs:
    tp = config.execution.tensor_parallel
    tp_group = popxl.gcg().ir.replica_grouping(stride=1, group_size=tp)
    x_shard_group = tp_group if x_buffer.meta_shape else popxl.gcg().ir.replica_grouping(group_size=1)
    dx_shard_group = tp_group if dx_buffer.meta_shape else popxl.gcg().ir.replica_grouping(group_size=1)

    bs_head = batch_serialise(
        head_graph,
        config.gradient_accumulation,
        load_handles={
            head_graph.graph.inputs[0]: RemoteHandle(x_buffer, config.model.layers, x_shard_group),
            head_graph.graph.inputs[1]: input_streams.labels,
        },
        store_streams={head_graph.graph.outputs[0]: output_streams.loss},
        store_buffers={head_graph.graph.outputs[1]: RemoteHandle(dx_buffer, config.model.layers, dx_shard_group)},
        io_mode="io",
    )
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


def global_norm_reduce(config: GPTJConfig, grad_norm: popxl.Tensor, grads: NamedTensors):
    for g in grads.tensors:
        ops.add_(grad_norm, grad_reduce_square_add(g, config.execution.loss_scaling))


def finetuning(config: GPTJConfig, no_init: bool = True) -> TaskSession:
    replicas = config.execution.data_parallel * config.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas
    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"

    with timer("PopXL IR construction"):
        main = ir.main_graph
        tp_group = ir.replica_grouping(stride=1, group_size=config.execution.tensor_parallel)
        with main:
            # ----- Define input and output streams -----
            input_shape = (config.execution.micro_batch_size * config.model.sequence_length,)
            input_streams = addons.InputStreams(
                words=(input_shape, popxl.int32), labels=(input_shape, popxl.int32), lr=((), popxl.float32)
            )
            output_streams = addons.OutputStreams(loss=((), config.model.dtype), grad_norm=((), popxl.float32))

            # ---- Initialise Random Seed ----
            seed_v, seed = addons.seed_variable(config.model.seed, tp_group)

            # ----- Build compute graphs -----
            optimizer = AdamOptimizerStep()

            embeddings = create_embeddings_graph(config, optimizer, input_streams.words.spec, seed=seed.spec)

            decoder_block = create_decoder_block_graph(
                config, optimizer, embeddings.fwd.graph.outputs[0], seed=seed.spec
            )

            head = create_task_head_graph(
                config, optimizer, decoder_block.fwd.graph.outputs[0], input_streams.labels.spec
            )

            # ---- Transform graphs ----

            # Recomputation
            embeddings.bwd = addons.recompute_graph(embeddings.bwd)
            decoder_block.bwd = addons.recompute_graph(decoder_block.bwd)

            # Batch Serialisation
            #   Buffers
            x_buffer = batch_serial_buffer(
                embeddings.fwd.graph.outputs[0],
                steps=config.gradient_accumulation,
                rows=config.model.layers + 1,
                shard_group=get_activ_shard_group(embeddings.fwd.graph.outputs[0], tp_group),
            )
            dx_buffer = batch_serial_buffer(
                embeddings.bwd.graph.inputs[0],
                steps=config.gradient_accumulation,
                rows=config.model.layers + 1,
                shard_group=get_activ_shard_group(embeddings.bwd.graph.inputs[0], tp_group),
            )

            #   Graphs
            embeddings_batch_serialise(config, embeddings, input_streams, x_buffer, dx_buffer)
            decoder_block_batch_serialise(config, decoder_block, x_buffer, dx_buffer)
            head.fwd = head_batch_serialise(config, head.fwd, input_streams, output_streams, x_buffer, dx_buffer)

            # Available Memory Proportion
            addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

            # ----- Create Variables -----
            variables = NamedTensors(random_seed=seed_v)
            transformer = NamedTensors()
            variables.insert("transformer", transformer)
            for key in ("fwd", "bwd", "optim"):
                empty = no_init and key == "fwd"
                if key in embeddings.facts.keys():
                    transformer.insert(
                        f"embeddings.{key}",
                        embeddings.facts[key].init_remote(embeddings.buffers[key], 0, f"embeddings.{key}", empty=empty),
                        overwrite=True,
                    )
                if key in decoder_block.facts.keys():
                    for n in range(config.model.layers):
                        transformer.insert(
                            f"decoder.{n}.{key}",
                            decoder_block.facts[key].init_remote(
                                decoder_block.buffers[key], n, f"decoder.{n}.{key}", empty=empty
                            ),
                            overwrite=True,
                        )
                if key in head.facts.keys():
                    variables.insert(
                        f"lm_head.{key}",
                        head.facts[key].init_remote(head.buffers[key], 0, "lm_head", empty=empty),
                        overwrite=True,
                    )

            # ---- Execute ----

            with popxl.in_sequence():
                # Increment random seed
                seed += 1
                # Load current learning rate
                lr = ops.host_load(input_streams.lr)
                # Increment random seed
                seed += 1

                def embedding_fwd_phase(seed):
                    # Load Embedding layer
                    embeddings_vars = embeddings.fwd_load(0)
                    embeddings_vars = embeddings.fwd_all_gather(embeddings_vars)
                    # Forward
                    seed, embed_seed = ops.split_random_seed(seed)
                    embeddings.fwd.bind(embeddings_vars).call(0, embed_seed)
                    return seed

                embed_fwd_graph = ir.create_graph(embedding_fwd_phase, seed)
                if config.execution.code_load:
                    ops.remote_code_load(embed_fwd_graph, "executable")
                (seed,) = ops.call(embed_fwd_graph, seed)

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

                def task_head_fwd_grad_phase():
                    # Load task head layer
                    head_vars = head.fwd_load(0)
                    head_vars = NamedTensors(fwd=head.fwd_all_gather(head_vars), bwd=head.grad_facts.init_zero())
                    # Forward + Gradient
                    head.fwd.bind(head_vars).call(0)
                    # Data parallel reduce
                    reduced_grads = head.grad_reduce(head_vars.bwd)
                    # Global Norm calculation
                    grad_norm = ops.init((), popxl.float32, name="grad_norm", init_type="zero")
                    global_norm_reduce(config, grad_norm, reduced_grads)
                    # Store Gradients
                    head.grad_store(reduced_grads, 0)
                    return grad_norm

                task_graph = ir.create_graph(task_head_fwd_grad_phase)
                if config.execution.code_load:
                    ops.remote_code_load(task_graph, "executable")
                (grad_norm,) = ops.call(task_graph)

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
                    # Global Norm calculation
                    global_norm_reduce(config, grad_norm, reduced_grads)
                    # Finalise global bwd norm with an all reduce and sqrt
                    grad_norm = ops.sqrt(ops.collectives.replicated_all_reduce(grad_norm, op="add"))
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
                (grad_norm,) = ops.call(embed_bwd_graph, lr, grad_norm)

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
        apply_pre_alias_patterns(ir, level="default")

        repeat_graph(main, config.execution.device_iterations)

        fwd_vars = NamedTensors.from_dict(
            {
                "transformer.embeddings": variables.transformer.embeddings.fwd,
                "transformer.decoder": NamedTensors.from_dict(
                    {i: variables.transformer.decoder[i].fwd for i in range(config.model.layers)}
                ),
                "lm_head": variables.lm_head.fwd,
            }
        )

        optim_vars = NamedTensors.from_dict(
            {
                "transformer.embeddings": variables.transformer.embeddings.optim,
                "transformer.decoder": NamedTensors.from_dict(
                    {i: variables.transformer.decoder[i].optim for i in range(config.model.layers)}
                ),
                "lm_head": variables.lm_head.optim,
            }
        )

        if config.checkpoint.optim_state:
            state = NamedTensors(fwd=fwd_vars, optim=optim_vars)
        else:
            state = NamedTensors(fwd=fwd_vars)

    ir.num_host_transfers = config.execution.device_iterations * config.gradient_accumulation

    session = TaskSession(
        inputs=input_streams,
        outputs=output_streams,
        state=state,
        max_checkpoints=config.checkpoint.to_keep,
        ir=ir,
        device_desc="ipu_hw",
    )

    return session


def main():
    """Run a benchmark configuration"""
    config, *_ = gptj_config_setup(
        CONFIG_DIR / "finetuning.yml", "release", "gptj_6B_1024_pod64", wandb_setup=False, hf_model_setup=False
    )
    session = finetuning(config)
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

    samples_per_step = config.execution.device_iterations * config.training.global_batch_size
    result_str = f"Duration: {duration} s " f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
