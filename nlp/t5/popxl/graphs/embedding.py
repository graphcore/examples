# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popxl
import popxl_addons as addons
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, get_ild_replica_grouping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.batch_serialisation import (
    batch_serialise_fwd_and_grad,
    RemoteHandle,
)
from popxl_addons.rts import (
    all_gather_replica_sharded_graph,
    reduce_replica_sharded_graph,
)
from popxl_addons.remote import (
    named_variable_buffers,
    load_remote_graph,
    store_remote_graph,
)

from config import T5Config
from modelling.embedding import T5EmbeddingsTP, T5DecoderEmbeddingsTP
from graphs.graphs import (
    Graphs,
    optimizer_graphs,
    get_rts_groups,
    use_io_tiles,
)


def create_embeddings_graph(config: T5Config, optimizer: addons.Module, *args, **kwargs):
    embeddings = Graphs()

    # Create Graphs for computing forward, gradient and optimizer
    fwd_facts, embeddings.fwd = T5EmbeddingsTP(config).create_graph(*args, **kwargs)
    # where the variables are equal. If a variable has None as rg, it is assumed equal on all replicas.
    dp_group = popxl.gcg().ir.replica_grouping(
        stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
    )

    # Embedding needs no onward gradients
    required_grads = ()
    # Exclude the rel_pos_weight from autodiff
    accums = [t for t in embeddings.fwd.args.tensors if "rel_pos_weight" not in t.name]
    grad_facts, embeddings.bwd = addons.autodiff_with_accumulation(
        embeddings.fwd,
        accums,
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


def create_decoder_embeddings_graph(
    config: T5Config, optimizer: addons.Module, encoder_embeddings: Graphs, *args, **kwargs
):
    embeddings = Graphs()

    # Create Graphs for computing forward, gradient and optimizer
    fwd_facts, embeddings.fwd = T5DecoderEmbeddingsTP(config).create_graph(*args, **kwargs)
    # where the variables are equal. If a variable has None as rg, it is assumed equal on all replicas.
    dp_group = popxl.gcg().ir.replica_grouping(
        stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
    )

    # Create the grad accumulator for the embedding weight, but exclude the rel_pos_weight
    accums = [t for t in embeddings.fwd.args.tensors if "rel_pos_weight" not in t.name]
    accums += [embeddings.fwd.graph.inputs[1]]
    replica_groupings = fwd_facts.replica_groupings
    replica_groupings.insert("word_embedding", dp_group)
    # Embedding needs no onward gradients
    required_grads = ()
    grad_facts, embeddings.bwd = addons.autodiff_with_accumulation(
        embeddings.fwd,
        accums,
        grads_required=required_grads,
        replica_groupings=replica_groupings,
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
    # Remove the embedding weight from the grad accumulators: we'll create it elsewhere
    grad_facts.accum.pop("word_embedding")
    embeddings.grad_facts = grad_facts

    # Create remote buffers using only forward facts
    rts_fwd_optim_groups = get_rts_groups(embeddings.facts)
    shard_over = {k: rg.group_size for k, rg in rts_fwd_optim_groups.to_dict().items()}
    embeddings.buffers = named_variable_buffers(embeddings.facts, shard_over_dict=shard_over)

    # Create Graphs for loading/gathering/storing/reducing
    # Create the optim fwd store before adding the embedding weight to the buffers
    # (we don't want to store the embedding weight after the optimiser step)
    embeddings._optim_fwd_store = store_remote_graph(embeddings.buffers)
    # Then add the embedding weight buffer to the buffers for the fwd load and the optim fwd load
    embeddings.buffers.fwd.insert("weight", encoder_embeddings.buffers.fwd.word.weight)
    rts_fwd_optim_groups.fwd.insert(
        "weight", get_ild_replica_grouping(encoder_embeddings.facts.fwd.word.weight.replica_grouping)
    )
    embeddings._optim_fwd_load, embeddings._optim_fwd_load_names = load_remote_graph(embeddings.buffers)
    embeddings._fwd_load, embeddings._fwd_load_names = load_remote_graph(embeddings.buffers.fwd)

    embeddings._fwd_all_gather, embeddings._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(embeddings._fwd_load_names, embeddings._fwd_load.graph.outputs),
        replica_groups=rts_fwd_optim_groups.fwd,
        use_io_tiles=use_io_tiles,
    )
    grad_accums = embeddings.bwd.args.copy()
    grad_accums.pop("mean_accum_counter")
    # the embedding weight is handled elsewhere
    grad_accums.accum.pop("word_embedding")
    rts_bwd_group = NamedReplicaGrouping(accum=rts_fwd_optim_groups.fwd.copy())
    embeddings._grad_reduce, embeddings._grad_reduce_names = reduce_replica_sharded_graph(
        grad_accums, "mean", shard_groups=rts_bwd_group, replica_group=dp_group, use_io_tiles=use_io_tiles
    )
    return embeddings


def embeddings_batch_serialise(
    config: T5Config,
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


def decoder_embeddings_batch_serialise(
    config: T5Config,
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
            embeddings.fwd.graph.inputs[0]: input_streams.decoder_words,
            embeddings.bwd.graph.inputs[0]: RemoteHandle(dx_buffer, config.model.layers + 2, dx_shard_group),
        },
        store_streams={},
        store_buffers={
            embeddings.fwd.graph.outputs[0]: RemoteHandle(x_buffer, config.model.layers + 2, x_shard_group),
        },
        seed_input=embeddings.fwd.graph.inputs[2],
        rows=1,
        io_mode="io",
    )
    embeddings.fwd = fwd.graph
    embeddings.bwd = bwd.graph
