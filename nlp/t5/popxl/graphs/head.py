# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import popxl
import popxl_addons as addons
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.batch_serialisation import (
    batch_serialise,
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
from modelling.t5_lm import T5LMHeadLossAndGradTP
from graphs.graphs import (
    Graphs,
    optimizer_graphs,
    get_rts_groups,
    use_io_tiles,
)


def create_task_head_graph(config: T5Config, optimizer: addons.Module, *args, **kwargs):
    """Combines the LM forward (which includes an initial layer norm, normally at the end of the T5 decoder stack),
    loss and bwd into a single Module."""
    head = Graphs()

    facts, graph = T5LMHeadLossAndGradTP(config).create_graph(*args, **kwargs)

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


def head_batch_serialise(
    config: T5Config,
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
            head_graph.graph.inputs[0]: RemoteHandle(x_buffer, 2 * config.model.layers + 2, x_shard_group),
            head_graph.graph.inputs[1]: input_streams.labels,
        },
        store_streams={head_graph.graph.outputs[0]: output_streams.loss},
        store_buffers={
            head_graph.graph.outputs[1]: RemoteHandle(dx_buffer, 2 * config.model.layers + 2, dx_shard_group)
        },
        seed_input=head_graph.graph.inputs[2],
        io_mode="io",
    )
    return bs_head.graph
