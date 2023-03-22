# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
import time

import numpy as np
from typing import Dict, Union, Type, Optional

import popdist
import popxl
from popxl import ops, gcg
import popxl_addons as addons
from pretraining_config import (
    RTS_THRESHOLD,
    RTS_ACTIVATIONS_THRESHOLD,
    USE_IO_TILES,
    GraphConf,
    PhaseConf,
    gen_layer_config,
    filter,
    RTS_ACT,
)
from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons import TaskSession
from popxl_addons.utils import OrderedDict, timer
from popxl_addons.patterns import apply_pre_alias_patterns

from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_replica_grouping import NamedReplicaGrouping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.transforms.batch_serialisation import (
    batch_serialise_fwd_and_grad,
    batch_serial_buffer,
    batch_serialise,
    RemoteHandle,
)
from popxl_addons.rts import (
    all_gather_replica_sharded_graph,
    replica_sharded_spec,
    reduce_replica_sharded_graph,
    reduce_replica_sharded_tensor,
)
from popxl_addons.remote import (
    named_variable_buffers,
    load_remote_graph,
    store_remote_graph,
    create_remote_buffer,
    NamedRemoteBuffers,
)
from popxl_addons.ops.grad_reduce_square_add import grad_reduce_square_add

from config import GPTConfig, CONFIG_DIR
from utils.setup import gpt_config_setup
from modelling.embedding import GPTEmbeddingsTP2D, generate_positions
from modelling.decoder import GPTDecoderBlockTP2D
from modelling.gpt_lm import GPTLMHeadLossTP2D, HeadFwdBwd
from utils.utils import tp2d_replica_groups

__all__ = ["pretraining"]


OptimGraphs = Dict[str, GraphWithNamedArgs]


def get_activ_shard_group(a: popxl.Tensor, shard_group: popxl.ReplicaGrouping):
    return shard_group if a.nelms >= RTS_ACTIVATIONS_THRESHOLD else popxl.gcg().ir.replica_grouping(group_size=1)


def get_rts_groups(facts: NamedVariableFactories) -> NamedReplicaGrouping:
    ir = popxl.gcg().ir

    rts_groups = {}
    for k, f in facts.to_dict().items():
        size = np.prod(f.shape)
        rg = f.replica_grouping.const_rg
        if size % rg.group_size == 0 and size >= RTS_THRESHOLD:
            rts_groups[k] = rg
        else:
            rts_groups[k] = ir.replica_grouping(group_size=1)
    return NamedReplicaGrouping.from_dict(rts_groups)


def requires_weight_decay(t: popxl.Tensor):
    return not any(map(lambda exclude: exclude in t.name, ["norm", "bias"]))


def optimizer_graphs(
    config: GPTConfig, optimizer: addons.Module, variables: NamedTensors, facts: NamedVariableFactories
):
    optim_facts = {}
    optim_graphs = {}
    replica_groups = facts.replica_groupings.to_dict()
    rts_groups = get_rts_groups(facts)
    for name, var in variables.to_dict().items():
        input_spec = replica_sharded_spec(var, shard_over=rts_groups[name])
        replica_group = replica_groups[name].const_rg
        optim_facts[name], optim_graphs[name] = optimizer.create_graph(
            input_spec,
            input_spec,
            lr=popxl.TensorSpec((), popxl.float32),
            replica_grouping=replica_group,
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
    def __init__(
        self,
        config: GPTConfig,
        layer_configs,
        optimizer: addons.Module,
        Layer: Type[addons.Module],
        entries: int,
        reuse_buffers: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        self.config = config
        self.Layer = Layer
        _, _, _, rg_dp = tp2d_replica_groups(config)

        self.layer_config = layer_configs[Layer]
        graph_settings: GraphConf = self.layer_config.graph_config

        # Create Graphs for computing forward, gradient and optimizer
        fwd_facts, self.fwd = Layer(config).create_graph(*args, **kwargs)

        # Autodiff
        # self.fwd.args only include named tensors/variables
        tensors_to_accum = filter(self.fwd.args, graph_settings.accumulate)
        grads_required = filter(self.fwd.graph.inputs, graph_settings.grads_required)

        called_graphs_grad_info = {}
        if config.execution.attention_serialisation > 1 and Layer == GPTDecoderBlockTP2D:
            # Optimisation to recompute each blk separately
            assert len(self.fwd.graph.called_graphs) == 1, "expected exactly 1 called graph by decoder layer fwd"
            blk_graph = GraphWithNamedArgs(self.fwd.graph.called_graphs[0])
            grad_blk_graph = addons.transforms.autodiff(blk_graph, grads_required=blk_graph.graph.inputs[:-2])
            grad_blk_graph = addons.transforms.recompute_graph(grad_blk_graph)
            called_graphs_grad_info[blk_graph.graph] = grad_blk_graph.grad_graph_info

        grad_facts, self.bwd = addons.autodiff_with_accumulation(
            self.fwd,
            tensors_to_accum.values_flat(),
            grads_required=grads_required,
            replica_groupings=fwd_facts.replica_groupings,
            called_graphs_grad_info=called_graphs_grad_info,
        )

        popxl.transforms.decompose_sum(self.bwd.graph)

        reuse_rg = {}
        if graph_settings.reuse:
            assert len(graph_settings.reuse) == len(reuse_buffers)
            for var_name in graph_settings.reuse:
                assert var_name in reuse_buffers
                grad_facts.accum.pop(var_name)
                tensors_to_accum.pop(var_name)
                fwd_fact = fwd_facts.pop(var_name)
                reuse_rg[var_name] = fwd_fact.replica_grouping

        # Optimiser
        optim_facts, self.optim = optimizer_graphs(config, optimizer, tensors_to_accum, fwd_facts)

        # Variables required
        self.facts = NamedVariableFactories(fwd=fwd_facts, optim=optim_facts)
        self.grad_facts = grad_facts

        remote_buffer_facts = NamedVariableFactories()

        if graph_settings.remote_buffer_fwd:
            remote_buffer_facts.insert("fwd", fwd_facts.copy())
        if graph_settings.remote_buffer_bwd:
            remote_buffer_facts.insert("bwd", grad_facts.copy())
            remote_buffer_facts.bwd.pop("mean_accum_counter")
        if graph_settings.remote_buffer_optim:
            remote_buffer_facts.insert("optim", optim_facts.copy())

        rts_groups = get_rts_groups(remote_buffer_facts)
        shard_over = {k: rg.group_size for k, rg in rts_groups.to_dict().items()}
        self.buffers = named_variable_buffers(remote_buffer_facts, shard_over_dict=shard_over)
        self.remote_buffer_facts = remote_buffer_facts

        ### Create Graphs for loading/gathering/storing/reducing remote buffers

        # Store fwd and optim
        self._optim_fwd_store = store_remote_graph(self.buffers.filter_keys(["fwd", "optim"]), entries)

        # Store bwd
        if "bwd" in self.buffers:
            self._grad_store = store_remote_graph(self.buffers.bwd, entries)

        # Load fwd
        if "fwd" in self.buffers:
            fwd_buffers: NamedRemoteBuffers = self.buffers.fwd.copy()

            if graph_settings.reuse:
                for var_name in graph_settings.reuse:
                    fwd_buffers.insert(var_name, reuse_buffers[var_name], overwrite=True)
                    rts_groups.fwd.insert(var_name, reuse_rg[var_name], overwrite=True)

            self._fwd_load, self._fwd_load_names = load_remote_graph(fwd_buffers, entries)

        # Load optim + fwd
        self._optim_fwd_load, self._optim_fwd_load_names = load_remote_graph(self.buffers, entries)

        self._fwd_all_gather, self._fwd_all_gather_names = all_gather_replica_sharded_graph(
            NamedTensors.pack(self._fwd_load_names, self._fwd_load.graph.outputs),
            replica_groups=rts_groups.fwd,
            use_io_tiles=USE_IO_TILES,
        )

        # RTS graph: reduce
        grad_accums = self.bwd.args.copy()
        grad_accums.pop("mean_accum_counter")

        if graph_settings.reuse:
            for var_name in graph_settings.reuse:
                grad_accums.accum.pop(var_name)

        rts_bwd_group = NamedReplicaGrouping(accum=rts_groups.fwd.copy())
        self._grad_reduce, self._grad_reduce_names = reduce_replica_sharded_graph(
            grad_accums, "mean", shard_groups=rts_bwd_group, replica_group=rg_dp, use_io_tiles=USE_IO_TILES
        )

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


def batch_serialise_layer(
    graphs: Graphs,
    input_streams: addons.InputStreams,
    output_streams: addons.OutputStreams,
    buffers: Dict[str, popxl.RemoteBuffer],
    shard_group: Optional[popxl.ReplicaGrouping],
):

    config = graphs.config
    phase_config: PhaseConf = graphs.layer_config.phase_config

    shard_groups = {
        name: shard_group if buffer.meta_shape else gcg().ir.replica_grouping(group_size=1)
        for name, buffer in buffers.items()
    }

    load_handles = {}
    store_streams = {}
    store_buffers = {}
    seed_input = None
    for io in ("fwd_inputs", "bwd_inputs", "fwd_outputs", "bwd_outputs"):
        if io == "fwd_inputs":
            graph_tensors = OrderedDict([(t.name, t) for t in graphs.fwd.graph.inputs])
            conf = phase_config.fwd_inputs
        elif io == "bwd_inputs":
            graph_tensors = OrderedDict([(t.name, t) for t in graphs.bwd.graph.inputs])
            conf = phase_config.bwd_inputs
        elif io == "fwd_outputs":
            graph_tensors = OrderedDict([(t.name, t) for t in graphs.fwd.graph.outputs])
            conf = phase_config.fwd_outputs
        elif io == "bwd_outputs":
            graph_tensors = OrderedDict([(t.name, t) for t in graphs.bwd.graph.outputs])
            conf = phase_config.bwd_outputs

        for name_or_idx, handle in conf.items():
            if isinstance(name_or_idx, str):
                t = graph_tensors[name_or_idx]
            else:
                t = graph_tensors.idx(name_or_idx)
            if handle.type == "stream":
                if "inputs" in io:
                    stream = input_streams[handle.name]
                    load_handles[t] = stream
                else:
                    stream = output_streams[handle.name]
                    store_streams[t] = stream
            elif handle.type == "seed":
                assert seed_input is None and "inputs" in io
                seed_input = t
            elif handle.type == "buffer":
                buffer = buffers[handle.name]
                shard_group = shard_groups[handle.name] if handle.rts else None
                remote = RemoteHandle(buffer, handle.row_offset, shard_group)
                if "inputs" in io:
                    load_handles[t] = remote
                else:
                    store_buffers[t] = remote
            else:
                raise Exception("unknown type")

    if not phase_config.fwd_only:
        fwd, bwd = batch_serialise_fwd_and_grad(
            graphs.fwd,
            graphs.bwd,
            graphs.fwd.args,
            config.gradient_accumulation,
            load_handles=load_handles,
            store_streams=store_streams,
            store_buffers=store_buffers,
            seed_input=seed_input,
            rows=phase_config.rows,
            io_mode="io",
        )
        graphs.fwd = fwd.graph
        graphs.bwd = bwd.graph

    else:
        fwd = batch_serialise(
            graphs.fwd,
            config.gradient_accumulation,
            load_handles=load_handles,
            store_streams=store_streams,
            store_buffers=store_buffers,
            seed_input=seed_input,
            rows=phase_config.rows,
            io_mode="io",
        )
        graphs.fwd = fwd.graph


def optimizer_step(optim_graphs: OptimGraphs, ts: NamedTensors, lr: popxl.Tensor, global_norm: popxl.Tensor):
    _variables = ts.fwd.to_dict()
    _state = ts.optim
    _grads = ts.bwd.accum.to_dict()
    for name, graph in optim_graphs.items():
        graph.bind(_state[name]).call(_variables[name], _grads[name], lr, global_norm)


def task_head_optimizer_step(optim_graphs: OptimGraphs, ts: NamedTensors, lr: popxl.Tensor, global_norm: popxl.Tensor):
    _variables = ts.fwd.to_dict()
    _state = ts.optim
    _grads = {name.replace("accum.", ""): t for name, t in ts.bwd.to_dict().items()}
    for name, graph in optim_graphs.items():
        graph.bind(_state.get(name)).call(_variables[name], _grads[name], lr, global_norm)


def global_norm_reduce(config: GPTConfig, grad_norm: popxl.Tensor, grads: NamedTensors):
    for g in grads.tensors:
        ops.add_(grad_norm, grad_reduce_square_add(g, config.execution.loss_scaling))


def pretraining(config: GPTConfig) -> TaskSession:
    replicas = config.execution.data_parallel * config.execution.tensor_parallel_1 * config.execution.tensor_parallel_2
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas

    layer_config = gen_layer_config(config)

    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"

    main = ir.main_graph

    with timer("PopXL IR construction"):
        with main:
            rg_tp1, rg_tp2, rg_tp_all, rg_dp = tp2d_replica_groups(config)
            rg_rts_activations = rg_tp1

            # -----  Define input and output streams -----
            input_shape = (config.execution.micro_batch_size * config.model.sequence_length,)
            input_streams = addons.InputStreams(
                words=(input_shape, popxl.int32), labels=(input_shape, popxl.int32), lr=((), popxl.float32)
            )
            output_streams = addons.OutputStreams(loss=((), config.model.dtype), grad_norm=((), popxl.float32))

            positions = popxl.constant(generate_positions(config), popxl.int32, name="positions")

            # ---- Initialise Random Seed ----
            # Same seed for tp1 group. Different across tp2+dp groups
            seed_v, seed = addons.seed_variable(config.model.seed, replica_grouping=rg_tp1)

            # ----- Build compute graphs -----
            optimizer = AdamOptimizerStep()

            embeddings = Graphs(
                config,
                layer_config,
                optimizer,
                GPTEmbeddingsTP2D,
                1,
                None,
                input_streams.words.spec,
                positions.spec,
                seed=seed.spec,
            )

            x_spec = embeddings.fwd.graph.outputs[0]

            decoder_block = Graphs(
                config, layer_config, optimizer, GPTDecoderBlockTP2D, config.model.layers, None, x_spec, seed=seed.spec
            )

            tied_weight_spec = embeddings.fwd.args.word.weight

            head = Graphs(
                config,
                layer_config,
                optimizer,
                GPTLMHeadLossTP2D,
                1,
                {"head.word_embedding": embeddings.buffers.fwd.word.weight},
                x_spec,
                input_streams.labels.spec,
            )

            # Make Head a single Fwd+Bwd layer to improve phase efficiency
            _, head.fwd = HeadFwdBwd(config, head.fwd, head.bwd, head.facts.fwd, head.grad_facts).create_graph(
                x_spec, input_streams.labels.spec, tied_weight_spec, tied_weight_spec
            )

            # ---- Transform graphs ----

            # Recomputation
            embeddings.bwd = addons.recompute_graph(embeddings.bwd)
            decoder_block.bwd = addons.recompute_graph(decoder_block.bwd)

            # Batch Serialisation
            #   Buffers
            act_shard_group = get_activ_shard_group(x_spec, rg_rts_activations) if RTS_ACT else None
            x_buffer = batch_serial_buffer(
                embeddings.fwd.graph.outputs[0],
                steps=config.gradient_accumulation,
                rows=config.model.layers + 1,
                shard_group=act_shard_group,
            )
            dx_buffer = batch_serial_buffer(
                embeddings.bwd.graph.inputs[0],
                steps=config.gradient_accumulation,
                rows=config.model.layers + 1,
                shard_group=act_shard_group,
            )
            buffers = {"x": x_buffer, "dx": dx_buffer}

            #   Graphs
            batch_serialise_layer(embeddings, input_streams, output_streams, buffers, act_shard_group)
            batch_serialise_layer(decoder_block, input_streams, output_streams, buffers, act_shard_group)
            batch_serialise_layer(head, input_streams, output_streams, buffers, act_shard_group)

            # Available Memory Proportion
            addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

            # ----- Create Variables -----
            read_only_if_exists = config.execution.test_mode
            variables = NamedTensors(random_seed=seed_v)
            variables.insert(
                "embeddings",
                embeddings.remote_buffer_facts.init_remote(
                    embeddings.buffers,
                    0,
                    "embeddings",
                    memmap_dir=config.checkpoint.memmap_dir,
                    read_only_if_exists=read_only_if_exists,
                ),
            )
            variables.insert(
                "decoder",
                NamedTensors.from_dict(
                    {
                        n: decoder_block.facts.init_remote(
                            decoder_block.buffers,
                            n,
                            f"decoder.{n}",
                            memmap_dir=config.checkpoint.memmap_dir,
                            read_only_if_exists=read_only_if_exists,
                        )
                        for n in range(config.model.layers)
                    }
                ),
            )
            variables.insert(
                "head",
                head.facts.init_remote(
                    head.buffers,
                    0,
                    "head",
                    memmap_dir=config.checkpoint.memmap_dir,
                    read_only_if_exists=read_only_if_exists,
                ),
            )

            # ---- Execute ----

            with popxl.in_sequence():
                # Load current learning rate
                lr = ops.host_load(input_streams.lr)

                # Increment random seed
                seed += 1

                def embedding_fwd_phase(seed: popxl.Tensor, positions: popxl.Tensor):
                    # Load Embedding layer
                    embeddings_vars = embeddings.fwd_load(0)
                    embeddings_vars = embeddings.fwd_all_gather(embeddings_vars)
                    # Forward
                    seed, embed_seed = ops.split_random_seed(seed)
                    embeddings.fwd.bind(embeddings_vars).call(0, embed_seed, positions)
                    return seed

                embed_fwd_graph = ir.create_graph(embedding_fwd_phase, seed, positions)
                (seed,) = ops.call(embed_fwd_graph, seed, positions)

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
                ops.repeat(fwd_graph, config.model.layers, i, seed)

                # Buffer to be used later
                tied_weight_grad_buffer = None

                def task_head_fwd_grad_phase():
                    nonlocal tied_weight_grad_buffer
                    # Load task head layer
                    head_vars = NamedTensors(fwd=head.fwd_all_gather(head.fwd_load(0)), bwd=head.grad_facts.init_zero())
                    # Tied weight
                    tied_weight = head_vars.fwd.head.pop("word_embedding")
                    tied_weight_grad = ops.init(tied_weight.shape, tied_weight.dtype, "word_embedding_grad", "zero")
                    # Forward + Gradient
                    head.fwd.bind(head_vars).call(0, tied_weight, tied_weight_grad)
                    # Data parallel reduce
                    reduced_grads = head.grad_reduce(head_vars.bwd)

                    # Global Norm calculation
                    grad_norm = ops.init((), popxl.float32, name="grad_norm", init_type="zero")
                    global_norm_reduce(config, grad_norm, reduced_grads)
                    # Store Gradients
                    head.grad_store(reduced_grads, 0)

                    # Reduce and Store the tied gradient
                    grad_t = reduce_replica_sharded_tensor(
                        tied_weight_grad, "mean", replica_group=rg_dp, shard_group=rg_dp
                    )

                    tied_weight_grad_buffer = create_remote_buffer(
                        grad_t.spec, replica_group=rg_dp, shard_over=rg_dp.group_size
                    )

                    ops.remote_store(tied_weight_grad_buffer, 0, grad_t)
                    return grad_norm

                task_graph = ir.create_graph(task_head_fwd_grad_phase)
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
                (grad_norm,) = ops.call(embed_bwd_graph, lr, grad_norm)

                # Optimizer Step for Layers
                def layer_optim(n: popxl.Tensor, lr: popxl.Tensor, grad_norm: popxl.Tensor):
                    layer_vars = decoder_block.optim_fwd_load(n)
                    optimizer_step(decoder_block.optim, layer_vars, lr, grad_norm)
                    decoder_block.optim_fwd_store(layer_vars, n)
                    return n + 1

                i = popxl.constant(0, name="layer_index")
                layer_optim_graph = ir.create_graph(layer_optim, i, lr, grad_norm)
                ops.repeat(layer_optim_graph, config.model.layers, i, lr, grad_norm)

                def head_optim(lr: popxl.Tensor, grad_norm: popxl.Tensor):
                    # Optimizer Step for Task Head - Only layer norm, tied weights handled by embedding
                    head_vars = head.optim_fwd_load(0)
                    task_head_optimizer_step(head.optim, head_vars, lr, grad_norm)
                    # Store
                    head.optim_fwd_store(head_vars, 0)

                head_optim_graph = ir.create_graph(head_optim, lr, grad_norm)
                ops.call(head_optim_graph, lr, grad_norm)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        repeat_graph(main, config.execution.device_iterations)

        fwd_vars = NamedTensors(
            embeddings=variables.embeddings.fwd,
            decoder=NamedTensors.from_dict({i: variables.decoder[i].fwd for i in range(config.model.layers)}),
            head=variables.head.fwd,
        )

    ir.num_host_transfers = config.execution.device_iterations * config.gradient_accumulation

    session = TaskSession(
        input_streams,
        output_streams,
        fwd_vars,
        ir=ir,
        device_desc="ipu_hw",
        weights_to_host_on_exit=not config.execution.test_mode,
    )

    return session


def main():
    """Run a benchmark configuration"""
    config, _, _ = gpt_config_setup(
        CONFIG_DIR / "pretraining.yml", "release", "tiny", wandb_setup=False, hf_model_setup=False
    )

    session = pretraining(config)
    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    with session:
        # Skip one result
        session.run(inputs)

        durations = []
        for i in range(5):
            start = time.perf_counter()
            outputs = session.run(inputs)
            loss = outputs[session.outputs[0]].mean()
            durations.append(time.perf_counter() - start)
            logging.info(f"Step {i}. Loss {loss:.2f}")
    duration = np.mean(durations)

    samples_per_step = config.execution.device_iterations * config.training.global_batch_size
    result_str = f"Duration: {duration} s " f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
