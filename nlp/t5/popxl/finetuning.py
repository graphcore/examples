# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import logging
import time
import numpy as np

import popdist
import popxl
from popxl import ops
import popxl_addons as addons
from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons import TaskSession
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.utils import timer
from popxl_addons.named_replica_grouping import get_ild_replica_grouping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.transforms.batch_serialisation import batch_serial_buffer
from popxl_addons.rts import reduce_replica_sharded_tensor
from popxl_addons.remote import create_remote_buffer
from popxl_addons.ops.grad_reduce_square_add import grad_reduce_square_add

from config import T5Config, CONFIG_DIR
from utils.setup import t5_config_setup
from graphs.graphs import (
    Graphs,
    OptimGraphs,
    get_activ_shard_group,
)
from graphs.embedding import (
    create_embeddings_graph,
    create_decoder_embeddings_graph,
    embeddings_batch_serialise,
    decoder_embeddings_batch_serialise,
)
from graphs.encoder_decoder import (
    create_t5_block_graph,
    create_encoder_head_graph,
    t5_block_batch_serialise,
    encoder_head_batch_serialise,
)
from graphs.head import (
    create_task_head_graph,
    head_batch_serialise,
)

__all__ = ["finetuning"]


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


def global_norm_reduce(config: T5Config, grad_norm: popxl.Tensor, grads: NamedTensors):
    for g in grads.tensors:
        ops.add_(grad_norm, grad_reduce_square_add(g, config.execution.loss_scaling))


def init_remote_vars(
    variables: NamedTensors,
    name: str,
    key: str,
    graph: Graphs,
    empty: bool,
    entry: int = 0,
    prefix: str = "",
):
    if not prefix:
        prefix = name
    variables.insert(
        name,
        graph.facts[key].init_remote(graph.buffers[key], entry, prefix, empty=empty),
        overwrite=True,
    )


def finetuning(config: T5Config, args: argparse.Namespace = None, no_init: bool = True) -> TaskSession:
    replicas = config.execution.data_parallel * config.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas
    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"
    # The following options are only needed for the XXL config
    size = args.config.split("_")[0] if args is not None else "xxl"
    if size == "xxl":
        opts.engineOptions["target.extendedMemory"] = "true"
        opts.engineOptions["opt.internalExchangeOptimisationTarget"] = "memory"

    with timer("PopXL IR construction"):
        main = ir.main_graph
        dp_group = ir.replica_grouping(
            stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
        )
        tp_group = ir.replica_grouping(stride=1, group_size=config.execution.tensor_parallel)
        with main:
            # ----- Define input and output streams -----
            input_shape = (config.execution.micro_batch_size * config.model.sequence_length,)
            input_streams = addons.InputStreams(
                words=(input_shape, popxl.int32),
                attention_mask=(input_shape, config.model.dtype),
                decoder_words=(input_shape, popxl.int32),
                decoder_attention_mask=(input_shape, config.model.dtype),
                labels=(input_shape, popxl.int32),
                lr=((), popxl.float32),
            )
            output_streams = addons.OutputStreams(loss=((), config.model.dtype), grad_norm=((), popxl.float32))

            # ---- Initialise Random Seed ----
            seed_v, seed = addons.seed_variable(config.model.seed, tp_group)

            # ----- Build compute graphs -----
            optimizer = AdamOptimizerStep()

            embeddings = create_embeddings_graph(config, optimizer, input_streams.words.spec, seed=seed.spec)

            decoder_embeddings = create_decoder_embeddings_graph(
                config,
                optimizer,
                embeddings,
                input_streams.decoder_words.spec,
                embeddings.fwd.args.word.weight.spec,
                seed=seed.spec,
            )

            scale_spec = popxl.TensorSpec((), config.model.dtype)
            t5_block = create_t5_block_graph(
                config,
                optimizer,
                embeddings.fwd.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
                embeddings.fwd.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
                scale_spec,
                embeddings.fwd.args.rel_pos_weight.spec,
                seed=seed.spec,
            )

            encoder_head = create_encoder_head_graph(
                config, optimizer, t5_block.fwd.graph.outputs[0].spec, seed=seed.spec
            )

            head = create_task_head_graph(
                config, optimizer, t5_block.fwd.graph.outputs[0], input_streams.labels.spec, seed=seed.spec
            )

            # ---- Transform graphs ----

            # Recomputation
            embeddings.bwd = addons.recompute_graph(embeddings.bwd)
            decoder_embeddings.bwd = addons.recompute_graph(decoder_embeddings.bwd)
            t5_block.bwd = addons.recompute_graph(t5_block.bwd)
            encoder_head.bwd = addons.recompute_graph(encoder_head.bwd)

            # Batch Serialisation
            steps = config.gradient_accumulation
            #   Buffers
            x_buffer = batch_serial_buffer(
                embeddings.fwd.graph.outputs[0],
                steps=steps,
                rows=config.model.layers + 2 + config.model.layers + 1,
                shard_group=get_activ_shard_group(embeddings.fwd.graph.outputs[0], tp_group),
            )
            dx_buffer = batch_serial_buffer(
                embeddings.bwd.graph.inputs[0],
                steps=steps,
                rows=config.model.layers + 2 + config.model.layers + 1,
                shard_group=get_activ_shard_group(embeddings.bwd.graph.inputs[0], tp_group),
            )
            mask_buffer = batch_serial_buffer(
                t5_block.fwd.graph.inputs[1],
                steps=steps,
                rows=2,
            )
            # Buffer to store the dx wrt the encoder output for each decoder layer
            dx_enc_buffer = batch_serial_buffer(
                encoder_head.bwd.graph.inputs[0],
                steps=steps,
                rows=config.model.layers,
                shard_group=get_activ_shard_group(encoder_head.bwd.graph.inputs[0], tp_group),
            )

            # Graphs
            embeddings_batch_serialise(config, embeddings, input_streams, x_buffer, dx_buffer)
            decoder_embeddings_batch_serialise(config, decoder_embeddings, input_streams, x_buffer, dx_buffer)
            t5_block_batch_serialise(
                config,
                t5_block,
                x_buffer,
                mask_buffer,
                dx_buffer,
                dx_enc_buffer,
            )
            encoder_head_batch_serialise(config, encoder_head, x_buffer, dx_buffer)
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
                    init_remote_vars(transformer, f"embeddings.{key}", key, embeddings, empty)
                if key in decoder_embeddings.facts.keys():
                    init_remote_vars(transformer, f"decoder_embeddings.{key}", key, decoder_embeddings, empty)
                if key in t5_block.facts.keys():
                    for n in range(2 * config.model.layers):
                        name = "encoder" if n < config.model.layers else "decoder"
                        idx = n % config.model.layers
                        init_remote_vars(transformer, f"{name}.{idx}.{key}", key, t5_block, empty, n)
                if key in encoder_head.facts.keys():
                    init_remote_vars(transformer, f"encoder_head.{key}", key, encoder_head, empty)
                if key in head.facts.keys():
                    init_remote_vars(variables, f"lm_head.{key}", key, head, empty, 0, "lm_head")

            # ---- Execute ----
            with popxl.in_sequence():
                # Load current learning rate
                lr = ops.host_load(input_streams.lr)
                # Increment random seed
                seed += 1

                @popxl.io_tiles()
                def fill_buffer_from_host(
                    i: popxl.Tensor,
                    stream1: popxl.HostToDeviceStream,
                    stream2: popxl.HostToDeviceStream,
                    buffer: popxl.RemoteBuffer,
                ):
                    t1 = ops.host_load(stream1)
                    t2 = ops.host_load(stream2)
                    ops.remote_store(buffer, i, t1)
                    ops.remote_store(buffer, steps + i, t2)

                # Load from host then store all masks.
                # The first row of the buffer is going to be populated by the encoder masks,
                # while the second row is going to be populated by the decoder masks
                mask_fill_graph = ir.create_graph(
                    fill_buffer_from_host,
                    popxl.constant(0, popxl.uint32),
                    input_streams.attention_mask,
                    input_streams.decoder_attention_mask,
                    mask_buffer,
                )
                for i in range(steps):
                    ops.call(mask_fill_graph, i)

                def embedding_fwd_phase(seed):
                    # Load Embedding layer
                    embeddings_vars = embeddings.fwd_load(0)
                    embeddings_vars = embeddings.fwd_all_gather(embeddings_vars)
                    rel_pos_weight = embeddings_vars.rel_pos_weight
                    # Forward
                    seed, embed_seed = ops.split_random_seed(seed)
                    embeddings.fwd.bind(embeddings_vars).call(0, embed_seed)
                    return seed, rel_pos_weight

                embed_fwd_graph = ir.create_graph(embedding_fwd_phase, seed)
                (seed, rel_pos_weight) = ops.call(embed_fwd_graph, seed)

                def single_t5_block_fwd_phase(
                    n: popxl.Tensor,
                    seed: popxl.Tensor,
                    rel_pos_weight: popxl.Tensor,
                    scale: popxl.Tensor,
                    offset: popxl.Tensor,
                ):
                    # Load T5 block
                    layer_vars = t5_block.fwd_load(n)
                    layer_vars = t5_block.fwd_all_gather(layer_vars)
                    # Forward
                    seed, layer_seed = ops.split_random_seed(seed)
                    t5_block.fwd.bind(layer_vars).call(n + offset, layer_seed, scale, rel_pos_weight)
                    return n + 1, seed

                # For encoder layers, i is in [0, N-1]
                i = popxl.constant(0, name="layer_index")
                offset = popxl.constant(0, name="offset")
                # Encoder layers mask out the cross-attention part
                scale = popxl.constant(0, config.model.dtype, "cross_attn_scale")
                # Pass the shared weight to the encoder layers
                t5_block_fwd_graph = ir.create_graph(single_t5_block_fwd_phase, i, seed, rel_pos_weight, scale, offset)
                ops.repeat(t5_block_fwd_graph, config.model.layers, i, seed, rel_pos_weight, scale, offset)

                def encoder_head_fwd_phase(seed: popxl.Tensor):
                    # Load encoder head
                    head_vars = encoder_head.fwd_load(0)
                    head_vars = encoder_head.fwd_all_gather(head_vars)
                    # Forward
                    seed, embed_seed = ops.split_random_seed(seed)
                    encoder_head.fwd.bind(head_vars).call(0, embed_seed)
                    return seed

                encoder_head_fwd_graph = ir.create_graph(encoder_head_fwd_phase, seed)
                (seed,) = ops.call(encoder_head_fwd_graph, seed)

                def decoder_embedding_fwd_phase(seed):
                    # Load Embedding layer
                    embeddings_vars = decoder_embeddings.fwd_load(0)
                    embeddings_vars = decoder_embeddings.fwd_all_gather(embeddings_vars)
                    # Get the embedding weights
                    embedding_weight_t = embeddings_vars.pop("weight")
                    rel_pos_weight = embeddings_vars.rel_pos_weight
                    # Forward
                    seed, embed_seed = ops.split_random_seed(seed)
                    decoder_embeddings.fwd.bind(embeddings_vars).call(0, embed_seed, embedding_weight_t)
                    return seed, rel_pos_weight

                dec_embed_fwd_graph = ir.create_graph(decoder_embedding_fwd_phase, seed)
                (seed, rel_pos_weight) = ops.call(dec_embed_fwd_graph, seed)

                # For decoder layers, i is in [N, 2N-1]
                i = popxl.constant(config.model.layers, name="layer_index")
                offset = popxl.constant(2, name="offset")
                # Decoder layers don't mask out the cross-attention part
                scale = popxl.constant(1, config.model.dtype, "cross_attn_scale")
                # Pass the shared weight to the decoder layers
                ops.repeat(t5_block_fwd_graph, config.model.layers, i, seed, rel_pos_weight, scale, offset)

                def task_head_fwd_grad_phase(seed: popxl.Tensor):
                    # Load task head layer
                    head_vars = head.fwd_load(0)
                    head_vars = NamedTensors(fwd=head.fwd_all_gather(head_vars), bwd=head.grad_facts.init_zero())
                    # Forward + Gradient
                    seed, head_seed = ops.split_random_seed(seed)
                    head.fwd.bind(head_vars).call(0, head_seed)
                    # Data parallel reduce
                    reduced_grads = head.grad_reduce(head_vars.bwd)
                    # Global Norm calculation
                    grad_norm = ops.init((), popxl.float32, name="grad_norm", init_type="zero")
                    global_norm_reduce(config, grad_norm, reduced_grads.accum)
                    # Store Gradients
                    head.grad_store(reduced_grads, 0)
                    return grad_norm, seed

                task_graph = ir.create_graph(task_head_fwd_grad_phase, seed)
                (grad_norm, seed) = ops.call(task_graph, seed)

                def single_t5_block_grad_phase(
                    n: popxl.Tensor,
                    grad_norm: popxl.TensorByRef,
                    rel_pos_weight_grad: popxl.TensorByRef,
                    offset: popxl.Tensor,
                ):
                    # Load layer
                    layer_vars = t5_block.fwd_load(n)
                    layer_vars = t5_block.fwd_all_gather(layer_vars)
                    # Gradient
                    grads = t5_block.grad_facts.init_zero()
                    bwd_vars = grads.copy()
                    bwd_vars.update(layer_vars)
                    input_dict = {t5_block.bwd.args.accum.rel_pos_weight: rel_pos_weight_grad}
                    # We need the offset to be in [2N+1, N+2]
                    t5_block.bwd.bind(bwd_vars).call(n + offset, args=input_dict)
                    # Data parallel reduce
                    reduced_grads = t5_block.grad_reduce(grads)
                    # Global Norm calculation
                    global_norm_reduce(config, grad_norm, reduced_grads)
                    # Store gradient
                    t5_block.grad_store(reduced_grads, n)
                    return n - 1

                # The layer index starts at 2N-1, and will go down to N
                i = popxl.constant(2 * config.model.layers - 1, name="layer_index")
                offset = popxl.constant(2, name="offset")
                # Prepare a tensor that will contain the aggregated grad of the rel_pos_weight
                rel_pos_weight = decoder_embeddings.fwd.args.rel_pos_weight
                dec_rel_pos_weight_grad = ops.init(
                    rel_pos_weight.shape, rel_pos_weight.dtype, "dec_rel_pos_grad", "zero"
                )
                t5_block_bwd_graph = ir.create_graph(
                    single_t5_block_grad_phase, i, grad_norm, dec_rel_pos_weight_grad, offset
                )
                ops.repeat(t5_block_bwd_graph, config.model.layers, i, grad_norm, dec_rel_pos_weight_grad, offset)
                # Data parallel reduce the gradients of the rel_pos_weight
                dec_rel_pos_weight_grad = reduce_replica_sharded_tensor(
                    dec_rel_pos_weight_grad,
                    "mean",
                    replica_group=dp_group,
                    shard_group=get_ild_replica_grouping(dp_group),
                )

                # Buffer to be used by the 2 embedding layers to aggregate the gradients
                emb_weight_grad_buffer = None

                def decoder_embedding_grad_phase(grad_norm: popxl.TensorByRef):
                    nonlocal emb_weight_grad_buffer
                    # Load Embedding layer
                    embeddings_vars = decoder_embeddings.fwd_load(0)
                    embeddings_vars = decoder_embeddings.fwd_all_gather(embeddings_vars)
                    # Get the embedding weights
                    embedding_weight_t = embeddings_vars.pop("weight")
                    # Gradient
                    grads = decoder_embeddings.grad_facts.init_zero()
                    bwd_vars = grads.copy()
                    bwd_vars.update(embeddings_vars)
                    emb_weight_grad_t = ops.init(
                        embedding_weight_t.shape, embedding_weight_t.dtype, "word_embedding_grad_t", "zero"
                    )
                    input_dict = {decoder_embeddings.bwd.args.accum.word_embedding: emb_weight_grad_t}
                    decoder_embeddings.bwd.bind(bwd_vars).call(0, args=input_dict)

                    # Replica-reduce the gradient wrt the embedding weight, and store it
                    # in a buffer that will be read by the encoder embedding
                    grad_t = reduce_replica_sharded_tensor(
                        emb_weight_grad_t,
                        "mean",
                        replica_group=dp_group,
                        shard_group=get_ild_replica_grouping(dp_group),
                    )
                    emb_weight_grad_buffer = create_remote_buffer(
                        grad_t.spec, replica_group=dp_group, shard_over=get_ild_replica_grouping(dp_group).group_size
                    )
                    ops.remote_store(emb_weight_grad_buffer, 0, grad_t)
                    return grad_norm

                dec_embed_bwd_graph = ir.create_graph(decoder_embedding_grad_phase, grad_norm)
                (grad_norm,) = ops.call(dec_embed_bwd_graph, grad_norm)

                # At this point the decoder layers will have written to dx_enc_buffer (n_layers x grad_acc)
                # We need to reduce sum it across the n_layer dimension, and write the result in dx_buffer[n_layers + 1] (grad_acc),
                # which the encoder head is then going to read from

                def single_layer_acc(
                    l_idx: popxl.Tensor,
                    batch_idx: popxl.Tensor,
                    acc: popxl.TensorByRef,
                    from_buffer: popxl.RemoteBuffer,
                ):
                    t = ops.remote_load(from_buffer, l_idx * steps + batch_idx)
                    ops.add_(acc, t)
                    return l_idx + 1

                @popxl.io_tiles()
                def accumulate_enc_grad(
                    batch_idx: popxl.Tensor, from_buffer: popxl.RemoteBuffer, to_buffer: popxl.RemoteBuffer
                ):
                    # This function performs the following operations for the given batch index:
                    # - Initialise an accumulator
                    # - Accumulate the dx_enc from the source buffer, for all the layers
                    # - Store the result in the destination buffer
                    dx_enc_acc = ops.init(dx_enc_buffer.tensor_shape, dx_enc_buffer.tensor_dtype, "dx_enc_acc", "zero")
                    l_idx = popxl.constant(0, name="layer_index")
                    acc_graph = ir.create_graph(single_layer_acc, l_idx, batch_idx, dx_enc_acc, from_buffer)
                    ops.repeat(acc_graph, config.model.layers, l_idx, batch_idx, dx_enc_acc)
                    ops.remote_store(to_buffer, (config.model.layers + 1) * steps + batch_idx, dx_enc_acc)

                accumulate_enc_grad = ir.create_graph(
                    accumulate_enc_grad, popxl.constant(0, popxl.int32), dx_enc_buffer, dx_buffer
                )
                for i in range(steps):
                    ops.call(accumulate_enc_grad, i)

                def encoder_head_grad_phase(grad_norm: popxl.TensorByRef):
                    # Encoder head
                    layer_vars = encoder_head.fwd_load(0)
                    layer_vars = encoder_head.fwd_all_gather(layer_vars)
                    # Gradient
                    grads = encoder_head.grad_facts.init_zero()
                    bwd_vars = grads.copy()
                    bwd_vars.update(layer_vars)
                    encoder_head.bwd.bind(bwd_vars).call(0)
                    # Data parallel reduce
                    reduced_grads = encoder_head.grad_reduce(grads)
                    # Global Norm calculation
                    global_norm_reduce(config, grad_norm, reduced_grads)
                    # Store gradient
                    encoder_head.grad_store(reduced_grads, 0)
                    return grad_norm

                bwd_graph = ir.create_graph(encoder_head_grad_phase, grad_norm)
                (grad_norm,) = ops.call(bwd_graph, grad_norm)

                # The layer index starts at N-1, and will go down to 0
                i = popxl.constant(config.model.layers - 1, name="layer_index")
                offset = popxl.constant(0, name="offset")
                # Prepare a tensor that will contain the aggregated grad of the rel_pos_weight
                rel_pos_weight = embeddings.fwd.args.rel_pos_weight
                rel_pos_weight_grad = ops.init(rel_pos_weight.shape, rel_pos_weight.dtype, "rel_pos_grad", "zero")
                ops.repeat(t5_block_bwd_graph, config.model.layers, i, grad_norm, rel_pos_weight_grad, offset)
                # Data parallel reduce the gradients of the rel_pos_weight
                rel_pos_weight_grad = reduce_replica_sharded_tensor(
                    rel_pos_weight_grad,
                    "mean",
                    replica_group=dp_group,
                    shard_group=get_ild_replica_grouping(dp_group),
                )

                def embedding_grad_optimizer_phase(
                    lr: popxl.Tensor, grad_norm: popxl.TensorByRef, rel_pos_weight_grad: popxl.Tensor
                ):
                    nonlocal emb_weight_grad_buffer
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
                    # Add the gradient from the decoder embedding
                    emb_weight_grad_t = ops.remote_load(emb_weight_grad_buffer, 0)
                    ops.add_(reduced_grads.accum.word.weight, emb_weight_grad_t)

                    # Global Norm calculation
                    global_norm_reduce(config, grad_norm, reduced_grads)
                    # Finalise global bwd norm with an all reduce and sqrt
                    grad_norm = ops.sqrt(ops.collectives.replicated_all_reduce(grad_norm, op="add"))
                    ops.host_store(output_streams.grad_norm, grad_norm)

                    # Optimizer Step for Embeddings.
                    # Note: No need to store then load the gradient: just use it directly
                    embeddings_vars.insert("bwd", reduced_grads)
                    # Insert in the bwd accumulators the grad of rel_pos_weight
                    embeddings_vars.bwd.accum.insert("rel_pos_weight", rel_pos_weight_grad)
                    optimizer_step(embeddings.optim, embeddings_vars, lr, grad_norm)
                    # Store
                    embeddings.optim_fwd_store(embeddings_vars, 0)
                    return grad_norm

                embed_bwd_graph = ir.create_graph(embedding_grad_optimizer_phase, lr, grad_norm, rel_pos_weight_grad)
                (grad_norm,) = ops.call(embed_bwd_graph, lr, grad_norm, rel_pos_weight_grad)

                def decoder_embedding_optimizer_phase(
                    lr: popxl.Tensor, grad_norm: popxl.TensorByRef, rel_pos_weight_grad: popxl.Tensor
                ):
                    embeddings_vars = decoder_embeddings.optim_fwd_load(0)
                    # Pop the embedding weights
                    embeddings_vars.fwd.pop("weight")

                    # Insert in the bwd accumulators the grad of rel_pos_weight
                    embeddings_vars.insert("bwd.accum.rel_pos_weight", rel_pos_weight_grad)
                    optimizer_step(decoder_embeddings.optim, embeddings_vars, lr, grad_norm)
                    # Store
                    decoder_embeddings.optim_fwd_store(embeddings_vars, 0)
                    return grad_norm

                dec_embed_opt_graph = ir.create_graph(
                    decoder_embedding_optimizer_phase, lr, grad_norm, dec_rel_pos_weight_grad
                )
                (grad_norm,) = ops.call(dec_embed_opt_graph, lr, grad_norm, dec_rel_pos_weight_grad)

                def t5_layer_optim(n: popxl.Tensor, lr: popxl.Tensor, grad_norm: popxl.Tensor):
                    layer_vars = t5_block.optim_fwd_load(n)
                    optimizer_step(t5_block.optim, layer_vars, lr, grad_norm)
                    t5_block.optim_fwd_store(layer_vars, n)
                    return n + 1

                # The index will go from 0 to 2N-1, optimising both the encoder and decoder layers
                i = popxl.constant(0, name="layer_index")
                layer_optim_graph = ir.create_graph(t5_layer_optim, i, lr, grad_norm)
                ops.repeat(layer_optim_graph, 2 * config.model.layers, i, lr, grad_norm)

                def encoder_head_optim(lr: popxl.Tensor, grad_norm: popxl.Tensor):
                    layer_vars = encoder_head.optim_fwd_load(0)
                    optimizer_step(encoder_head.optim, layer_vars, lr, grad_norm)
                    encoder_head.optim_fwd_store(layer_vars, 0)

                enc_head_optim_graph = ir.create_graph(encoder_head_optim, lr, grad_norm)
                ops.call(enc_head_optim_graph, lr, grad_norm)

                # Note: we've already performed the optimizer step for the decoder embedding
                # at the end of its backward phase

                def head_optim(lr: popxl.Tensor, grad_norm: popxl.Tensor):
                    # Optimizer Step for the task head
                    head_vars = head.optim_fwd_load(0)
                    task_head_optimizer_step(head.optim, head_vars, lr, grad_norm)
                    # Store
                    head.optim_fwd_store(head_vars, 0)

                head_optim_graph = ir.create_graph(head_optim, lr, grad_norm)
                ops.call(head_optim_graph, lr, grad_norm)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

        repeat_graph(main, config.execution.device_iterations)

        fwd_vars = NamedTensors.from_dict(
            {
                "transformer.embeddings": variables.transformer.embeddings.fwd,
                "transformer.decoder_embeddings": variables.transformer.decoder_embeddings.fwd,
                "transformer.encoder": NamedTensors.from_dict(
                    {i: variables.transformer.encoder[i].fwd for i in range(config.model.layers)}
                ),
                "transformer.encoder_head": variables.transformer.encoder_head.fwd,
                "transformer.decoder": NamedTensors.from_dict(
                    {i: variables.transformer.decoder[i].fwd for i in range(config.model.layers)}
                ),
                "lm_head": variables.lm_head.fwd,
            }
        )

        optim_vars = NamedTensors.from_dict(
            {
                "transformer.embeddings": variables.transformer.embeddings.optim,
                "transformer.decoder_embeddings": variables.transformer.decoder_embeddings.optim,
                "transformer.encoder": NamedTensors.from_dict(
                    {i: variables.transformer.encoder[i].optim for i in range(config.model.layers)}
                ),
                "transformer.encoder_head": variables.transformer.encoder_head.optim,
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
        weights_to_host_on_exit=False,
    )

    return session


def main():
    """Run a benchmark configuration"""
    config, args, _ = t5_config_setup(
        CONFIG_DIR / "finetuning.yml", "release", "xxl_pod16", wandb_setup=False, hf_model_setup=False
    )
    session = finetuning(config, args)
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
