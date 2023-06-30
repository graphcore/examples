# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
import time

import numpy as np

import popdist
import popxl
from popxl import ops
import popxl_addons as addons

from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons import TaskSession
from popxl_addons.utils import timer
from popxl_addons.patterns import apply_pre_alias_patterns

from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.transforms.batch_serialisation import (
    batch_serial_buffer,
)

from config import GPTConfig, CONFIG_DIR
from utils.setup import gpt_config_setup
from modelling.embedding import GPTEmbeddingsTP, generate_positions
from modelling.decoder import GPTDecoderBlockTP
from modelling.mnli import GPTMnliLossHead
from modelling.gpt_lm import HeadFwdBwd
from utils.utils import replica_groups
from pretraining import (
    get_activ_shard_group,
    Graphs,
    batch_serialise_layer,
    optimizer_step,
    task_head_optimizer_step,
    global_norm_reduce,
)
from mnli_finetuning_config import gen_layer_config, RTS_ACTIVATIONS_THRESHOLD

__all__ = ["mnli_finetuning"]


def mnli_finetuning(config: GPTConfig, no_init: bool = True) -> TaskSession:
    replicas = config.execution.data_parallel * config.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas

    layer_config = gen_layer_config(config)

    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"
    if config.execution.extended_memory:
        opts.engineOptions["target.extendedMemory"] = "true"

    main = ir.main_graph

    with timer("PopXL IR construction"):
        with main:
            rg_tp, rg_dp = replica_groups(config)
            rg_rts_activations = rg_tp

            # -----  Define input and output streams -----
            input_shape = (config.execution.micro_batch_size * config.model.sequence_length,)
            input_streams = addons.InputStreams(
                words=(input_shape, popxl.int32),
                unpadded_length=((config.execution.micro_batch_size,), popxl.int32),
                labels=((config.execution.micro_batch_size,), popxl.int32),
                lr=((), popxl.float32),
            )
            output_streams = addons.OutputStreams(
                logits=((config.execution.micro_batch_size, config.inference.mnli_n_classes), config.model.dtype),
                loss=((), config.model.dtype),
                grad_norm=((), popxl.float32),
            )

            positions = popxl.constant(generate_positions(config), popxl.int32, name="positions")

            # ---- Initialise Random Seed ----
            # Same seed for tp1 group. Different across tp2+dp groups
            seed_v, seed = addons.seed_variable(config.model.seed, replica_grouping=rg_tp)

            # ----- Build compute graphs -----
            optimizer = AdamOptimizerStep()

            embeddings = Graphs(
                config,
                layer_config,
                optimizer,
                GPTEmbeddingsTP,
                1,
                None,
                input_streams.words.spec,
                positions.spec,
                seed=seed.spec,
            )

            x_spec = embeddings.fwd.graph.outputs[0]

            decoder_block = Graphs(
                config, layer_config, optimizer, GPTDecoderBlockTP, config.model.layers, None, x_spec, seed=seed.spec
            )

            head = Graphs(
                config,
                layer_config,
                optimizer,
                GPTMnliLossHead,
                1,
                None,
                x_spec,
                input_streams.unpadded_length.spec,
                input_streams.labels.spec,
            )

            # Make Head a single Fwd+Bwd layer to improve phase efficiency
            _, head.fwd = HeadFwdBwd(config, head.fwd, head.bwd, head.facts.fwd, head.grad_facts).create_graph(
                x_spec, input_streams.unpadded_length.spec, input_streams.labels.spec
            )

            # ---- Transform graphs ----

            # Recomputation
            embeddings.bwd = addons.recompute_graph(embeddings.bwd)
            decoder_block.bwd = addons.recompute_graph(decoder_block.bwd)

            # Batch Serialisation
            #   Buffers
            act_shard_group = (
                get_activ_shard_group(x_spec, rg_rts_activations, RTS_ACTIVATIONS_THRESHOLD)
                if config.execution.rts_activations
                else None
            )
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
            # Structure should match gpt_lm.GPTLMHeadModelTP.hf_mapping
            variables = NamedTensors(random_seed=seed_v)
            transformer = NamedTensors()
            variables.insert("transformer", transformer)
            transformer.insert(
                "embeddings",
                embeddings.remote_buffer_facts.init_remote(
                    embeddings.buffers,
                    0,
                    "embeddings",
                ),
            )
            variables.insert(
                "head",
                head.facts.init_remote(
                    head.buffers,
                    0,
                    "head",
                ),
            )
            # Only don't init forward vars of decoder block.
            # Embedding and heads include an offset variable that needs to be initialised
            for key in ("fwd", "bwd", "optim"):
                empty = no_init and key == "fwd"
                if key in decoder_block.facts.keys():
                    for n in range(config.model.layers):
                        transformer.insert(
                            f"decoder.{n}.{key}",
                            decoder_block.facts[key].init_remote(
                                decoder_block.buffers[key], n, f"decoder.{n}.{key}", empty=empty
                            ),
                            overwrite=True,
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

                def task_head_fwd_grad_phase():
                    # Load task head layer
                    head_vars = NamedTensors(fwd=head.fwd_all_gather(head.fwd_load(0)), bwd=head.grad_facts.init_zero())
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

        fwd_vars = NamedTensors.from_dict(
            {
                "transformer.embeddings": variables.transformer.embeddings.fwd,
                "transformer.decoder": NamedTensors.from_dict(
                    {i: variables.transformer.decoder[i].fwd for i in range(config.model.layers)}
                ),
                "head": variables.head.fwd,
            }
        )

    ir.num_host_transfers = config.execution.device_iterations * config.gradient_accumulation

    session = TaskSession(
        input_streams,
        output_streams,
        fwd_vars,
        ir=ir,
        device_desc="ipu_hw",
    )

    return session


def main():
    """Run a benchmark configuration"""
    config, _, _ = gpt_config_setup(
        CONFIG_DIR / "mnli_finetuning.yml", "release", "tiny", wandb_setup=False, hf_model_setup=False
    )

    session = mnli_finetuning(config)
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
            loss = outputs[session.outputs.loss].mean()
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
