# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
This file defines the PopXL inference session that will be executed.  This
example makes use of phased execution to avoid loading the entire billions of
parameters model into memory at one time, loading one transformer block at a time
instead. The main function `inference` is intended to be imported and used in
some front-end such as another inference script or a Jupyter notebook. Running
this file directly is for testing purposes only.
"""

import logging
import time

import numpy as np
import popdist

import popxl
from popxl import ops
from math import ceil
from typing import Tuple

import popxl_addons as addons
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.remote import named_variable_buffers, load_remote_graph, create_remote_buffer
from popxl_addons.utils import timer
from popxl_addons.task_session import TaskSession

from config import CONFIG_DIR, LlamaConfig
from modelling.embedding import LlamaEmbeddingsTP
from modelling.decoder import LlamaDecoderBlockTP
from modelling.llama_lm import LlamaLMHeadTP, gather_logits_tp
from utils.setup import llama_config_setup
from transformers.models.llama.modeling_llama import LlamaForCausalLM

__all__ = ["inference"]


def prepare_cache(config: LlamaConfig) -> Tuple[popxl.TensorSpec]:
    attn_tp = (
        config.execution.attention_tensor_parallel
        if config.execution.attention_tensor_parallel
        else config.execution.tensor_parallel
    )

    sharded_k_shape = (
        config.execution.micro_batch_size,
        config.model.attention.kv_heads // attn_tp,
        config.model.hidden_size // config.model.attention.heads,
        config.model.sequence_length,
    )

    sharded_v_shape = (
        *sharded_k_shape[:2],
        config.model.sequence_length,
        config.model.hidden_size // config.model.attention.heads,
    )

    k_spec = popxl.TensorSpec(shape=sharded_k_shape, dtype=config.model.dtype, meta_shape=sharded_k_shape)

    v_spec = popxl.TensorSpec(shape=sharded_v_shape, dtype=config.model.dtype, meta_shape=sharded_v_shape)

    past_k_buffer = create_remote_buffer(
        spec=k_spec,
        entries=config.model.layers,
        replica_group=popxl.gcg().ir.replica_grouping(
            stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
        ),
    )

    past_v_buffer = create_remote_buffer(
        spec=v_spec,
        entries=config.model.layers,
        replica_group=popxl.gcg().ir.replica_grouping(
            stride=config.execution.tensor_parallel, group_size=config.execution.data_parallel
        ),
    )
    return past_k_buffer, past_v_buffer, k_spec, v_spec


def inference(config: LlamaConfig) -> TaskSession:
    assert config.execution.data_parallel == 1, "You can't use data-parallelism for inference"
    replicas = config.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas

    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"
    # opts.engineOptions["target.extendedMemory"] = "true"

    with timer("PopXL IR construction"):
        with ir.main_graph:
            # -----  Define input and output streams -----

            input_shape = (
                config.execution.micro_batch_size,
                config.model.sequence_length,
            )

            if config.execution.use_cache:
                input_shape = (config.execution.micro_batch_size,)
                past_k_buffer, past_v_buffer, past_k_spec, past_v_spec = prepare_cache(config)

            input_streams = addons.InputStreams(
                words=(input_shape, popxl.int32), last_token_indices=((config.execution.micro_batch_size,), popxl.int32)
            )

            output_streams = addons.OutputStreams(
                next_token_logits=(
                    (config.execution.micro_batch_size, config.model.embedding.vocab_size),
                    config.model.dtype,
                )
            )

            # ----- Build compute graphs -----

            embeddings_facts, embeddings_graph = LlamaEmbeddingsTP(config).create_graph(input_streams.words.spec)

            decoder_inputs = (embeddings_graph.graph.outputs[0],)

            if config.execution.use_cache:
                decoder_inputs = (
                    embeddings_graph.graph.outputs[0],
                    input_streams.last_token_indices.spec,
                    past_k_spec,
                    past_v_spec,
                )

            layer_facts, layer_graph = LlamaDecoderBlockTP(config).create_graph(*decoder_inputs)

            lm_facts, lm_graph = LlamaLMHeadTP(config).create_graph(layer_graph.graph.outputs[0])

            # ---- Transform graphs ----

            addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

            # ----- Create Variables -----

            # Create RemoteBuffers for each variable
            embeddings_buffers = named_variable_buffers(embeddings_facts, shard_over_dict=False)
            layer_buffers = named_variable_buffers(layer_facts, entries=config.model.layers, shard_over_dict=False)
            lm_buffers = named_variable_buffers(lm_facts, shard_over_dict=False)

            variables = NamedTensors()
            transformer = NamedTensors()
            transformer.insert(
                "embeddings", embeddings_facts.init_remote(embeddings_buffers, 0, "embeddings", empty=True)
            )

            transformer.insert(
                "decoder",
                NamedTensors.from_dict(
                    {
                        n: layer_facts.init_remote(layer_buffers, n, f"decoder.{n}", empty=True)
                        for n in range(config.model.layers)
                    }
                ),
            )

            variables.insert("transformer", transformer)
            variables.insert("lm_head", lm_facts.init_remote(lm_buffers, 0, "lm_head", empty=True))

            # ---- Execute ----
            with popxl.in_sequence():
                word = ops.host_load(input_streams.words)
                last_token_indices = ops.host_load(input_streams.last_token_indices)
                # Embeddings
                load_graph, names = load_remote_graph(embeddings_buffers)
                embedding_vars = NamedTensors.pack(names, load_graph.call(0))
                (x,) = embeddings_graph.bind(embedding_vars).call(word)

                # Decoder
                load_graph, names = load_remote_graph(layer_buffers)

                def layer(x, n, last_token_indices):
                    load_graph, names = load_remote_graph(layer_buffers)
                    layer_vars = NamedTensors.pack(names, load_graph.call(n))

                    if config.execution.use_cache:
                        past_k = ops.remote_load(past_k_buffer, offset=n, name="past_k")
                        past_v = ops.remote_load(past_v_buffer, offset=n, name="past_v")

                        (x, k, v,) = layer_graph.bind(
                            layer_vars
                        ).call(x, last_token_indices, past_k, past_v)

                        ops.remote_store(past_k_buffer, n, k)
                        ops.remote_store(past_v_buffer, n, v)
                    else:
                        (x,) = layer_graph.bind(layer_vars).call(x)

                    return x, n + 1

                i = popxl.constant(0, name="layer_index")
                layers_graph = ir.create_graph(layer, x, i, last_token_indices)
                x, _ = ops.repeat(layers_graph, config.model.layers, x, i, last_token_indices)

                # LM head
                load_graph, names = load_remote_graph(lm_buffers)
                squad_vars = NamedTensors.pack(names, load_graph.call(0))
                (logits,) = lm_graph.bind(squad_vars).call(x)

                logits = gather_logits_tp(config, logits, last_token_indices)
                ops.host_store(
                    output_streams.next_token_logits, logits.reshape_(output_streams.next_token_logits.shape)
                )

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

    logging.info("PopXL IR construction complete")

    ir.num_host_transfers = config.execution.device_iterations

    session = TaskSession(
        inputs=input_streams,
        outputs=output_streams,
        state=NamedTensors(fwd=variables),
        ir=ir,
        device_desc="ipu_hw",
        weights_to_host_on_exit=False,
    )
    logging.info("PopXL compilation complete")
    return session


def main():
    """Run a benchmark configuration"""
    config, _, _ = llama_config_setup(CONFIG_DIR / "inference.yml", "release", "llama2_70b_pod16", hf_model_setup=False)
    config.execution.use_cache = True

    session = inference(config)

    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    with session:
        # Skip one result
        session.run(inputs)

        durations = []
        for _ in range(5):
            start = time.time()
            session.run(inputs)
            durations.append(time.time() - start)
    duration = np.mean(durations)

    samples_per_step = config.execution.micro_batch_size
    result_str = f"Duration: {duration} s " f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    main()
