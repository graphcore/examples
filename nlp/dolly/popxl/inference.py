# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
This file defines the PopXL inference session that will be executed.  This
example makes use of phased execution to avoid loading the entire 12 billion
parameter model into memory at one time, loading one transformer block at a time
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

import popxl_addons as addons
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.remote import named_variable_buffers, load_remote_graph
from popxl_addons.utils import timer
from popxl_addons.task_session import TaskSession

from config import CONFIG_DIR, DollyConfig
from modelling.embedding import DollyEmbeddingsTP
from modelling.decoder import DollyDecoderBlockTP
from modelling.dolly_lm import DollyLMHeadTP, gather_logits_tp
from utils.setup import dolly_config_setup

__all__ = ["inference"]


def inference(config: DollyConfig) -> TaskSession:
    assert config.execution.data_parallel == 1, "You can't use data-parallelism for inference"
    replicas = config.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas

    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"

    with timer("PopXL IR construction"):
        with ir.main_graph:
            # -----  Define input and output streams -----
            shard_size = ceil(config.model.embedding.vocab_size / config.execution.tensor_parallel)
            input_shape = (
                config.execution.micro_batch_size,
                config.model.sequence_length,
            )
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

            embeddings_facts, embeddings_graph = DollyEmbeddingsTP(config).create_graph(input_streams.words.spec)
            layer_facts, layer_graph = DollyDecoderBlockTP(config).create_graph(*embeddings_graph.graph.outputs)
            lm_facts, lm_graph = DollyLMHeadTP(config).create_graph(layer_graph.graph.outputs[0])
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

                def layer(x, n):
                    load_graph, names = load_remote_graph(layer_buffers)
                    layer_vars = NamedTensors.pack(names, load_graph.call(n))
                    (x,) = layer_graph.bind(layer_vars).call(x)
                    return x, n + 1

                i = popxl.constant(0, name="layer_index")
                layers_graph = ir.create_graph(layer, x, i)
                x, _ = ops.repeat(layers_graph, config.model.layers, x, i)

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
    from modelling.hf_mapping import hf_mapping_lm_tp

    """Run a benchmark configuration"""
    config, _, _ = dolly_config_setup(CONFIG_DIR / "inference.yml", "release", "tiny", hf_model_setup=False)
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
