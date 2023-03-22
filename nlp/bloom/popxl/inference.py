# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
import time
from typing import Dict

import numpy as np
import popdist
import popxl_addons as addons
from popxl_addons import TaskSession
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.remote import NamedRemoteBuffers, create_remote_buffer, load_remote_graph, named_variable_buffers
from popxl_addons.utils import timer
from popxl_addons.transforms.use_fc_pass import disable_fc_pass

import popxl
from config import CONFIG_DIR, BloomConfig
from modelling.bloom_lm import BloomLMHeadTP2D, gather_logits_tp
from modelling.decoder import BloomDecoderBlockTP2D
from modelling.embedding import BloomEmbeddingTP2D
from popxl import ops
from utils.setup import bloom_config_setup

__all__ = ["inference"]


def inference(config: BloomConfig) -> TaskSession:
    replicas = config.execution.tensor_parallel_1 * config.execution.tensor_parallel_2

    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas

    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"
    opts.engineOptions["target.extendedMemory"] = "true"

    t = time.time()
    main = ir.main_graph

    with timer("PopXL IR construction"), main:
        # -----  Define input and output streams -----
        input_shape = (config.model.sequence_length,)
        input_streams = addons.InputStreams(words=(input_shape, popxl.int32), last_token_indices=((), popxl.int32))
        output_streams = addons.OutputStreams(
            next_token_logits=((config.model.embedding.vocab_size,), config.model.dtype)
        )

        embedding_fact, embedding_graph = BloomEmbeddingTP2D(config).create_graph(input_streams.words.spec)
        decoder_fact, decoder_graph = BloomDecoderBlockTP2D(config).create_graph(embedding_graph.graph.outputs[0])
        head_fact, head_graph = BloomLMHeadTP2D(config).create_graph(
            decoder_graph.graph.outputs[0],
            embedding_graph.args.weight_1,
            embedding_graph.args.weight_2,
        )

        # Available Memory Proportion
        addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

        # Disable `useFullyConnectedPass` Poplar option
        if config.execution.disable_fc_pass:
            disable_fc_pass(ir)

        embedding_buffers = named_variable_buffers(embedding_fact, shard_over_dict=False)
        decoder_buffers = named_variable_buffers(decoder_fact, entries=config.model.layers, shard_over_dict=False)
        head_buffers = named_variable_buffers(head_fact, shard_over_dict=False)

        # ----- Create Variables -----

        variables = NamedTensors()
        transformer = NamedTensors()
        transformer.insert(
            "embedding",
            embedding_fact.init_remote(
                embedding_buffers,
                0,
                "embedding",
                empty=True,
                memmap_dir=config.execution.memmap_dir,
            ),
        )

        transformer.insert(
            "decoder",
            NamedTensors.from_dict(
                {
                    n: decoder_fact.init_remote(
                        decoder_buffers,
                        n,
                        f"decoder.{n}",
                        empty=True,
                        memmap_dir=config.execution.memmap_dir,
                    )
                    for n in range(config.model.layers)
                }
            ),
        )
        variables.insert("transformer", transformer)
        variables.insert(
            "head",
            head_fact.init_remote(
                head_buffers,
                0,
                "head",
                empty=True,
                memmap_dir=config.execution.memmap_dir,
            ),
        )

        # ---- Execute ----
        with popxl.in_sequence():
            words = ops.host_load(input_streams.words)

            last_token_indices = ops.host_load(input_streams.last_token_indices)

            def embedding_phase(x: popxl.Tensor):
                # Load Embedding layer
                load_graph, names = load_remote_graph(embedding_buffers)
                embedding_vars = NamedTensors.pack(names, load_graph.call(0))
                # Forward
                (x,) = embedding_graph.bind(embedding_vars).call(x)
                return x

            embed_graph = ir.create_graph(embedding_phase, words)
            (x,) = ops.call(embed_graph, words)

            def decoder_block_phase(x, n: popxl.Tensor):
                # Load decoder block
                load_graph, names = load_remote_graph(decoder_buffers)
                layer_vars = NamedTensors.pack(names, load_graph.call(n))
                # Forward
                (x,) = decoder_graph.bind(layer_vars).call(x)
                return x, n + 1

            i = popxl.constant(0, name="layer_index")
            decoder_graph = ir.create_graph(decoder_block_phase, x, i)
            x, _ = ops.repeat(decoder_graph, config.model.layers, x, i)

            def head_phase(x):
                load_graph, names = load_remote_graph(head_buffers)
                layer_vars = NamedTensors.pack(names, load_graph.call(0))

                # Embedding weights are split into two pieces. There is a maximum
                # size for a single transfer in poplar, which the full embedding
                # would exceed.
                (x,) = head_graph.bind(layer_vars).call(
                    x,
                    ops.remote_load(embedding_buffers.weight_1, 0),
                    ops.remote_load(embedding_buffers.weight_2, 0),
                )
                return x

            head_graph = ir.create_graph(head_phase, x)
            (logits,) = ops.call(head_graph, x)

            next_token_logits = gather_logits_tp(config, logits, last_token_indices)

            ops.host_store(
                output_streams.next_token_logits,
                next_token_logits.reshape_(output_streams.next_token_logits.shape),
            )

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level="default")

    logging.info(f"popxl IR construction duration: {(time.time() - t) / 60:.2f} mins")

    session = TaskSession(
        input_streams, output_streams, variables, ir=ir, device_desc="ipu_hw", weights_to_host_on_exit=False
    )

    return session


def main():
    """Run a benchmark configuration"""
    config, *_ = bloom_config_setup(
        CONFIG_DIR / "inference.yml",
        "release",
        "bloom_560M",
        hf_model_setup=False,
    )

    session = inference(config)
    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    with session:
        # Skip one result
        session.run(inputs)

        durations = []
        for _ in range(10):
            start = time.time()
            session.run(inputs)
            durations.append(time.time() - start)
    duration = np.mean(durations)

    result_str = f"Duration: {duration} s " f"Throughput: {1/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
