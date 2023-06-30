# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
import time
from functools import partial

import numpy as np
import popdist

import popxl
from popxl import ops
from math import ceil
from modelling.mnli import GPTMnliHead

import popxl_addons as addons
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.remote import named_variable_buffers, load_remote_graph
from popxl_addons.utils import timer
from popxl_addons.task_session import TaskSession

from config import CONFIG_DIR, GPTConfig
from modelling.embedding import GPTEmbeddingsTP, generate_positions
from modelling.decoder import GPTDecoderBlockTP
from utils.setup import gpt_config_setup

__all__ = ["mnli_inference"]


def mnli_inference(config: GPTConfig) -> TaskSession:
    assert config.model.eval, "Eval mode must be True"
    assert config.execution.data_parallel == 1, "You can't use DP for inference"
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
            input_shape = (config.execution.micro_batch_size * config.model.sequence_length,)
            input_streams = addons.InputStreams(
                words=(input_shape, popxl.int32),
                unpadded_length=((config.execution.micro_batch_size,), popxl.int32),
            )
            output_streams = addons.OutputStreams(
                logits=((config.execution.micro_batch_size, config.inference.mnli_n_classes), config.model.dtype),
            )

            positions = popxl.constant(generate_positions(config), popxl.int32, name="positions")

            # ----- Build compute graphs -----

            embeddings_facts, embeddings_graph = GPTEmbeddingsTP(config).create_graph(
                input_streams.words.spec, positions.spec
            )
            layer_facts, layer_graph = GPTDecoderBlockTP(config).create_graph(embeddings_graph.graph.outputs[0])
            # Tied embedding is copied to weight of GPTLMHeadTP
            head_facts, head_graph = GPTMnliHead(config).create_graph(
                layer_graph.graph.outputs[0], input_streams.unpadded_length.spec
            )
            # ---- Transform graphs ----

            addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

            # ----- Create Variables -----

            # Create RemoteBuffers for each variable
            embeddings_buffers = named_variable_buffers(embeddings_facts, shard_over_dict=False)
            layer_buffers = named_variable_buffers(layer_facts, entries=config.model.layers, shard_over_dict=False)
            head_buffers = named_variable_buffers(head_facts, shard_over_dict=False)

            variables = NamedTensors()
            transformer = NamedTensors()
            variables.insert("transformer", transformer)
            transformer.insert(
                # Do not use empty=True with embedding as contains offset which doesn't get overwritten
                "embeddings",
                embeddings_facts.init_remote(embeddings_buffers, 0, "embeddings", empty=False),
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
            # Do not use empty=True with head as contains offset which doesn't get overwritten
            # Note use `head.head` namespace to match training
            variables.insert("head.head", head_facts.init_remote(head_buffers, 0, "head.head", empty=False))

            # ---- Execute ----
            with popxl.in_sequence():
                word = ops.host_load(input_streams.words)
                unpadded_length = ops.host_load(input_streams.unpadded_length)
                # Embeddings
                load_graph, names = load_remote_graph(embeddings_buffers)
                embedding_vars = NamedTensors.pack(names, load_graph.call(0))
                (x,) = embeddings_graph.bind(embedding_vars).call(word, positions)

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
                # Debugging note: last layer output will not correspond to HF we include the last LayerNorm ln_f within the head

                # Squad head
                load_graph, names = load_remote_graph(head_buffers)
                head_vars = NamedTensors.pack(names, load_graph.call(0))
                (logits,) = head_graph.bind(head_vars).call(x, unpadded_length)
                ops.host_store(output_streams.logits, logits)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

    ir.num_host_transfers = config.execution.device_iterations

    session = TaskSession(inputs=input_streams, outputs=output_streams, state=variables, ir=ir, device_desc="ipu_hw")
    return session


def main():
    """Run a benchmark configuration"""
    config, *_ = gpt_config_setup(
        CONFIG_DIR / "inference.yml", "release", "gpt2_small", wandb_setup=False, hf_model_setup=False
    )
    session = mnli_inference(config)
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
    try:
        main()
    except Exception as e:
        logging.exception(e, exc_info=False)  # Log time of exception
        raise
