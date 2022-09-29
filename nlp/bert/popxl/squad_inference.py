# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import time
import numpy as np

import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons import TaskSession
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.remote import (
    named_variable_buffers,
    load_remote_graph
)

from config import CONFIG_DIR, BertConfig
from utils.setup import bert_config_setup
from modelling.embedding import BertEmbeddings
from modelling.bert_model import BertLayer
from modelling.squad import BertSquadHead


__all__ = ["squad_inference_phased"]


def squad_inference_phased(config: BertConfig) -> TaskSession:
    config.model.eval = True
    assert config.execution.data_parallel == 1

    ir = popxl.Ir()
    ir.replication_factor = config.execution.data_parallel

    t = time.time()
    with ir.main_graph:
        # -----  Define input and output streams -----
        input_shape = (
            config.execution.micro_batch_size * config.model.sequence_length,)
        input_streams = [
            popxl.h2d_stream(input_shape, popxl.uint32, name="words"),
            popxl.h2d_stream(input_shape, popxl.uint32, name="token_type"),
            popxl.h2d_stream(input_shape, config.model.dtype, name="mask"),
        ]
        output_streams = [
            popxl.d2h_stream((config.execution.micro_batch_size,
                              config.model.sequence_length, 2), config.model.dtype, name="output")
        ]

        # ----- Build compute graphs -----

        embeddings_args, embeddings_graph = BertEmbeddings(
            config).create_graph(input_streams[0].spec, input_streams[1].spec)
        layer_args, layer_graph = BertLayer(config).create_graph(*embeddings_graph.graph.outputs, input_streams[2].spec)
        squad_args, squad_graph = BertSquadHead(
            config).create_graph(*embeddings_graph.graph.outputs)

        # ---- Transform graphs ----

        addons.set_available_memory_proportion_by_ipu(
            ir, config.execution.available_memory_proportion)

        # ----- Create Variables -----

        # Create RemoteBuffers for each variable
        embeddings_buffers = named_variable_buffers(embeddings_args)
        squad_buffers = named_variable_buffers(squad_args)
        layer_buffers = named_variable_buffers(layer_args, config.model.layers)

        variables = NamedTensors()
        variables.insert("embeddings", embeddings_args.init_remote(
            embeddings_buffers, 0, "embeddings"))
        variables.insert("squad", squad_args.init_remote(
            squad_buffers, 0, "squad"))
        variables.insert("layer", NamedTensors.from_dict({
            n: layer_args.init_remote(layer_buffers, n, f"layer.{n}")
            for n in range(config.model.layers)
        }))

        # ---- Execute ----
        with popxl.in_sequence():
            with popxl.transforms.merge_exchange(), popxl.in_sequence(False):
                word, ttype, mask = (ops.host_load(s) for s in input_streams)

            load_graph, names = load_remote_graph(embeddings_buffers)
            embedding_vars = NamedTensors.pack(names, load_graph.call(0))
            x, = embeddings_graph.bind(embedding_vars).call(word, ttype)

            load_graph, names = load_remote_graph(layer_buffers)
            for n in range(config.model.layers):
                layer_vars = NamedTensors.pack(names, load_graph.call(n))
                x, = layer_graph.bind(layer_vars).call(x, mask)

            load_graph, names = load_remote_graph(squad_buffers)
            squad_vars = NamedTensors.pack(names, load_graph.call(0))
            out, = squad_graph.bind(squad_vars).call(x)

            ops.host_store(output_streams[0], out.reshape_(
                output_streams[0].shape))

    logging.info(f"popxl IR construction duration: {(time.time() - t) / 60:.2f} mins")

    ir.num_host_transfers = config.execution.device_iterations

    session = TaskSession(
        input_streams,
        output_streams,
        variables,
        ir,
        "ipu_hw")
    return session


def main():
    """Run a benchmark configuration"""
    config, _ = bert_config_setup(
        CONFIG_DIR / "squad_inference.yml",
        "phased",
        "large")
    session = squad_inference_phased(config)

    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()}

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
    result_str = \
        f"Duration: {duration} s " \
        f"throughput: {samples_per_step/duration:6.1f} samples/sec "
    logging.info(result_str)


if __name__ == "__main__":
    main()
