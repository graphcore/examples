# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
import time
from functools import partial

import numpy as np
import popdist

import popxl
from popxl import ops
from math import ceil

import popxl_addons as addons
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.remote import named_variable_buffers, load_remote_graph, NamedRemoteBuffers
from popxl_addons.rts import all_gather_replica_sharded_graph
from popxl_addons.utils import timer
from popxl_addons.task_session import TaskSession
from popxl_addons.performance_utils import total_FLOPs
from configs import CONFIG_DIR, MagmaConfig

from modelling.image_prefix import ImagePrefix
from modelling.gptj.embedding import GPTJEmbeddingsTP
from modelling.gptj.decoder import GPTJDecoderBlockTP
from modelling.gptj.gptj_lm import GPTJLMHeadTP
from utils.setup import magma_config_setup

__all__ = ["inference"]


def get_image_prefix_buffers(config, facts):
    buffers = NamedRemoteBuffers(
        proj=named_variable_buffers(facts.proj, shard_over_dict=False),
        ln=named_variable_buffers(facts.ln, shard_over_dict=False),
    )
    encoder_buffers = NamedRemoteBuffers(stem=named_variable_buffers(facts.enc.stem, shard_over_dict=False))
    for i in range(1, len(config.layers) + 1):
        layer_facts = facts.enc.layer[i]
        encoder_buffers.insert(
            f"layer.{i}.{0}", named_variable_buffers(layer_facts[0], shard_over_dict=False), overwrite=True
        )
        encoder_buffers.insert(
            f"layer.{i}.block",
            named_variable_buffers(layer_facts[1], entries=config.layers[i - 1] - 1, shard_over_dict=False),
            overwrite=True,
        )
    buffers.insert("enc", encoder_buffers)
    return buffers


def init_image_prefix_vars(facts, buffers, config):
    nts = NamedTensors(
        proj=facts.proj.init_remote(buffers.proj, empty=True), ln=facts.ln.init_remote(buffers.ln, empty=True)
    )
    ts = {"stem": facts.enc.stem.init_remote(buffers.enc.stem, empty=True)}
    for i in range(1, len(config.layers) + 1):
        layer_facts = facts.enc.layer[i]
        layer_buffers = buffers.enc.layer[i]
        ts.update({f"layer.{i}.{0}": layer_facts[0].init_remote(layer_buffers[0], empty=True)})
        blocks_ts = {
            f"layer.{i}.{n}": layer_facts[1].init_remote(layer_buffers.block, n - 1, f"layer.{i}.{n}", empty=True)
            for n in range(1, config.layers[i - 1])
        }
        ts.update(blocks_ts)
    nts.insert("enc", NamedTensors.from_dict(ts))
    return nts


def load_image_prefix_vars(buffers, config):
    # load everything that is not in layers
    buffs = NamedRemoteBuffers(proj=buffers.proj, ln=buffers.ln)
    buffs.insert("enc.stem", buffers.enc.stem)
    # load layers from shared buffers
    g1, names = load_remote_graph(buffs)
    nts = {name: t for name, t in zip(names, g1.call(0))}
    for i in range(1, len(config.layers) + 1):
        layer_buff = buffers.enc.layer[i]
        g, names = load_remote_graph(layer_buff[0])
        nts.update({f"enc.layer.{i}.{0}.{name}": t for name, t in zip(names, g.call(0))})
        g_repeated, names = load_remote_graph(layer_buff.block, config.layers[i - 1] - 1)
        for j in range(1, config.layers[i - 1]):
            block_nts = {f"enc.layer.{i}.{j}.{name}": t for name, t in zip(names, g_repeated.call(j - 1))}
            nts.update(block_nts)
    return NamedTensors.from_dict(nts)


def inference(config: MagmaConfig) -> TaskSession:
    replicas = config.transformer.execution.tensor_parallel
    ir = popxl.Ir(replication="popdist" if popdist.isPopdistEnvSet() else replicas)
    assert ir.replication_factor == replicas
    assert config.transformer.execution.micro_batch_size == 1
    # Options
    opts = ir._pb_ir.getSessionOptions()
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"

    gptj_config = config.transformer
    resnet_config = config.visual

    with timer("PopXL IR construction"):
        with ir.main_graph:
            # -----  Define input and output streams -----
            shard_size = ceil(gptj_config.embedding.vocab_size / gptj_config.execution.tensor_parallel)
            word_input_shape = (gptj_config.execution.micro_batch_size * gptj_config.sequence_length,)
            image_input_shape = (
                resnet_config.execution.micro_batch_size,
                3,
                resnet_config.image_resolution,
                resnet_config.image_resolution,
            )
            logits_shape = (gptj_config.execution.micro_batch_size * (gptj_config.sequence_length + 144), shard_size)

            input_streams = addons.InputStreams(
                words=(word_input_shape, popxl.int32),
                image=(image_input_shape, resnet_config.dtype),
            )
            output_streams = addons.OutputStreams(logits=(logits_shape, popxl.float16))

            # ----- Build compute graphs -----

            image_prefix_facts, image_prefix_graph = ImagePrefix(config).create_graph(input_streams.image.spec)
            image_seq_len = image_prefix_graph.graph.outputs[0].shape[1]
            gptj_config.sequence_length += image_seq_len
            decoder_input_spec = popxl.TensorSpec(
                (gptj_config.execution.micro_batch_size * gptj_config.sequence_length, gptj_config.hidden_size),
                gptj_config.dtype,
            )
            embeddings_facts, embeddings_graph = GPTJEmbeddingsTP(gptj_config).create_graph(input_streams.words.spec)
            layer_facts, layer_graph = GPTJDecoderBlockTP(gptj_config).create_graph(decoder_input_spec)
            lm_facts, lm_graph = GPTJLMHeadTP(gptj_config).create_graph(layer_graph.graph.outputs[0])

            # ---- Transform graphs ----

            gptj_amp = gptj_config.execution.available_memory_proportion
            resnet_amp = resnet_config.execution.available_memory_proportion
            addons.set_graph_available_memory_proportion_by_ipu(embeddings_graph.graph, gptj_amp)
            addons.set_graph_available_memory_proportion_by_ipu(layer_graph.graph, gptj_amp)
            addons.set_graph_available_memory_proportion_by_ipu(lm_graph.graph, gptj_amp)
            addons.set_graph_available_memory_proportion_by_ipu(image_prefix_graph.graph, resnet_amp)

            # ----- Create Variables -----

            # Create RemoteBuffers for each variable
            image_prefix_buffers = get_image_prefix_buffers(config.visual, image_prefix_facts)
            embeddings_buffers = named_variable_buffers(embeddings_facts, shard_over_dict=False)
            layer_buffers = named_variable_buffers(layer_facts, entries=gptj_config.layers, shard_over_dict=False)
            lm_buffers = named_variable_buffers(lm_facts, shard_over_dict=False)

            variables = NamedTensors(
                image_prefix=init_image_prefix_vars(image_prefix_facts, image_prefix_buffers, config.visual)
            )
            transformer = NamedTensors()
            transformer.insert(
                "embeddings", embeddings_facts.init_remote(embeddings_buffers, 0, "embeddings", empty=True)
            )

            transformer.insert(
                "decoder",
                NamedTensors.from_dict(
                    {
                        n: layer_facts.init_remote(layer_buffers, n, f"decoder.{n}", empty=True)
                        for n in range(gptj_config.layers)
                    }
                ),
            )
            variables.insert("transformer", transformer)
            variables.insert("lm_head", lm_facts.init_remote(lm_buffers, 0, "lm_head", empty=True))

            # ---- Execute ----
            with popxl.in_sequence():
                word = ops.host_load(input_streams.words)
                image = ops.host_load(input_streams.image)

                # ---- Concat visual emebedding and text embedding ----
                def embed():
                    # Load all variables
                    bs = gptj_config.execution.micro_batch_size
                    hidden_size = gptj_config.hidden_size

                    # resnet
                    image_embed_vars = load_image_prefix_vars(image_prefix_buffers, config.visual)
                    (image_embed,) = image_prefix_graph.bind(image_embed_vars).call(image)

                    # Embeddings
                    load_graph, names = load_remote_graph(embeddings_buffers)
                    text_embed_vars = NamedTensors.pack(names, load_graph.call(0))
                    (text_embed,) = embeddings_graph.bind(text_embed_vars).call(word)
                    text_embed = text_embed.reshape((bs, -1, hidden_size))

                    # always image embed first
                    embed = ops.concat([image_embed, text_embed], axis=1)
                    embed = embed.reshape((-1, hidden_size))
                    return embed

                x = embed()
                # ---- Run LM with adapters ----

                # Decoder

                def layer(x, n):
                    load_graph, names = load_remote_graph(layer_buffers)
                    layer_vars = NamedTensors.pack(names, load_graph.call(n))
                    (x,) = layer_graph.bind(layer_vars).call(x)
                    return x, n + 1

                i = popxl.constant(0, name="layer_index")
                layers_graph = ir.create_graph(layer, x, i)
                x, _ = ops.repeat(layers_graph, gptj_config.layers, x, i)

                # LM head
                load_graph, names = load_remote_graph(lm_buffers)
                squad_vars = NamedTensors.pack(names, load_graph.call(0))
                (logits,) = lm_graph.bind(squad_vars).call(x)

                ops.host_store(output_streams.logits, logits)

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

    ir.num_host_transfers = 1

    session = TaskSession(
        inputs=input_streams, outputs=output_streams, state=NamedTensors(fwd=variables), ir=ir, device_desc="ipu_hw"
    )
    return session


def main():
    """Run a benchmark configuration"""
    config, *_ = magma_config_setup(CONFIG_DIR / "inference.yml", "release", "magma_v1_1024")
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

    TFLOPS = total_FLOPs(session.ir) / 10e12
    samples_per_step = config.transformer.execution.micro_batch_size
    result_str = (
        f"Duration: {duration:6.2f} s "
        f"Throughput: {samples_per_step/duration:6.1f} samples/s "
        f"TFLOPS/s: {TFLOPS/duration:6.2f} "
    )
    logging.info(result_str)


if __name__ == "__main__":
    main()
