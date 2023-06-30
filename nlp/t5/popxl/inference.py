# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
import time

import numpy as np
import popdist

import popxl
from popxl import ops

import popxl_addons as addons
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.remote import named_variable_buffers, load_remote_graph, NamedRemoteBuffers
from popxl_addons.utils import timer
from popxl_addons.task_session import TaskSession

from config import T5Config, CONFIG_DIR
from utils.setup import t5_config_setup
from modelling.embedding import T5EmbeddingsTP, T5DecoderEmbeddingsTP
from modelling.encoder import T5EncoderBlockTP, T5EncoderHead
from modelling.decoder import T5DecoderBlockTP
from modelling.t5_lm import T5LMHeadTP, generate_greedy_tp


__all__ = ["inference"]


def inference(config: T5Config) -> TaskSession:
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
                attention_mask=(input_shape, config.model.dtype),
                decoder_words=(input_shape, popxl.int32),
                decoder_attention_mask=(input_shape, config.model.dtype),
                last_token_indices=((config.execution.micro_batch_size,), popxl.int32),
            )
            output_streams = addons.OutputStreams(next_token=((config.execution.micro_batch_size,), popxl.int32))

            # ----- Build compute graphs -----
            embeddings_facts, embeddings_graph = T5EmbeddingsTP(config).create_graph(input_streams.words.spec)
            first_enc_layer_facts, first_enc_layer_graph = T5EncoderBlockTP(config).create_graph(
                embeddings_graph.graph.outputs[0].spec, input_streams.attention_mask.spec
            )
            enc_layer_facts, enc_layer_graph = T5EncoderBlockTP(config).create_graph(
                first_enc_layer_graph.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
                first_enc_layer_graph.args.attention.heads.rel_pos_embedding.weight.spec,
            )
            enc_head_facts, enc_head_graph = T5EncoderHead(config).create_graph(enc_layer_graph.graph.outputs[0].spec)
            dec_embeddings_facts, dec_embeddings_graph = T5DecoderEmbeddingsTP(config).create_graph(
                input_streams.decoder_words.spec, embeddings_graph.args.word.weight.spec
            )
            first_dec_layer_facts, first_dec_layer_graph = T5DecoderBlockTP(config).create_graph(
                dec_embeddings_graph.graph.outputs[0].spec,
                input_streams.decoder_attention_mask.spec,
                enc_head_graph.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
            )
            dec_layer_facts, dec_layer_graph = T5DecoderBlockTP(config).create_graph(
                first_dec_layer_graph.graph.outputs[0].spec,
                input_streams.decoder_attention_mask.spec,
                enc_head_graph.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
                first_dec_layer_graph.args.attention.heads.rel_pos_embedding.weight.spec,
            )
            lm_facts, lm_graph = T5LMHeadTP(config).create_graph(dec_layer_graph.graph.outputs[0])

            # ---- Transform graphs ----
            addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

            # ----- Create Variables -----

            # Create RemoteBuffers for each variable
            embeddings_buffers = named_variable_buffers(embeddings_facts, shard_over_dict=False)
            first_enc_layer_buffers = named_variable_buffers(first_enc_layer_facts, shard_over_dict=False)
            enc_layer_buffers = named_variable_buffers(
                enc_layer_facts, entries=config.model.layers - 1, shard_over_dict=False
            )
            enc_head_buffers = named_variable_buffers(enc_head_facts, shard_over_dict=False)
            # Note that the decoder embedding has no variables of its own
            dec_embeddings_buffers = NamedRemoteBuffers.from_dict({"weight": embeddings_buffers.word.weight})
            first_dec_layer_buffers = named_variable_buffers(first_dec_layer_facts, shard_over_dict=False)
            dec_layer_buffers = named_variable_buffers(
                dec_layer_facts, entries=config.model.layers - 1, shard_over_dict=False
            )
            lm_buffers = named_variable_buffers(lm_facts, shard_over_dict=False)

            variables = NamedTensors()
            transformer = NamedTensors()
            transformer.insert(
                "embeddings", embeddings_facts.init_remote(embeddings_buffers, 0, "embeddings", empty=True)
            )
            transformer.insert(
                "encoder.0", first_enc_layer_facts.init_remote(first_enc_layer_buffers, 0, "encoder.0", empty=True)
            )
            for n in range(1, config.model.layers):
                transformer.insert(
                    f"encoder.{n}",
                    enc_layer_facts.init_remote(enc_layer_buffers, n - 1, f"encoder.{n}", empty=True),
                    overwrite=True,
                )
            transformer.insert(
                "encoder_head", enc_head_facts.init_remote(enc_head_buffers, 0, "encoder_head", empty=True)
            )
            transformer.insert(
                "decoder.0", first_dec_layer_facts.init_remote(first_dec_layer_buffers, 0, "decoder.0", empty=True)
            )
            for n in range(1, config.model.layers):
                transformer.insert(
                    f"decoder.{n}",
                    dec_layer_facts.init_remote(dec_layer_buffers, n - 1, f"decoder.{n}", empty=True),
                    overwrite=True,
                )
            variables.insert("transformer", transformer)
            variables.insert("lm_head", lm_facts.init_remote(lm_buffers, 0, "lm_head", empty=True))

            # ---- Execute ----
            with popxl.in_sequence():
                word = ops.host_load(input_streams.words)
                attention_mask = ops.host_load(input_streams.attention_mask)
                decoder_word = ops.host_load(input_streams.decoder_words)
                decoder_attention_mask = ops.host_load(input_streams.decoder_attention_mask)
                last_token_indices = ops.host_load(input_streams.last_token_indices)

                # Embeddings
                load_graph, names = load_remote_graph(embeddings_buffers)
                embedding_vars = NamedTensors.pack(names, load_graph.call(0))
                (x,) = embeddings_graph.bind(embedding_vars).call(word)

                # Encoder
                # First layer
                load_graph, names = load_remote_graph(first_enc_layer_buffers)
                layer_vars = NamedTensors.pack(names, load_graph.call(0))
                (x,) = first_enc_layer_graph.bind(layer_vars).call(x, attention_mask)

                # Following layers
                def enc_layer(x, n, attention_mask, rel_pos_weight):
                    load_graph, names = load_remote_graph(enc_layer_buffers)
                    layer_vars = NamedTensors.pack(names, load_graph.call(n))
                    (x,) = enc_layer_graph.bind(layer_vars).call(x, attention_mask, rel_pos_weight)
                    return x, n + 1

                i = popxl.constant(0, name="layer_index")
                # Pass the shared weight to the other layers
                rel_pos_weight = layer_vars["attention.heads.rel_pos_embedding.weight"]
                layers_graph = ir.create_graph(enc_layer, x, i, attention_mask, rel_pos_weight)
                x, _ = ops.repeat(layers_graph, config.model.layers - 1, x, i, attention_mask, rel_pos_weight)

                # Encoder head
                load_graph, names = load_remote_graph(enc_head_buffers)
                layer_vars = NamedTensors.pack(names, load_graph.call(0))
                (x,) = enc_head_graph.bind(layer_vars).call(x)

                # Decoder embeddings
                # Get the embedding weight from the encoder embedding layer
                load_graph, names = load_remote_graph(dec_embeddings_buffers)
                embedding_vars = NamedTensors.pack(names, load_graph.call(0))
                embedding_weight_t = embedding_vars.pop("weight")
                (x_dec,) = embeddings_graph.bind(embedding_vars).call(decoder_word, embedding_weight_t)

                # Decoder
                # First layer
                load_graph, names = load_remote_graph(first_dec_layer_buffers)
                layer_vars = NamedTensors.pack(names, load_graph.call(0))
                (x_dec,) = first_dec_layer_graph.bind(layer_vars).call(x_dec, decoder_attention_mask, x, attention_mask)

                # Following layers
                def dec_layer(x_dec, n, decoder_attention_mask, x, attention_mask, rel_pos_weight):
                    load_graph, names = load_remote_graph(dec_layer_buffers)
                    layer_vars = NamedTensors.pack(names, load_graph.call(n))
                    (x_dec,) = dec_layer_graph.bind(layer_vars).call(
                        x_dec, decoder_attention_mask, x, attention_mask, rel_pos_weight
                    )
                    return x_dec, n + 1

                i = popxl.constant(0, name="layer_index")
                # Pass the shared weight to the other layers
                rel_pos_weight = layer_vars["attention.heads.rel_pos_embedding.weight"]
                layers_graph = ir.create_graph(
                    dec_layer, x_dec, i, decoder_attention_mask, x, attention_mask, rel_pos_weight
                )
                x, _ = ops.repeat(
                    layers_graph,
                    config.model.layers - 1,
                    x_dec,
                    i,
                    decoder_attention_mask,
                    x,
                    attention_mask,
                    rel_pos_weight,
                )

                # LM head
                load_graph, names = load_remote_graph(lm_buffers)
                lm_vars = NamedTensors.pack(names, load_graph.call(0))
                (logits,) = lm_graph.bind(lm_vars).call(x)
                next_token_id = generate_greedy_tp(config, logits, last_token_indices)
                ops.host_store(output_streams.next_token, next_token_id.reshape_(output_streams.next_token.shape))

        # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
        apply_pre_alias_patterns(ir, level="default")

    ir.num_host_transfers = config.execution.device_iterations

    session = TaskSession(
        inputs=input_streams,
        outputs=output_streams,
        state=NamedTensors(fwd=variables),
        ir=ir,
        device_desc="ipu_hw",
        weights_to_host_on_exit=False,
    )
    return session


def main():
    """Run a benchmark configuration"""
    config, *_ = t5_config_setup(
        CONFIG_DIR / "inference.yml", "release", "xxl", wandb_setup=False, hf_model_setup=False
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
        for _ in range(5):
            start = time.perf_counter()
            session.run(inputs)
            durations.append(time.perf_counter() - start)
    duration = np.mean(durations)

    samples_per_step = config.execution.micro_batch_size
    result_str = f"Duration: {duration} s " f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
