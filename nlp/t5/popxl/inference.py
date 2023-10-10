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
from popxl_addons.remote import named_variable_buffers, load_remote_graph
from popxl_addons.utils import timer
from popxl_addons.task_session import TaskSession

from config import T5Config, CONFIG_DIR
from utils.setup import t5_config_setup
from modelling.embedding import T5EmbeddingsTP, T5DecoderEmbeddingsTP
from modelling.encoder_decoder import T5BlockTP, T5EncoderHead
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
            dec_embeddings_facts, dec_embeddings_graph = T5DecoderEmbeddingsTP(config).create_graph(
                input_streams.decoder_words.spec, embeddings_graph.args.word.weight.spec
            )
            scale_spec = popxl.TensorSpec((), config.model.dtype)
            t5_block_facts, t5_block_graph = T5BlockTP(config).create_graph(
                embeddings_graph.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
                embeddings_graph.graph.outputs[0].spec,
                input_streams.attention_mask.spec,
                scale_spec,
                embeddings_graph.args.rel_pos_weight.spec,
            )
            enc_head_facts, enc_head_graph = T5EncoderHead(config).create_graph(t5_block_graph.graph.outputs[0].spec)
            lm_facts, lm_graph = T5LMHeadTP(config).create_graph(t5_block_graph.graph.outputs[0].spec)

            # ---- Transform graphs ----
            addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

            # ----- Create Variables -----

            # Create RemoteBuffers for each variable
            embeddings_buffers = named_variable_buffers(embeddings_facts, shard_over_dict=False)
            t5_block_buffers = named_variable_buffers(
                t5_block_facts, entries=2 * config.model.layers, shard_over_dict=False
            )
            enc_head_buffers = named_variable_buffers(enc_head_facts, shard_over_dict=False)
            dec_embeddings_buffers = named_variable_buffers(dec_embeddings_facts, shard_over_dict=False)
            lm_buffers = named_variable_buffers(lm_facts, shard_over_dict=False)

            variables = NamedTensors()
            transformer = NamedTensors()
            transformer.insert(
                "embeddings", embeddings_facts.init_remote(embeddings_buffers, 0, "embeddings", empty=True)
            )
            transformer.insert(
                "decoder_embeddings",
                dec_embeddings_facts.init_remote(dec_embeddings_buffers, 0, "decoder_embeddings", empty=True),
            )
            for n in range(2 * config.model.layers):
                name = "encoder" if n < config.model.layers else "decoder"
                idx = n % config.model.layers
                transformer.insert(
                    f"{name}.{idx}",
                    t5_block_facts.init_remote(t5_block_buffers, n, f"{name}.{idx}", empty=True),
                    overwrite=True,
                )
            transformer.insert(
                "encoder_head", enc_head_facts.init_remote(enc_head_buffers, 0, "encoder_head", empty=True)
            )
            variables.insert("transformer", transformer)
            variables.insert("lm_head", lm_facts.init_remote(lm_buffers, 0, "lm_head", empty=True))

            # The decoder embedding uses also the embedding weights
            dec_embeddings_buffers.insert("weight", embeddings_buffers.word.weight)

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
                rel_pos_weight = embedding_vars.rel_pos_weight

                # Encoder
                def t5_block(n, x, attention_mask, enc_out, enc_attention_mask, scale, rel_pos_weight):
                    load_graph, names = load_remote_graph(t5_block_buffers)
                    layer_vars = NamedTensors.pack(names, load_graph.call(n))
                    (x,) = t5_block_graph.bind(layer_vars).call(
                        x, attention_mask, enc_out, enc_attention_mask, scale, rel_pos_weight
                    )
                    return n + 1, x

                # For encoder layers, i is in [0, N-1]
                i = popxl.constant(0, name="layer_index")
                # Encoder layers mask out the cross-attention part
                scale = popxl.constant(0, config.model.dtype, "cross_attn_scale")
                layers_graph = ir.create_graph(t5_block, i, x, attention_mask, x, attention_mask, scale, rel_pos_weight)
                # We use identity() here to re-use the x and mask for the cross-attention.
                # As long as we don't use something that would generate nans it doesn't matter:
                # that part is going to be zeroed out in the encoder
                _, x = ops.repeat(
                    layers_graph,
                    config.model.layers,
                    i,
                    x,
                    attention_mask,
                    ops.identity(x),
                    ops.identity(attention_mask),
                    scale,
                    rel_pos_weight,
                )

                # Encoder head
                load_graph, names = load_remote_graph(enc_head_buffers)
                layer_vars = NamedTensors.pack(names, load_graph.call(0))
                (x,) = enc_head_graph.bind(layer_vars).call(x)

                # Decoder embeddings
                # Get the embedding weight from the encoder embedding layer
                load_graph, names = load_remote_graph(dec_embeddings_buffers)
                embedding_vars = NamedTensors.pack(names, load_graph.call(0))
                embedding_weight_t = embedding_vars.pop("weight")
                (x_dec,) = dec_embeddings_graph.bind(embedding_vars).call(decoder_word, embedding_weight_t)
                rel_pos_weight = embedding_vars.rel_pos_weight

                # Decoder
                # For decoder layers, i is in [N, 2N-1]
                i = popxl.constant(config.model.layers, name="layer_index")
                # Decoder layers don't mask out the cross-attention part
                scale = popxl.constant(1, config.model.dtype, "cross_attn_scale")
                _, x = ops.repeat(
                    layers_graph,
                    config.model.layers,
                    i,
                    x_dec,
                    decoder_attention_mask,
                    x,
                    attention_mask,
                    scale,
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
