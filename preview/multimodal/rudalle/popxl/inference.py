# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import time

import numpy as np
import popxl_addons as addons
import torch
import torchvision
from popxl_addons.patterns import apply_pre_alias_patterns
from python_translator import Translator as g_Translator
from rudalle import get_rudalle_model, get_tokenizer
from rudalle.dalle import MODELS
from rudalle.vae import get_vae

import popxl
from config import InferenceConfig
from modeling.image_decoder import VQGanGumbelVAE
from modeling.modeling_cached_TP import EmbeddingsTP, LMModelTP, LMModelTP2
from popxl import ops
from popxl.utils import to_numpy

g_translator = g_Translator()
def preprocess(config, text = None):
    tokenizer = get_tokenizer()

    if not text:
        text = "The sunrise"
    else:
        logging.info("input: " + text)
        translate_begin = time.time()
        text = g_translator.translate(text, "russian", "english").new_text
        logging.info(f"Text translation duration: {time.time()-translate_begin} s")
    text = text.lower().strip()

    input_ids = tokenizer.encode_text(text, text_seq_length=config.text_seq_len)
    text_pad = config.vocab_size + torch.arange(config.text_seq_len)
    input_ids = torch.where(input_ids == 0, text_pad, input_ids)
    input_ids = input_ids.to(torch.int32).reshape(-1,)

    words_offsetted, pos_offsetted = EmbeddingsTP.offset_inputs(config, to_numpy(input_ids))
    return words_offsetted, pos_offsetted


def init(config):

    n_head = config.n_head
    layers = config.layers
    micro_bs = config.micro_batch_size
    hidden_size = config.n_embd
    output_len = config.output_len
    n_ipus = config.ipus

    # Offset inputs
    word_offsets, _ = EmbeddingsTP.get_offsets(config)
    word_offsets = word_offsets.reshape(n_ipus, 1)

    words_offsetted, pos_offsetted = preprocess(config)
    seed = np.array([0, 42], dtype=np.uint)

    # To avoid UnicodeEncodeError
    MODELS['Malevich']['description'] = MODELS['Malevich']['description'].encode("utf-8").decode("latin1")
    hf_model = get_rudalle_model('Malevich', pretrained=True, fp16=False, device='cpu')

    # popart.ir
    ir = popxl.Ir()
    ir.replication_factor = n_ipus
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    opts = ir._pb_ir.getSessionOptions()
    opts.partialsTypeMatMuls = "half"
    opts.convolutionOptions["partialsType"] = "half"
    opts.engineOptions["opt.internalExchangeOptimisationTarget"] = "memory"
    opts.engineOptions["opt.maxComputeSetsPerLoweredCopy"] = "6"

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(words_offsetted[0], popxl.int32, name="words"),
            addons.host_load(word_offsets[0], popxl.int32, name="word_offset"),
            addons.host_load(seed, popxl.uint32, name="seed"),
        ])
        words, word_offset, seed = inputs_tensors  # mask, mask1,

        ## stage 0: generate the first image token
        pos = popxl.variable(pos_offsetted, words.dtype, name="positions", replica_grouping=replica_grouping)

        args, graph = LMModelTP(config, replica_grouping).create_graph(words, position_ids=pos, seed=seed, word_offset=word_offset)
        vars = args.init()
        model = graph.bind(vars)
        next_token, presents_k, presents_v = model.call(words, pos, seed, word_offset)

        ## stage 1: generate subsequent image tokens
        past_k_padding = popxl.constant(
            np.zeros((micro_bs * layers, n_head // n_ipus, hidden_size // n_head, output_len-1)).astype(np.float16))
        past_v_padding = popxl.constant(
            np.zeros((micro_bs * layers, n_head // n_ipus, output_len-1, hidden_size // n_head)).astype(np.float16))
        presents_k = ops.concat([presents_k, past_k_padding], axis=3)
        presents_v = ops.concat([presents_v, past_v_padding], axis=2)

        position_ids = popxl.variable(pos_offsetted[:, -1]+1, words.dtype, name="positions", replica_grouping=replica_grouping)

        output = ops.concat([next_token, popxl.variable(np.zeros((config.output_len-1)), words.dtype)])
        update_index = popxl.variable(config.text_seq_len-1, popxl.int32)
        inputs = [next_token, position_ids, presents_k, presents_v, update_index, word_offset, seed, output]
        args1, stage1_graph = LMModelTP2(config, replica_grouping).create_graph(inputs)
        layer1 = stage1_graph.bind(vars)
        outputs = ops.repeat(layer1.graph, config.output_len-1, inputs, inputs_dict={**layer1.args})

        ## stage 2: image decoder
        input = outputs[-1].reshape((1, -1)) + word_offset
        args2, stage2_graph = VQGanGumbelVAE(config, n_ipus, replica_grouping=replica_grouping).create_graph(input)
        vars2 = args2.init()
        image_decoder = stage2_graph.bind(vars2)
        image, = image_decoder.call(input)
        out = addons.host_store(image)

    # Run `OpToIdentityPattern` among others part of `PreAliasPatterns`
    apply_pre_alias_patterns(ir, level='default')
    ir.num_host_transfers = config.device_iterations

    weights = LMModelTP.hf_mapping(config, vars, hf_model)
    decoder_model = get_vae()
    weights.update(VQGanGumbelVAE.hf_mapping(config, vars2, decoder_model))
    session = popxl.Session(ir, "ipu_hw")
    session.write_variables_data(weights)
    return session, inputs_host_steam, out, word_offsets


def postprocess(outputs_popxl, out_stream, filename):
    response = outputs_popxl[out_stream]
    pil_image = torchvision.transforms.functional.to_pil_image(
        torch.from_numpy(response[0][0]).float()).convert('RGB')
    pil_image.save(filename)


def main():
    config = InferenceConfig()

    session, _, _, _ = init(config)
    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape),
                        stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    with session:
        # Skip one result
        session.run(inputs)
        durations = []
        for _ in range(5):
            start_time = time.time()
            session.run(inputs)
            durations.append(time.time() - start_time)

    duration = np.mean(durations)
    samples_per_step = config.micro_batch_size
    result_str = \
        f"Duration: {duration} s " \
        f"Throughput: {samples_per_step/duration:.2f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    main()
