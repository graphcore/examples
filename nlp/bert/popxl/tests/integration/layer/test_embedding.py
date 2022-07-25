# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import torch
from transformers.models.bert import BertConfig as HFConfig
from transformers.models.bert.modeling_bert import BertEmbeddings as HFBertEmbeddings

import popxl_addons as addons

from config import BertConfig
from modelling.embedding import BertEmbeddings


def test_embedding_cmp_huggingface(test_config: BertConfig):
    torch.manual_seed(42)

    config = HFConfig(
        hidden_size=test_config.model.hidden_size,
        seq_len=test_config.model.sequence_length,
        vocab_size=test_config.model.embedding.vocab_size,
        max_position_embeddings=test_config.model.embedding.max_positional_length,
    )
    module = HFBertEmbeddings(config).eval()

    # HuggingFace
    micro_bs = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size

    # HF forwards
    words = torch.randint(1, config.vocab_size,
                          (micro_bs, seq_len), dtype=torch.int32)
    token_types = torch.randint(2, (micro_bs, seq_len), dtype=torch.int32)
    output_HF = module(words, token_types)

    # HF backwards
    grad_wrt = torch.rand(output_HF.shape)
    output_HF.backward(gradient=grad_wrt)
    assert module.word_embeddings.weight is not None
    assert module.position_embeddings.weight is not None
    assert module.token_type_embeddings.weight is not None
    words_grad_HF = module.word_embeddings.weight.grad.detach().numpy()
    positions_grad_HF = module.position_embeddings.weight.grad.detach().numpy()
    token_types_grad_HF = module.token_type_embeddings.weight.grad.detach().numpy()
    output_HF = output_HF.detach().numpy()

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = test_config.execution.data_parallel

    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(words.flatten(), popxl.int32, name="words"),
            addons.host_load(token_types.flatten(),
                             popxl.int32, name="token_type"),
        ])
        args, graph = BertEmbeddings(test_config).create_graph(*inputs_tensors)
        grad_graph = addons.autodiff(
            graph,
            grads_required=graph.args.tensors)

        embed = args.init("embedding")
        layer = graph.bind(embed)
        call_info = layer.call_with_info(*inputs_tensors)
        act, *_ = call_info.outputs

        output = addons.host_store(act)

        # Backwards
        gradient = popxl.constant(grad_wrt.reshape(
            act.shape).numpy().copy(), act.dtype, "gradient")
        grad_call_info = grad_graph.call_with_info(
            gradient, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        grad_map = grad_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(
            grad_call_info)

        words_grad_stream = addons.host_store(grad_map[graph.args.word.weight])
        positions_grad_stream = addons.host_store(
            grad_map[graph.args.positional.weight])
        token_types_grad_stream = addons.host_store(
            grad_map[graph.args.token_type.weight])

    weights = BertEmbeddings.hf_mapping(test_config, embed, module)

    inputs = dict(zip(inputs_host_steam, inputs_data))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    np.testing.assert_almost_equal(
        output_HF, outs[output].reshape(output_HF.shape), 5)
    np.testing.assert_almost_equal(words_grad_HF, outs[words_grad_stream][:test_config.model.embedding.vocab_size, :], 5)
    np.testing.assert_almost_equal(
        positions_grad_HF, outs[positions_grad_stream][:test_config.model.embedding.max_positional_length, :], 5)
    np.testing.assert_almost_equal(
        token_types_grad_HF, outs[token_types_grad_stream][:2, :], 5)
