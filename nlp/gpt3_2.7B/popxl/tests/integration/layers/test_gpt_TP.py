# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import torch

# HF
from transformers.models.gpt2 import GPT2Config as HFConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import popxl
from popxl.utils import to_numpy

import popxl_addons as addons
from popxl_addons.patterns import apply_pre_alias_patterns
from popxl_addons.array_munging import repeat

from config import GPTConfig
from modelling.embedding import GPTEmbeddingsTP, generate_positions
from modelling.gpt_model import GPTModelTP


def test_gpt_TP_cmp_huggingface(test_config: GPTConfig):
    torch.manual_seed(42)

    batch_size = test_config.execution.micro_batch_size
    hidden_size = test_config.model.hidden_size
    intermediate_size = hidden_size * 4

    # HuggingFace
    config = HFConfig(
        n_layer=test_config.model.layers,
        vocab_size=test_config.model.embedding.vocab_size,
        n_positions=test_config.model.embedding.max_positional_length,
        hidden_size=hidden_size,
        n_inner=intermediate_size,
        n_head=test_config.model.attention.heads,
    )
    hf_model = GPT2Model(config).eval()

    # HF forward
    input_t = torch.randint(0, test_config.model.embedding.vocab_size, (batch_size, test_config.model.sequence_length))
    output_HF = hf_model(input_ids=input_t)[0]

    # HF backward
    grad_wrt = torch.rand(output_HF.shape)
    output_HF.backward(gradient=grad_wrt)
    words_grad_HF = hf_model.wte.weight.grad.detach().numpy()
    positions_grad_HF = hf_model.wpe.weight.grad.detach().numpy()

    output_HF = output_HF.detach().numpy()

    # TP
    tp = 4
    test_config.execution.tensor_parallel = tp

    # popxl
    ir = popxl.Ir()
    ir.replication_factor = tp
    replica_grouping = ir.replica_grouping(stride=1, group_size=1)
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(
            *[
                addons.host_load(input_t, popxl.int32, name="words"),
            ]
        )
        (words,) = inputs_tensors
        pos = popxl.constant(generate_positions(test_config), popxl.int32, name="positions")

        facts, graph = GPTModelTP(test_config).create_graph(words, pos)

        vars = facts.init()
        gpt = graph.bind(vars)
        call_info = gpt.call_with_info(words, pos)
        act, *_ = call_info.outputs
        act_stream = addons.host_store(act)
        gradient = popxl.constant(grad_wrt.reshape(act.shape).numpy().copy(), act.dtype, "gradient")

        # Backwards
        grads_required = [t for t in graph.args.tensors if "offset" not in t.name]
        grad_graph = addons.autodiff(graph, grads_required=grads_required)

        grad_call_info = grad_graph.call_with_info(gradient, args=grad_graph.grad_graph_info.inputs_dict(call_info))

        tensor_to_grad_tensor = grad_graph.grad_graph_info.fwd_graph_ins_to_grad_parent_outs(grad_call_info)
        words_grad_d2h = addons.host_store(tensor_to_grad_tensor[graph.args.embeddings.word.weight])
        pos_grad_d2h = addons.host_store(tensor_to_grad_tensor[graph.args.embeddings.positional.weight])

    apply_pre_alias_patterns(ir, level="default")

    # Map weights from huggingface
    weights = GPTModelTP.hf_mapping(test_config, vars, hf_model)

    inputs = dict(zip(inputs_host_steam, [repeat(input_t, tp)]))

    ir.num_host_transfers = test_config.execution.device_iterations

    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    # Fwd output
    fwd_data = outs[act_stream]

    assert len(fwd_data) == tp
    for i in range(1, tp):
        np.testing.assert_equal(fwd_data[0], fwd_data[i])

    np.testing.assert_almost_equal(output_HF, fwd_data[0].reshape(output_HF.shape), 3)

    # Grad outputs
    grad_data_words = outs[words_grad_d2h].reshape(-1, hidden_size)[: test_config.model.embedding.vocab_size, :]
    grad_data_pos = outs[pos_grad_d2h].reshape(-1, hidden_size)[: test_config.model.embedding.max_positional_length, :]

    np.testing.assert_almost_equal(words_grad_HF, grad_data_words, 3)
    np.testing.assert_almost_equal(positions_grad_HF, grad_data_pos, 3)
