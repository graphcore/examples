# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import torch
import torch.nn.functional as F
from transformers.models.bert import BertConfig as HFConfig
from transformers.models.bert.modeling_bert import BertPooler, BertPreTrainingHeads

import popxl_addons as addons
from config import BertConfig
from modelling.bert_model import BertPretrainingLossAndGrad


def test_bert_pretraining_cmp_huggingface(test_config: BertConfig):
    torch.manual_seed(1024)

    test_config.model.hidden_size = 4
    test_config.model.sequence_length = 2
    test_config.model.embedding.vocab_size = 8
    test_config.execution.micro_batch_size = 2
    test_config.model.mlm.mask_tokens = 1

    config = HFConfig(
        hidden_size=test_config.model.hidden_size,
        seq_len=test_config.model.sequence_length,
        vocab_size=test_config.model.embedding.vocab_size,
        num_attention_heads=test_config.model.attention.heads,
        max_position_embeddings=test_config.model.embedding.max_positional_length,
        intermediate_size=test_config.model.hidden_size * 4,
        num_hidden_layers=test_config.model.layers,
    )
    hf_pooler = BertPooler(config).eval()
    hf_cls = BertPreTrainingHeads(config).eval()

    # HuggingFace
    micro_bs = test_config.execution.micro_batch_size
    seq_len = test_config.model.sequence_length
    hidden_size = test_config.model.hidden_size

    # HF forwards
    sequence_output = torch.rand((micro_bs, seq_len, hidden_size), requires_grad=True)
    # randperm used to guarentee no duplicate positions
    masked_positions = torch.vstack(
        [torch.randperm(seq_len)[:test_config.model.mlm.mask_tokens] for _ in range(micro_bs)])
    labels = torch.randint(0, config.vocab_size, (micro_bs, seq_len), dtype=torch.long)
    nsp_labels = torch.randint(0, 2, (micro_bs, ), dtype=torch.long)
    # Set non-masked positions to 0
    for seq, pos in zip(labels, masked_positions):
        not_pos = torch.ones_like(seq).to(torch.bool)
        not_pos[pos] = False
        seq[not_pos] = 0

    # HF execution
    pooled_output = hf_pooler(sequence_output)
    prediction_scores, seq_relationship_score = hf_cls(sequence_output, pooled_output)

    masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, config.vocab_size), labels.view(-1), ignore_index=0)
    next_sentence_loss = F.cross_entropy(seq_relationship_score.view(-1, 2), nsp_labels.view(-1))
    hf_loss = masked_lm_loss + next_sentence_loss

    # HF backwards
    hf_loss.backward()
    hf_loss = hf_loss.detach().numpy()
    assert sequence_output.grad is not None
    hf_dx = sequence_output.grad.detach().numpy()

    # Reduce labels to only mask_tokens
    labels_ = []
    for seq, pos in zip(labels, masked_positions):
        labels_.append(seq[pos])
    labels = torch.vstack(labels_)

    # popxl
    ir = popxl.Ir()
    main = ir.main_graph

    with main:
        inputs_data, inputs_host_steam, inputs_tensors = zip(*[
            addons.host_load(sequence_output.reshape((micro_bs * seq_len, test_config.model.hidden_size)),
                             popxl.float32,
                             name="sequence_output"),
            addons.host_load(labels.reshape((micro_bs, test_config.model.mlm.mask_tokens)), popxl.int32, name="labels"),
            addons.host_load(masked_positions.reshape((micro_bs, test_config.model.mlm.mask_tokens)),
                             popxl.int32,
                             name="masked_positions"),
            addons.host_load(nsp_labels.reshape((micro_bs, )), popxl.int32, name="nsp_labels"),
        ])
        sequence_output, labels, masked_positions, nsp_labels = inputs_tensors
        tied_weight = popxl.constant(
            popxl.utils.to_numpy(hf_cls.predictions.decoder.weight.data.T, test_config.model.dtype))
        args, head = BertPretrainingLossAndGrad(test_config).create_graph(sequence_output, tied_weight, tied_weight,
                                                                          masked_positions, labels, nsp_labels)
        tied_weight_grad_t = popxl.ops.init(tied_weight.shape, tied_weight.dtype, 'word_embedding_grad_t', "zero")

        # Forwards
        variables = args.init()
        loss, dx = head.bind(variables).call(sequence_output, tied_weight, tied_weight_grad_t, masked_positions, labels,
                                             nsp_labels)
        loss_out = addons.host_store(loss)
        dx_out = addons.host_store(dx)

    # Map weights from huggingface
    weights = BertPretrainingLossAndGrad.hf_mapping(test_config, variables, hf_cls, hf_pooler)

    inputs = dict(zip(inputs_host_steam, inputs_data))
    with popxl.Session(ir, "ipu_hw") as session:
        session.write_variables_data(weights)
        outs = session.run(inputs)

    np.testing.assert_almost_equal(hf_loss.flatten(), outs[loss_out].flatten(), 3)
    np.testing.assert_almost_equal(hf_dx.flatten(), outs[dx_out].flatten(), 3)
