# Copyright 2019 Graphcore Ltd.
import torch
import popart
import numpy as np

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertForQuestionAnswering

from .full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full squad graph.
'''


def test_squad_fwd():
    #  ------------------- PopART --------------------
    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        num_layers=2,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        inference=True)
    popart_model = Bert(config, builder=builder)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForQuestionAnswering(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        num_labels=2))

    fwd_graph(popart_model,
              torch_model,
              mapping={
                  "cls.transform.dense.weight": "CLS/LMPredictionW",
                  "cls.transform.dense.bias": "CLS/LMPredictionB",
                  "cls.transform.LayerNorm.weight": "CLS/Gamma",
                  "cls.transform.LayerNorm.bias": "CLS/Beta",
                  "qa_outputs.weight": "Squad/SquadW",
                  "qa_outputs.bias": "Squad/SquadB"
              },
              transform={
                  "cls.transform.dense.weight": np.transpose,
                  "qa_outputs.weight": np.transpose
              })


def test_squad_bwd():
    #  ------------------- PopART --------------------
    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(task="SQUAD",
                        vocab_length=9728,
                        num_layers=1,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        update_embedding_dict=False)
    popart_model = Bert(config, builder=builder)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForQuestionAnswering(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        num_labels=2))

    l1_lambda = 0.1

    def popart_loss_fn(outputs):
        losses = [
            builder.aiGraphcore.l1loss(
                [outputs[0]], l1_lambda, debugPrefix="startsLossVal", reduction=popart.ReductionType.Sum),
            builder.aiGraphcore.l1loss(
                [outputs[1]], l1_lambda, debugPrefix="endsLossVal", reduction=popart.ReductionType.Sum),
        ]
        for loss in losses:
            builder.virtualGraph(loss, popart_model.squad_scope.virtualGraph)

        final_loss = builder.aiOnnx.sum(losses, debugPrefix="finalLoss")
        builder.virtualGraph(final_loss, popart_model.squad_scope.virtualGraph)
        return final_loss

    def torch_loss_fn(outputs):
        torch_losses = [l1_lambda * torch.norm(output, 1)
                        for output in outputs]
        return torch.add(*torch_losses)

    bwd_graph(popart_model,
              torch_model,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=torch_loss_fn,
              mapping={
                  "cls.transform.dense.weight": "CLS/LMPredictionW",
                  "cls.transform.dense.bias": "CLS/LMPredictionB",
                  "cls.transform.LayerNorm.weight": "CLS/Gamma",
                  "cls.transform.LayerNorm.bias": "CLS/Beta",
                  "qa_outputs.weight": "Squad/SquadW",
                  "qa_outputs.bias": "Squad/SquadB"
              },
              transform={
                  "cls.transform.dense.weight": np.transpose,
                  "qa_outputs.weight": np.transpose
              })
