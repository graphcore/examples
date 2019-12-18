# Copyright 2019 Graphcore Ltd.
import os
import ctypes
import torch
import popart
import numpy as np

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertForNextSentencePrediction

from .full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full nsp graph.
'''

NSP_MAPPING = {
    "bert.pooler.dense.weight": "NSP/PoolW",
    "bert.pooler.dense.bias": "NSP/PoolB",
    "cls.seq_relationship.weight": "NSP/NspW",
    "cls.seq_relationship.bias": "NSP/NspB"
}

NSP_TRANSFORM = {
    "bert.pooler.dense.weight": np.transpose,
    "cls.seq_relationship.weight": np.transpose
}


def test_nsp_fwd(custom_ops):
    #  ------------------- PopART --------------------
    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(task="NSP",
                        vocab_length=9728,
                        num_layers=2,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=["gather", "attention"],
                        inference=True)
    popart_model = Bert(config, builder=builder)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForNextSentencePrediction(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act=config.activation_type,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        num_labels=2))

    fwd_graph(popart_model,
              torch_model,
              mapping=NSP_MAPPING,
              transform=NSP_TRANSFORM)


def test_nsp_bwd(custom_ops):
    #  ------------------- PopART --------------------
    builder = popart.Builder(
        opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(task="NSP",
                        vocab_length=9728,
                        num_layers=1,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        activation_type="relu",
                        popart_dtype="FLOAT",
                        no_dropout=True,
                        custom_ops=["gather", "attention"])
    popart_model = Bert(config, builder=builder)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForNextSentencePrediction(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act=config.activation_type,
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens,
                        num_labels=2))

    def popart_loss_fn(outputs):
        loss = popart.L1Loss(outputs[0], "l1Loss", 0.1)
        loss.virtualGraph(popart_model.nsp_scope.virtualGraph)
        return [loss]

    def torch_loss_fn(outputs):
        return 0.1 * torch.norm(outputs[0], 1)

    bwd_graph(popart_model,
              torch_model,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=torch_loss_fn,
              mapping=NSP_MAPPING,
              transform=NSP_TRANSFORM)
