# Copyright 2019 Graphcore Ltd.
import os
import ctypes
import torch
import popart
import numpy as np

from bert_model import BertConfig, Bert
from tests.torch_bert import BertConfig as TorchBertConfig, BertForMaskedLM

from tests.unit.pytorch.full_graph_utils import fwd_graph, bwd_graph


'''
Tests the full pretraining graph.
'''

onnx_torch_mapping = {
    "cls.transform.dense.weight": "CLS/LMPredictionW",
    "cls.transform.dense.bias": "CLS/LMPredictionB",
    "cls.transform.LayerNorm.weight": "CLS/Gamma",
    "cls.transform.LayerNorm.bias": "CLS/Beta",
}

onnx_torch_tform = {
    "cls.transform.dense.weight": np.transpose,
}


def test_pretraining_fwd(custom_ops):
    #  ------------------- PopART --------------------
    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        projection_serialization_steps=4,
                        num_layers=2,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        activation_type="relu",
                        no_dropout=True,
                        custom_ops=["gather", "attention"],
                        inference=True)
    popart_model = Bert(config, builder=builder)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForMaskedLM(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens))

    fwd_graph(popart_model, torch_model, mapping=onnx_torch_mapping, transform=onnx_torch_tform)


def test_pretraining_bwd(custom_ops):
    #  ------------------- PopART --------------------
    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1})
    config = BertConfig(task="PRETRAINING",
                        vocab_length=9728,
                        projection_serialization_steps=4,
                        num_layers=1,
                        batch_size=1,
                        hidden_size=768,
                        sequence_length=128,
                        popart_dtype="FLOAT",
                        activation_type="relu",
                        no_dropout=True,
                        custom_ops=["gather", "attention"],
                        update_embedding_dict=False)
    popart_model = Bert(config, builder=builder)

    #  ------------------- PyTorch -------------------------
    torch_model = BertForMaskedLM(
        TorchBertConfig(config.vocab_length, config.hidden_size,
                        num_hidden_layers=config.num_layers,
                        num_attention_heads=config.attention_heads,
                        intermediate_size=config.ff_size,
                        hidden_act="relu",
                        max_position_embeddings=config.max_positional_length,
                        layer_norm_eps=config.layer_norm_eps,
                        mask_tokens=config.mask_tokens))

    l1_lambda = 0.1

    def popart_loss_fn(logits):
        loss = popart.L1Loss(logits[0], "l1LossVal", l1_lambda)
        loss.virtualGraph(popart_model.mlm_scope.virtualGraph)
        return [loss]

    bwd_graph(popart_model,
              torch_model,
              popart_loss_fn=popart_loss_fn,
              torch_loss_fn=lambda logits: l1_lambda * torch.norm(logits[0], 1),
              mapping=onnx_torch_mapping, transform=onnx_torch_tform)
