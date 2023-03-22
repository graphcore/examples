# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import numpy as np
from torch import nn
import poptorch

import nbfnet_utils


NUM_NODES = 100
LATENT_DIM = 16
BATCH_SIZE = 4
NUM_NEGATIVE = 8

feature = torch.rand(BATCH_SIZE, NUM_NODES, LATENT_DIM)
tail_id = torch.randint(0, NUM_NODES, [BATCH_SIZE, NUM_NEGATIVE])


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, tail_id):
        tail_feature_gather = feature.gather(1, tail_id.unsqueeze(-1).expand(-1, -1, feature.shape[-1]))
        tail_feature_custom = nbfnet_utils.batch_index_select(feature, tail_id)
        return tail_feature_gather, tail_feature_custom


def test_batch_index_select():
    model = Model()
    pop_model = poptorch.inferenceModel(model)

    reference_result, _ = model(feature, tail_id)
    pt_result, custom_result = pop_model(feature, tail_id)

    np.testing.assert_allclose(reference_result, pt_result, atol=1e-6)
    np.testing.assert_allclose(reference_result, custom_result, atol=1e-6)
