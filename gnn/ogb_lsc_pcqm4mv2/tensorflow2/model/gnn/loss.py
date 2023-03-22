# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from functools import partial

from model.gnn.losses_and_metrics import MaskedMeanAbsoluteError, LossNoisyNodes


def get_loss_functions_gnn(
    dataset,
    use_noisy_nodes=False,
    use_noisy_edges=False,
    noisy_nodes_weight=0.05,
    noisy_edges_weight=0.05,
    noisy_node_method="combined_softmax",
):

    loss_functions = [MaskedMeanAbsoluteError]
    loss_weights = [1.0]

    if use_noisy_nodes:
        loss_weights.append(noisy_nodes_weight)
        loss_functions.append(
            partial(LossNoisyNodes, mode="nodes", method=noisy_node_method, vocab_size=dataset.node_feature_dims)
        )
    if use_noisy_edges:
        loss_weights.append(noisy_edges_weight)
        loss_functions.append(
            partial(LossNoisyNodes, mode="edges", method=noisy_node_method, vocab_size=dataset.edge_feature_dims)
        )

    assert len(loss_weights) == len(loss_functions)
    return loss_functions, loss_weights
