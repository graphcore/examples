# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Includes derived work from https://github.com/KiddoZhu/NBFNet-PyG
#   Copyright (c) 2021 MilaGraph
#   Licensed under the MIT License


from collections.abc import Sequence
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
import poptorch

import nbfnet_layer
import nbfnet_utils


class NBFNet(nn.Module):
    """Main class for NBFNet"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, list],
        num_relations: int,
        message_fct: str,
        aggregation_fct: str,
        num_mlp_layers: int,
        adversarial_temperature: float,
        relation_learning: str,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.adversarial_temperature = adversarial_temperature
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + list(hidden_dims)

        self.query = nn.Embedding(num_relations, input_dim)

        # add layers to model
        self.nbf_layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.nbf_layers.append(
                nbfnet_layer.GeneralizedRelationalConv(
                    input_dim=self.dims[i],
                    output_dim=self.dims[i + 1],
                    num_relations=num_relations,
                    query_input_dim=self.dims[0],
                    message_fct=message_fct,
                    aggregation_fct=aggregation_fct,
                    skip_connection=True,
                    relation_learning=relation_learning,
                )
            )

        self.mlp = nn.Sequential()
        feature_dim = hidden_dims[-1] + input_dim
        for i in range(num_mlp_layers - 1):
            self.mlp.add_module(f"linear{i}", nn.Linear(feature_dim, feature_dim))
            self.mlp.add_module(f"ReLU{i}", nn.ReLU())
        self.mlp.add_module(f"linear{num_mlp_layers - 1}", nn.Linear(feature_dim, 1))

    def link_predictor(self, hidden, query, tail_id):
        """Performs link prediction for a query (head, relation, ?) for multiple tails by
        gathering the embedding of tail entities and scoring them with an MLP
        :param hidden: The hidden state of all entities in the graph.
            Shape [batch_size, num_nodes, hidden_dim]
        :param query: Query embedding. Shape [batch_size, num_nodes, input_dim]
        :param tail_id: Shape [batch_size, 1 + num_negatives]
        :return: score for each tail_id. Shape [batch_size, 1 + num_negatives]
        """
        feature = torch.cat([hidden, query], dim=-1)
        tail_feature = nbfnet_utils.batch_index_select(feature, tail_id)
        score = self.mlp(tail_feature).squeeze(-1)
        return score

    def forward(
        self,
        graph: torch.Tensor,
        num_nodes: int,
        head_id: torch.Tensor,
        tail_id: torch.Tensor,
        relation_id: torch.Tensor,
        get_edge_importance: bool = False,
    ):
        """
        :param graph: The full set of triples (head, tail, relation).
            Shape [num_triples, 3]
        :param num_nodes: number of nodes in graph
        :param head_id: Shape [batch_size]
        :param tail_id: Shape [batch_size, 1 + num_negatives]
        :param relation_id: Shape [batch_size]
        :param get_edge_importance: Only for inference. Returns edge weights to
            calculate most important paths from head to tail
        :return:
        """

        with poptorch.Block("preprocessing"):
            if isinstance(num_nodes, list):
                num_nodes = num_nodes[0]
                graph = graph.squeeze(0)
                head_id = head_id.squeeze(0)
                tail_id = tail_id.squeeze(0)
                relation_id = relation_id.squeeze(0)

            batch_size, num_tails = tail_id.shape

            query = self.query(relation_id)  # (batch_size, input_dim)
            index = head_id.unsqueeze(-1).expand_as(query)
            # compute boundary condition
            boundary = scatter(query.unsqueeze(1), index.unsqueeze(1), dim=1, dim_size=num_nodes, reduce="sum")

            hidden = boundary
            all_query = self.query.weight
            edge_weights = []
            if get_edge_importance:
                edge_weight = torch.ones(graph.shape[0], device=hidden.device, dtype=hidden.dtype)
        for layer_id, layer in enumerate(self.nbf_layers):
            with poptorch.Block(f"layer{layer_id}"):
                nbfnet_utils.recomputation_checkpoint(layer)
                if get_edge_importance:
                    edge_weight = edge_weight.clone().requires_grad_()
                else:
                    edge_weight = 1
                hidden, query, boundary, graph, all_query = layer(
                    hidden, query, boundary, graph, all_query, edge_weight
                )
                edge_weights.append(edge_weight)

        with poptorch.Block("prediction"):
            node_query = query.unsqueeze(1).expand(-1, num_nodes, -1)
            prediction = self.link_predictor(hidden, node_query, tail_id)
            batch_mask = (head_id > 0).unsqueeze(1)
            prediction *= batch_mask
            count = batch_mask.sum()

            if self.training:
                target = torch.zeros_like(prediction, device=prediction.device)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
                loss *= batch_mask
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        pos_weight = torch.ones([batch_size, 1], device=prediction.device)
                        neg_weight = F.softmax(prediction[:, 1:] / self.adversarial_temperature, dim=-1)
                        weight = torch.cat([pos_weight, neg_weight], dim=-1)
                else:
                    weight = torch.ones_like(prediction, device=prediction.device)
                    weight[:, 1:] = 1 / (num_tails - 1)

                loss = (loss * weight).sum(dim=-1) / 2
                return poptorch.identity_loss(loss, "mean"), count

            return prediction, count, batch_mask.squeeze(), edge_weights
