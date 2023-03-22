# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Includes derived work from https://github.com/KiddoZhu/NBFNet-PyG
#   Copyright (c) 2021 MilaGraph
#   Licensed under the MIT License

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class GeneralizedRelationalConv(MessagePassing):
    eps = 1e-6

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int,
        query_input_dim: int,
        message_fct: str,
        aggregation_fct: str,
        skip_connection: bool,
        relation_learning: str,
    ):
        super().__init__()
        self.message_fct = message_fct
        if aggregation_fct == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        assert aggregation_fct in ("sum", "mean", "mul", "min", "max", "pna")
        self.aggregation_fct = aggregation_fct
        self.skip_connection = skip_connection
        assert relation_learning in ("identity", "linear", "linear_query", "independent")
        self.relation_learning = relation_learning
        self.layer_norm = nn.LayerNorm(output_dim)
        if relation_learning == "linear":
            self.relation_linear = nn.Linear(query_input_dim, input_dim)
        elif relation_learning == "linear_query":
            self.relation_linear = nn.Linear(query_input_dim, num_relations * input_dim)
        elif relation_learning == "independent":
            self.relation_embedding = nn.Embedding(num_relations, input_dim)

    def forward(
        self,
        hidden: torch.Tensor,  # (batch_size, num_nodes, dim)
        query: torch.Tensor,  # (batch_size, dim)
        boundary: torch.Tensor,  # (batch_size, num_nodes, dim)
        graph: torch.Tensor,  # (num_triples, 3)
        all_query: torch.Tensor,  # (num_relations, dim)
        edge_weight: torch.Tensor,
    ):  # (num_triples)

        batch_size, num_nodes, dim = hidden.shape
        if self.relation_learning == "identity":
            relation_embedding = all_query.expand(batch_size, -1, -1)
        elif self.relation_learning == "linear":
            relation_embedding = self.relation_linear(all_query).expand(batch_size, -1, -1)
        elif self.relation_learning == "linear_query":
            relation_embedding = self.relation_linear(query).view(batch_size, -1, dim)
        elif self.relation_learning == "independent":
            relation_embedding = self.relation_embedding.expand(batch_size, -1, -1)
        else:
            raise NotImplementedError
        output = self.propagate(
            input=hidden,
            relation=relation_embedding,
            boundary=boundary,
            edge_index=graph[:, :2].t(),
            edge_type=graph[:, 2].t(),
            edge_weight=edge_weight,
        )
        if self.skip_connection:
            output = hidden + output
        return output, query, boundary, graph, all_query

    def message(self, input_j, relation, boundary, edge_type):
        relation_embedding_j = relation.index_select(1, edge_type)
        if self.message_fct == "add":
            message = input_j * relation_embedding_j
        elif self.message_fct == "mult":
            message = input_j * relation_embedding_j
        else:
            raise NotImplementedError
        return torch.cat([message, boundary], dim=1)

    def aggregate(self, input, index, dim_size, edge_weight):
        index = torch.cat([index, torch.arange(dim_size, device=index.device)])
        if isinstance(edge_weight, torch.Tensor):
            edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device, dtype=input.dtype)])
            edge_weight = edge_weight.view([1, -1, 1])
        if self.aggregation_fct == "pna":
            mean_aggr = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            sq_mean_aggr = scatter(input**2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            std_aggr = (sq_mean_aggr - mean_aggr**2).clamp(min=self.eps).sqrt()
            max_aggr = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            min_aggr = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")

            features = torch.cat(
                [
                    mean_aggr,
                    std_aggr,
                    max_aggr,
                    min_aggr,
                ],
                dim=-1,
            )
            degree_out = degree(index, dim_size).unsqueeze(-1)  # including self loops
            scale = degree_out.log()
            scale = scale / scale.mean()
            scale = scale.to(features.dtype)
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            return torch.einsum("ijk, jl -> ijkl", features, scales).flatten(-2)
        else:
            return scatter(
                input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggregation_fct
            )

    def update(self, update, input):
        output = self.linear(torch.cat([update, input], dim=-1))
        output = self.layer_norm(output)
        output = F.relu(output)
        return output
