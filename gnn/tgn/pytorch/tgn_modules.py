# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Includes derived work from the PyTorch Geometric implementation at
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/tgn.py
#   Copyright (c) 2021 Matthias Fey, Jiaxuan You
#   Licensed under the MIT License
#


import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import torch_geometric as G
import copy
from torch_scatter import scatter_sum
import poptorch
import multiprocessing


torch.set_num_threads(multiprocessing.cpu_count() // 2)


class DataWrapper(torch.utils.data.IterableDataset):
    def __init__(self, data, partition):
        super(DataWrapper).__init__()
        self.batches = list(data.batches(partition=partition))
        self.length = data.n_batches(partition=partition)

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.batches)


class Data:
    """Data loading, batching, negative sampling & last neighbour loading."""

    def __init__(self, path, dtype, batch_size, nodes_size, edges_size):
        self.data = G.datasets.JODIEDataset(path, name="wikipedia")[0]
        self.batch_size = batch_size
        self.nodes_size = 1 + nodes_size  # rough empirical figures
        self.edges_size = edges_size
        train, val, test = self.data.train_val_test_split(
            val_ratio=0.15, test_ratio=0.15
        )
        self.partitions = dict(train=train, val=val, test=test)
        feature_size = self.data.msg.shape[-1]
        self.batch_spec = dict(
            # Map from idx -> (global) node ID
            node_ids=((self.nodes_size,), torch.long, -1),
            # Batch of events
            # ..(src, pos_dst, neg_dst)
            node_idx=((3, self.batch_size), torch.long, self.nodes_size - 1),
            node_t=((self.batch_size,), torch.int32, 0),
            node_msg=((self.batch_size, feature_size), dtype, 0.0),
            # ..mask most recent (src, pos_dst)
            most_recent=((2, self.batch_size), torch.bool, False),
            # Context events (from neighbour loader)
            edge_idx=((2, self.edges_size), torch.long, self.nodes_size - 1),
            edge_t=((self.edges_size,), torch.int32, 0),
            edge_msg=((self.edges_size, feature_size), dtype, 0.0),
        )

        # Precompute the correct starting state of LastNeighborLoader for each partition
        self.neighbour_loaders = {}
        neighbour_loader = G.nn.models.tgn.LastNeighborLoader(
            self.data.num_nodes, size=10
        )
        self.neighbour_loaders["train"] = copy.deepcopy(neighbour_loader)
        for batch in self.partitions["train"].seq_batches(self.batch_size):
            neighbour_loader.insert(batch.src, batch.dst)
        self.neighbour_loaders["val"] = copy.deepcopy(neighbour_loader)
        for batch in self.partitions["val"].seq_batches(self.batch_size):
            neighbour_loader.insert(batch.src, batch.dst)
        self.neighbour_loaders["test"] = copy.deepcopy(neighbour_loader)

        # Also precompute neg_samples, but only for validation & test.
        dst_min, dst_max = int(self.data.dst.min()), int(self.data.dst.max())
        self.neg_samples = {}
        for part in ["val", "test"]:
            torch.manual_seed(12345)
            self.neg_samples[part] = [
                torch.randint(dst_min, dst_max + 1, batch.src.shape, dtype=torch.long)
                for batch in self.partitions[part].seq_batches(self.batch_size)
            ]

    def n_batches(self, partition):
        """Exact total (padded) batch count for this partition."""
        return int(np.ceil(self.partitions[partition].num_events / self.batch_size))

    def unpadded_batches(self, partition):
        """Generate unpadded numpy batches (encapsulates PyTorch bits)."""
        neighbour_loader = copy.deepcopy(self.neighbour_loaders[partition])
        dst_min, dst_max = int(self.data.dst.min()), int(self.data.dst.max())
        node_id_to_idx = torch.empty(self.data.num_nodes, dtype=torch.long)
        expected_count = self.n_batches(partition)
        for batch_n, batch in enumerate(
            self.partitions[partition].seq_batches(self.batch_size)
        ):
            assert batch_n < expected_count
            neg_dst = (
                torch.randint(dst_min, dst_max + 1, batch.src.shape, dtype=torch.long)
                if partition == "train"
                else self.neg_samples[partition][batch_n]
            )
            node_ids, edges, edge_ids = neighbour_loader(
                torch.cat([batch.src, batch.dst, neg_dst]).unique()
            )
            node_id_to_idx[node_ids] = torch.arange(node_ids.shape[0])
            batch_idx = torch.stack(
                [node_id_to_idx[ids] for ids in [batch.src, batch.dst, neg_dst]]
            )
            # Transpose first because in "most recent" we want axis=1 (sequence)
            # ordered first, then axis=0 (src/dest)
            batch_most_recent = (
                most_recent_indices(batch_idx[:2].T.flatten()).reshape(-1, 2).T
            )
            yield dict(
                node_ids=node_ids,
                node_idx=batch_idx.type(torch.long),
                node_t=batch.t,
                node_msg=batch.msg,
                most_recent=batch_most_recent,
                edge_idx=edges,
                edge_t=self.data.t[edge_ids],
                edge_msg=self.data.msg[edge_ids],
            )
            neighbour_loader.insert(batch.src, batch.dst)
        assert batch_n == expected_count - 1

    def _pad_batch(self, batch):
        assert batch.keys() == self.batch_spec.keys()
        assert (
            batch["node_ids"].shape[0] <= self.nodes_size - 1
        ), "node_ids requires at least 1 padding element"

        out = {}
        for key, (shape, dtype, pad_value) in self.batch_spec.items():
            value = batch[key]
            dims = list(zip(value.shape, shape))

            assert all(
                actual <= target for actual, target in dims
            ), f"original shape {value.shape} larger than target {shape}"

            padding = []
            for actual, target in reversed(dims):
                padding.extend([0, target - actual])

            out[key] = F.pad(value.type(dtype), padding, value=pad_value)
            if key == 'node_ids':
                out[key] += 1

        return out

    def batches(self, partition):
        """Generate padded numpy batches of the correct dtype & shape."""
        return map(self._pad_batch, self.unpadded_batches(partition))


class TimeEncoder(nn.Module):
    def __init__(self, out_channels, dtype):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels, dtype=dtype)

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()


class TransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, dropout,
                 heads, bias=True):

        super(TransformerConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_skip = nn.Linear(in_channels, heads * out_channels, bias=bias)

    def forward(self, x, edge_index, edge_attr):
        # propagate
        x_i = x.index_select(0, edge_index[1])
        x_j = x.index_select(0, edge_index[0])
        size = x.shape[0]
        out = self.message(x_i, x_j, edge_attr, edge_index[1], size)
        out = scatter_sum(out, edge_index[1], dim=0, dim_size=size)
        # concatenate
        out = out.view(-1, self.heads * self.out_channels)
        # add residual
        out += self.lin_skip(x)
        return out

    def message(self, x_i, x_j, edge_attr, index, size):
        edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)
        key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(in={self.in_channels}, out={self.out_channels}, edge_dim={self.edge_dim}, heads={self.heads})'


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_dim, dropout, dtype):
        super(GraphAttentionEmbedding, self).__init__()
        # NOTE: Other versions of TGN re-use the time-encoder from the memory module. We
        # found model performance to be superior when using a different time-encoder
        self.time_enc = TimeEncoder(time_dim, dtype=dtype)
        edge_dim = msg_dim + time_dim
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            edge_dim=edge_dim,
            dropout=dropout,
            heads=2,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([msg, rel_t_enc], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_hid = nn.Linear(in_channels * 2, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_hid(torch.cat([z_src, z_dst], axis=-1))
        h = h.relu()
        return self.lin_final(h)


class TGNMemory(nn.Module):
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim, dtype, target):
        super(TGNMemory, self).__init__()
        self.dtype = dtype
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.target = target

        self.time_enc = TimeEncoder(time_dim, dtype)
        gru_in_dim = 2 * self.memory_dim + self.raw_msg_dim + self.time_dim
        self.gru = nn.GRUCell(gru_in_dim, memory_dim, dtype=self.dtype)

        # last_update, rel_t, pos_dst
        self.register_buffer('_memory_ints', torch.empty(num_nodes + 1, 3, dtype=torch.float))
        self.register_buffer('_memory', torch.empty(num_nodes + 1, memory_dim, dtype=self.dtype))
        self.register_buffer('_memory_msg', torch.empty(num_nodes + 1, raw_msg_dim, dtype=self.dtype))
        self.register_buffer('_direction', torch.empty(num_nodes + 1, 2, dtype=self.dtype))    #TODO: this can be combined with _memory

        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self._memory_ints)
        zeros(self._memory)
        zeros(self._memory_msg)
        zeros(self._direction)

    def detach(self):
        """Detaches the memory from gradient computation."""
        self._memory_ints.detach_()
        self._memory.detach_()
        self._memory_msg.detach_()
        self._direction.detach_()

    def forward(self, n_id):
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        return self.__get_memory__(n_id)

    def __get_memory__(self, n_id):
        # fetch data from message store
        last_update, rel_t, dst_id = torch.unbind(self._memory_ints[n_id].long(), 1)
        dst_memory = self._memory[dst_id.long()]
        time_encoding = self.time_enc(rel_t.type(torch.float32))

        src_memory = self._memory[n_id]
        raw_msg = self._memory_msg[n_id]
        direction = self._direction[n_id]
        # aggregate messages
        aggr = torch.cat([
            src_memory * direction[:, 0:1] + dst_memory * direction[:, 1:2],
            src_memory * direction[:, 1:2] + dst_memory * direction[:, 0:1],
            raw_msg,
            time_encoding,
        ], 1)
        memory = self.gru(aggr, src_memory)
        return memory, last_update.long()

    def update_state(
        self,
        memory,
        last_update,
        node_ids,
        node_idx,
        node_t,
        node_msg,
        most_recent,
    ):
        # only write node_ids for [src, pos_dst]
        idx = node_idx[:2]
        flat_idx = torch.reshape(idx, (-1, ))
        write_n_id = node_ids[flat_idx]    # [2xBatch, ] tensor with src&dst ids

        # mask out all but unique most recent src and dst
        masked_indices = torch.reshape(most_recent, (-1, )) * write_n_id   # not most recent will be written to padding node 0
        last_update = last_update[flat_idx]

        dt = node_t.repeat(2) - last_update
        neighbours = write_n_id.roll(int(idx.size(1)))   # swap src and dst for the symmetric triplet
        direction = torch.eye(2).repeat_interleave(idx.shape[1], 0)   # [1,0] for src ids, [0,1] for dst ids
        if self.target == "ipu":
            direction = direction.to(neighbours.device)

        self._memory_ints.index_put_(
            indices=(masked_indices,),
            values=torch.stack([
                node_t.repeat(2).float(),
                dt.float(),
                neighbours.float(),
            ], dim=-1)
        )

        self._memory.index_put_(
            indices=(masked_indices,),
            values=memory[flat_idx]  # memory for src and dst
        )

        self._memory_msg.index_put_(
            indices=(masked_indices,),
            values=node_msg.repeat(2, 1)  # messages btw them
        )

        self._direction.index_put_(
            indices=(masked_indices,),
            values=direction,
        )

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            prev_dtype = self.dtype
            if self.dtype == torch.float16:
                self.float()
                self.dtype = torch.float32
            # Flush message store to memory in case we just entered eval mode.
            memory, last_update = self.__get_memory__(torch.arange(self.num_nodes + 1))
            self._memory_ints = torch.empty(self.num_nodes + 1, 3, dtype=torch.float)
            self._memory = torch.empty(self.num_nodes + 1, self.memory_dim, dtype=self.dtype)
            self._memory_msg = torch.empty(self.num_nodes + 1, self.raw_msg_dim, dtype=self.dtype)
            # Flush message store to memory in case we just entered eval mode.
            self._memory, self._memory_ints[:, 0] = memory, last_update
            if prev_dtype == torch.float16:
                self.dtype = prev_dtype
                self.half()
        super(TGNMemory, self).train(mode)


class TGN(nn.Module):
    def __init__(self,
                 num_nodes,
                 raw_msg_dim,
                 memory_dim,
                 time_dim,
                 embedding_dim,
                 dtype,
                 dropout,
                 target):
        super(TGN, self).__init__()

        # Create an IPU compatible memory module
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            dtype=dtype,
            target=target,
        )
        self.memory.reset_state()

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=raw_msg_dim,
            time_dim=time_dim,
            dropout=dropout,
            dtype=dtype,
        )

        self.link_predictor = LinkPredictor(in_channels=embedding_dim)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        node_ids,
        node_idx,
        node_t,
        node_msg,
        most_recent,
        edge_idx,
        edge_t,
        edge_msg,
    ):
        node_ids = node_ids.squeeze(0)
        node_idx = node_idx.squeeze(0)
        node_t = node_t.squeeze(0)
        node_msg = node_msg.squeeze(0)
        most_recent = most_recent.squeeze(0)
        edge_idx = edge_idx.squeeze(0)
        edge_t = edge_t.squeeze(0)
        edge_msg = edge_msg.squeeze(0)

        memory, last_update = self.memory(node_ids)
        z = self.gnn(memory, last_update, edge_idx, edge_t, edge_msg)

        pos_out = self.link_predictor(z[node_idx[0]], z[node_idx[1]])
        neg_out = self.link_predictor(z[node_idx[0]], z[node_idx[2]])

        # mask out nodes in padded batches
        batch_mask = (node_idx[0] < node_ids.shape[0] - 1).reshape(pos_out.size())
        pos_out *= batch_mask
        neg_out *= batch_mask
        count = torch.sum(batch_mask)

        if self.training:
            loss = self.criterion(pos_out, torch.ones_like(pos_out))
            loss += self.criterion(neg_out, torch.zeros_like(neg_out))

        self.memory.update_state(memory, last_update, node_ids, node_idx,
                                 node_t, node_msg, most_recent)

        if self.training:
            return count, poptorch.identity_loss(loss, "none")
        else:
            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid()
            y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
            return count, y_true, y_pred


# FUNCTIONS
def most_recent_indices(indices):
    """Create a mask for the most recent (rightmost) instance of each index."""
    return ~torch.triu(indices.unsqueeze(0) == indices.unsqueeze(-1), 1).any(1)


def softmax(values, indices, n_indices):
    if values.dim() == 1:
        values = values.reshape(-1, 1)

    n_cols = values.shape[1]
    max_values = torch.amax(values, 0)
    exp_values = torch.exp(values - max_values)

    broad_ix = torch.stack([indices] * n_cols, 1)

    scatter_on = torch.zeros(n_indices, n_cols)
    sum_exp_values = torch.scatter_add(scatter_on, 0, broad_ix, exp_values)

    return exp_values / (sum_exp_values[indices] + 1e-16)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
