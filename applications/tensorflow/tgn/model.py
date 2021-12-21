# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Defines a Temporal Graph Network (https://arxiv.org/abs/2006.10637) for IPU."""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import numpy as np
import tensorflow.compat.v1 as tf

import optimiser
import utils

###############################################################################
# Generic helpers

U = TypeVar("U")


def assert_shape(tensor: tf.Tensor, expected: Tuple[Optional[int],
                                                    ...]) -> Tuple[int, ...]:
    """Check tensor shape against expected, ignoring None, returning `tensor.shape`."""
    actual = tensor.shape
    match = len(actual) == len(expected) and all(
        y is None or x == y for x, y in zip(actual, expected))
    assert match, f"wrong shape, expected {expected}, actual {actual}"
    return actual


def scoped_fn(fn: Callable[..., U]) -> Callable[..., U]:
    """Wrap a function with a variable scope, named with the function name."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tf.variable_scope(fn.__name__):
            return fn(*args, **kwargs)

    return wrapper


@scoped_fn
def index_softmax(values: tf.Tensor, indices: tf.Tensor,
                  n_indices: int) -> tf.Tensor:
    """Compute multiple softmax() in groups defined by indices.

    E.g.
        index_softmax([0, 0, ln(2), 2], [0, 0, 0, 1], 2)
          computes softmax([0, 0, ln(2)]) and softmax([2])
        => [0.25, 0.25, 0.5, 1.0]

    Acts over axis=0 of values.
    """
    # Run everything in float32, for stability
    dtype = values.dtype
    values = tf.cast(values, tf.float32)

    max_values = tf.reduce_max(values, axis=0, keepdims=True)
    exp_values = tf.exp(values - max_values)
    # Max(*, 1e-6) prevents a DIV0 error, caused by underflow of the sum-exp.
    sum_exp_values = tf.maximum(
        tf.unsorted_segment_sum(exp_values, indices, n_indices), 1e-6)
    return tf.cast(exp_values / tf.gather(sum_exp_values, indices), dtype)


@scoped_fn
def linear(input: tf.Tensor,
           n_output: int,
           use_bias: bool = True) -> tf.Tensor:
    """A standard linear layer `W x + b`."""
    weight = tf.get_variable(
        "weight",
        dtype=input.dtype,
        shape=(input.shape[-1], n_output),
        initializer=tf.glorot_normal_initializer(),
    )
    output = input @ weight
    if use_bias:
        bias = tf.get_variable(
            "bias",
            dtype=input.dtype,
            shape=(n_output, ),
            initializer=tf.zeros_initializer(),
        )
        output += bias
    return output


@scoped_fn
def cos_fp16(x: tf.Tensor) -> tf.Tensor:
    """Run cos(x) in FP16, first running mod(x, 2*pi) for range safety."""
    if x.dtype == tf.float16:
        return tf.cos(x)
    x_16 = tf.cast(tf.mod(x, 2 * np.pi), tf.float16)
    return tf.cos(x_16)


@scoped_fn
def gru_cell(prev_hidden: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
    """Compute a single step of a GRU (following the PyTorch parameterization).

    See PyTorch GRUCell, https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
    for the definition of this operation & trainable variables.

    Arguments:

      prev_hidden -- shape (batch_size x hidden_size), the previous GRU state,
                     e.g. returned by gru_cell

      input -- shape (batch_size x input_size)

    Returns:

      tensor of shape (batch_size x hidden_size), a new GRU state
    """
    batch_size, hidden_size = assert_shape(prev_hidden, (None, None))
    _, input_size = assert_shape(input, (batch_size, None))
    dtype = prev_hidden.dtype

    weight_i = tf.get_variable(
        "weight_i",
        (3, input_size, hidden_size),
        dtype=dtype,
        initializer=tf.glorot_normal_initializer(),
    )
    bias_i = tf.get_variable("bias_i", (3, hidden_size),
                             dtype=dtype,
                             initializer=tf.zeros_initializer())
    weight_h = tf.get_variable(
        "weight_h",
        (3, hidden_size, hidden_size),
        dtype=dtype,
        initializer=tf.glorot_normal_initializer(),
    )
    bias_h = tf.get_variable("bias_h", (3, hidden_size),
                             dtype=dtype,
                             initializer=tf.zeros_initializer())

    reset_i, update_i, candidate_i = tf.unstack(input @ weight_i +
                                                tf.expand_dims(bias_i, 1))
    reset_h, update_h, candidate_h = tf.unstack(prev_hidden @ weight_h +
                                                tf.expand_dims(bias_h, 1))

    reset = tf.sigmoid(reset_i + reset_h)
    update = tf.sigmoid(update_i + update_h)
    candidate = tf.tanh(candidate_i + reset * candidate_h)
    return (1 - update) * candidate + update * prev_hidden


@scoped_fn
def transformer_conv(
    n_output: int,
    n_heads: int,
    dropout: float,
    nodes: tf.Tensor,
    edge_idx: tf.Tensor,
    edges: tf.Tensor,
) -> tf.Tensor:
    """Implementation of Graph Transformer, https://arxiv.org/abs/2009.03509.

    Matches the specification of TransformerConv in PyTorch Geometric, always using
    a "skip" projection from inputs and shared key/value projections for edges.

    Arguments:

      n_output -- output feature size

      n_heads -- number of attention heads (note: head size is given by n_output/n_heads)

      dropout -- rate parameter for attention mask (post-softmax) dropout

      nodes -- shape (n_nodes, node_feature_size), input features for each node

      edge_idx -- shape (2, n_edges), (0 <= edge_idx < n_nodes), the source and
                  destination of each edge, indexing into nodes

      edges -- shape (n_edges, edge_feature_size), input features for each edge

    Returns:

      tensor of shape (n_nodes, n_output), node features after applying a graph
      transformer (attention) layer
    """
    assert n_output % n_heads == 0, \
        "graph transformer output size should be divisible by the number of heads"
    head_size = n_output // n_heads
    n_nodes, _ = assert_shape(nodes, (None, None))
    _, n_edges = assert_shape(edge_idx, (2, None))
    assert_shape(edges, (n_edges, None))

    with tf.variable_scope("skip"):
        skip = linear(nodes, n_output)

    with tf.variable_scope("edge_shared_kv"):
        edge_kv = linear(edges, n_output, use_bias=False)

    with tf.variable_scope("node_qkv"):
        node_qkv = linear(nodes, 3 * n_output)

    with tf.variable_scope("attention"):
        q = tf.gather(node_qkv[:, :n_output], edge_idx[1])
        kv = tf.reshape(
            tf.gather(node_qkv[:, n_output:], edge_idx[0]),
            (n_edges, 2, n_output),
        )
        k, v = tf.unstack(kv + edge_kv[:, tf.newaxis, :], axis=1)
        a = tf.reduce_sum(tf.reshape(q * k, (n_edges, n_heads, head_size)),
                          -1) / (head_size**0.5)
        a = index_softmax(a, edge_idx[1], n_nodes)
        if dropout:
            a = tf.nn.dropout(a, rate=dropout)
        attention = tf.unsorted_segment_sum(
            tf.repeat(a, head_size, axis=1) * v, edge_idx[1], n_nodes)

    return skip + attention


@scoped_fn
def time_encoder(dt: tf.Tensor, size: int, dtype: tf.DType) -> tf.Tensor:
    """Create TGN time encoder cos(dt @ weight + bias)."""
    weight = tf.get_variable(
        "weight",
        (size, ),
        dtype=dt.dtype,
        initializer=tf.random_normal_initializer(stddev=0.1),
    )
    bias = tf.get_variable("bias", (size, ),
                           dtype=dt.dtype,
                           initializer=tf.zeros_initializer())
    cos = cos_fp16 if dtype == tf.float16 else tf.cos
    return cos(dt[..., tf.newaxis] * weight + bias)


@dataclass
class TgnMemory:
    """Outputs from tgn_memory()."""

    output: tf.Tensor
    last_update: tf.Tensor
    updates: Tuple[tf.Tensor, ...]


TGN_MEMORY_VARIABLES_KEY = "tgn_memory_variables"


@scoped_fn
def tgn_memory(
    n_nodes: int,
    memory_size: int,
    time_embedding_size: int,
    node_ids: tf.Tensor,
    write_idx: tf.Tensor,
    write_mask: tf.Tensor,
    write_features: tf.Tensor,
    write_times: tf.Tensor,
) -> TgnMemory:
    """Create TGN memory read & update operations.

    A trainable memory for nodes in an temporal interaction graph. The memory
    state is computed using the latest interaction event that touched a node.
    The update is a GRU cell, taking as input the previous memory of both source
    and desination nodes for that edge, the edge feature vector and time difference
    from interaction to current time.

    Note that the GRU cell is computed lazily when the memory is read, rather than
    when it is stored, to support a single step of truncated backpropagation through
    time and obtain a gradient for GRU variables.

    Please see "Temporal Graph Network" (https://arxiv.org/abs/2006.10637) for full
    details.

    Arguments:

      n_nodes -- total number of slots in the memory

      memory_size -- size of stored state in the memory / GRU cell output size

      time_embedding_size -- size of the time encoding activation provided to the
                             GRU cell

      node_ids -- shape (n_read), (-1 <= ID < n_nodes), the memory locations to be read

      write_idx -- shape (2, n_write), (0 <= idx < n_read), the (src, dst) indices of
                   edges, selecting nodes that should be written with their updated
                   memory state

      write_mask -- shape (2, n_write), boolean tensor for elements in write_idx that
                    should be written (true) or skipped (false), such that each memory
                    location is written at most once

      write_features -- shape (n_write, feature_size), input features to be stored and
                        used to compute the memory when it is next accessed

      write_times -- shape (n_write), edge event times to be stored and used to compute
                     the memory when it next accessed

    Returns:

      TgnMemory(
        output      -- tensor of shape (n_read, memory_size), current memory for node_ids
        last_update -- tensor of shape (n_read), last update of output
        updates     -- tuple of operations to run to update the memory
      )
    """
    assert_shape(node_ids, (None, ))
    _, n_write = assert_shape(write_idx, (2, None))
    assert_shape(write_mask, (2, n_write))
    _, feature_size = assert_shape(write_features, (n_write, None))
    assert_shape(write_times, (n_write, ))
    dtype = write_features.dtype

    # Declare memory
    # As an optimisation, we concatenate the 6 fields required by the memory
    # into 2 tensors, one consisting of ints, the other of floats.
    # This requires some extra code to slice and concat, but means we can use
    # 2 (dynamic) gather operations instead of 6.

    # Each row: [last_update, dt, neighbour]
    v_ints = tf.get_variable(
        "ints",
        shape=(1 + n_nodes, 3),
        dtype=tf.int32,
        trainable=False,
        initializer=tf.zeros_initializer(),
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, TGN_MEMORY_VARIABLES_KEY],
    )
    # Each row: [memory, features, direction]
    v_floats = tf.get_variable(
        "floats",
        shape=(1 + n_nodes, memory_size + feature_size + 2),
        dtype=dtype,
        trainable=False,
        initializer=tf.zeros_initializer(),
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, TGN_MEMORY_VARIABLES_KEY],
    )

    # Memory[0] is used for padding (node_ids == -1)
    safe_node_ids = 1 + node_ids

    # Read memory for node_ids
    node_ints = tf.gather(v_ints, safe_node_ids)
    node_last_update, node_dt, node_neighbour_idx = tf.unstack(node_ints,
                                                               axis=1)
    node_neighbour = tf.gather(v_floats[:, :memory_size], node_neighbour_idx)
    node_time_encoding = time_encoder(tf.cast(node_dt, tf.float32),
                                      time_embedding_size, dtype)

    node_floats = tf.gather(v_floats, safe_node_ids)
    node_self = node_floats[:, :memory_size]
    node_features = node_floats[:, memory_size:memory_size + feature_size]
    node_direction = node_floats[:, memory_size + feature_size:]

    node_memory = gru_cell(
        node_self,
        tf.concat(
            [
                node_direction[:, 0, tf.newaxis] * node_self +
                node_direction[:, 1, tf.newaxis] * node_neighbour,
                node_direction[:, 1, tf.newaxis] * node_self +
                node_direction[:, 0, tf.newaxis] * node_neighbour,
                node_features,
                node_time_encoding,
            ],
            axis=1,
        ),
    )

    # Write memory according to (write_idx, write_mask)
    flat_write_idx = tf.reshape(write_idx, (-1, ))
    indices = tf.gather(safe_node_ids, flat_write_idx)
    masked_indices = indices * tf.cast(tf.reshape(write_mask,
                                                  (-1, )), indices.dtype)
    p_last_update = tf.reshape(tf.tile(write_times[tf.newaxis], (2, 1)),
                               (-1, ))
    p_dt = p_last_update - tf.gather(node_last_update, flat_write_idx)
    # Swap src and dst indices to get the neighbour index for each node
    p_neighbour = tf.roll(indices, n_write, 0)
    p_memory = tf.gather(node_memory, flat_write_idx)
    p_features = tf.tile(write_features, (2, 1))
    p_direction = tf.repeat(tf.eye(2, dtype=dtype), n_write,
                            0)  # src=[1, 0], dst=[0, 1]

    # There is already a data dependency, but just to be sure...
    with tf.control_dependencies([node_last_update, node_memory]):
        update_ints = v_ints.scatter_update(
            tf.IndexedSlices(
                tf.stack([p_last_update, p_dt, p_neighbour], axis=1),
                masked_indices))
        update_floats = v_floats.scatter_update(
            tf.IndexedSlices(
                tf.concat([p_memory, p_features, p_direction], axis=1),
                masked_indices))

    return TgnMemory(
        output=node_memory,
        last_update=node_last_update,
        updates=(update_ints, update_floats),
    )


@scoped_fn
def tgn_gnn(
    time_embedding_size: int,
    dropout: float,
    input: tf.Tensor,
    last_update: tf.Tensor,
    edge_idx: tf.Tensor,
    edge_times: tf.Tensor,
    edge_features: tf.Tensor,
) -> tf.Tensor:
    """The 'core' GNN from TGN, with time encoder & graph transformer.

    Computes transformed representations for a set of nodes, based on a set of
    interactions (edges) involving those nodes.

    Arguments:

      time_embedding_size -- number of features to use for the time encoding,
                             which is concatenated to edge_features for the GNN
                             step

      dropout -- rate parameter for transformer_conv

      input -- shape (n_nodes, memory_size), input node features (from memory)

      last_update -- shape (n_nodes), timestamps for the last memory update that
                     produced the input

      edge_idx -- shape (2, n_edges), indexing into input and last_update

      edge_times -- shape (n_edges), timestamps for current set of edges

      edge_features -- shape (n_edges, feature_size), input features for current
                       set of edges

    Returns:

      tensor of shape (n_nodes, memory_size) -- node output features
    """
    n_nodes, n_features = assert_shape(input, (None, None))
    assert_shape(last_update, (n_nodes, ))
    _, n_edges = assert_shape(edge_idx, (2, None))
    assert_shape(edge_times, (n_edges, ))
    assert_shape(edge_features, (n_edges, None))

    dt = tf.gather(last_update, edge_idx[0]) - edge_times
    time_encoding = time_encoder(tf.cast(dt, tf.float32), time_embedding_size,
                                 input.dtype)
    return transformer_conv(
        int(n_features),
        n_heads=2,
        dropout=dropout,
        nodes=input,
        edge_idx=edge_idx,
        edges=tf.concat([edge_features, time_encoding], axis=1),
    )


@scoped_fn
def tgn_link_predictor(src: tf.Tensor, dst: tf.Tensor) -> tf.Tensor:
    """Predict the logit for a link between src & dst.

    Implemented as a ReLU MLP with 1 hidden layer and 1 output.

    Arguments:

      src -- shape (* x feature_size), source node features

      dst -- shape (* x feature_size), destination node features

    Returns:

      tensor of shape (*), scores for each paired src and dst
    """
    assert src.shape == dst.shape
    feature_size = int(src.shape[-1])

    with tf.variable_scope("hidden"):
        hidden = tf.nn.relu(
            linear(tf.concat([src, dst], axis=-1), feature_size))
    with tf.variable_scope("output"):
        return linear(hidden, 1)[..., 0]


@scoped_fn
def tgn(
    # Settings
    n_nodes: int,
    memory_size: int,
    time_embedding_size: int,
    dropout: float,
    learning_rate: float,
    target: utils.Target,
    is_training: bool,
    # Inputs
    node_ids: tf.Tensor,
    batch_idx: tf.Tensor,
    batch_times: tf.Tensor,
    batch_features: tf.Tensor,
    batch_most_recent: tf.Tensor,
    edge_idx: tf.Tensor,
    edge_times: tf.Tensor,
    edge_features: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """Complete TGN including memory read/update, GNN and optional optimisation.

    Processes a batch of intearction events, pairs of (src, pos_dst), which update
    the node memory and predicts the probability of an event between (src, pos_dst)
    and between (src, neg_dst) to give a contrastive loss.

    See the component functions tgn_memory(), tgn_gnn(), tgn_link_predictor() for a
    functional description.

    Please see "Temporal Graph Network" (https://arxiv.org/abs/2006.10637) for full
    details.

    Arguments:

      n_nodes -- total number of slots in the memory

      memory_size -- size of stored state in the memory / GRU cell output size

      time_embedding_size -- size of the time encoding activation provided to the
                             GRU cell

      dropout -- rate parameter for transformer_conv() via tgn_gnn()

      learning_rate -- for Adam (training only)

      target -- device to execute on, note: this influences optimal mixed precision
                design

      is_training -- boolean flag enabling training: optimiser step and dropout

      node_ids -- shape (n_read), nodes to be read in this step

      batch_idx -- shape (3, batch_size), indices [src, pos_dst, neg_dst], indexing
                   into node_ids for each interaction event, paired with a negative
                   sample

      batch_times -- shape (batch_size), timestamps for each event

      batch_features -- shape (batch_size, feature_size), input features for each
                        event

      batch_most_recent -- shape (2, batch_size), boolean mask for the [src, pos_dst]
                           values in batch_idx that are most recent within the batch,
                           used to prevent write hazards in the memory

      edge_idx -- shape (2, edges_size), indices [src, dst] into node_ids, a history
                  of a few recent edges to add additional context separate from the
                  memory

      edge_times -- shape (edges_size), timestamps for the edge history

      edge_features -- shape (edges_size, feature_size), features for the edge history

    Returns:

      {"loss":  tensor of shape (), mean loss over non-masked (src, dst) pairs
       "count": tensor of shape (), number of non-masked pairs
       "probs": tensor of shape (2, batch_size), unless is_training, probability
                of consecutive link per pair, for calculating validation statistics
      }
    """

    memory = tgn_memory(
        n_nodes=n_nodes,
        memory_size=memory_size,
        time_embedding_size=time_embedding_size,
        node_ids=node_ids,
        write_idx=batch_idx[:2],
        write_mask=batch_most_recent,
        write_features=batch_features,
        write_times=batch_times,
    )

    hidden = tgn_gnn(
        time_embedding_size=time_embedding_size,
        dropout=is_training * dropout,
        input=memory.output,
        last_update=memory.last_update,
        edge_idx=edge_idx,
        edge_times=edge_times,
        edge_features=edge_features,
    )

    logits = tgn_link_predictor(
        tf.gather(hidden, tf.tile(batch_idx[0][tf.newaxis], (2, 1))),
        tf.gather(hidden, batch_idx[1:]),
    )

    # Masks any batch padding
    batch_mask = tf.not_equal(batch_idx[0], node_ids.shape[0] - 1)
    count = tf.reduce_sum(tf.cast(batch_mask, tf.int32))
    labels = tf.tile(tf.constant([[1], [0]], dtype=logits.dtype),
                     (1, logits.shape[1]))
    # *2 because the reference uses mean(pos_loss) + mean(neg_loss)
    loss = 2 * tf.reduce_mean(
        tf.cast(batch_mask, logits.dtype) *
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    if is_training:
        if target is utils.Target.IPU:
            step = optimiser.Adam(
                learning_rate=learning_rate).minimize_with_global_step(loss)
        else:
            # Allows AMP with TF_ENABLE_AUTO_MIXED_PRECISION=1
            step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss)
        with tf.control_dependencies(memory.updates + (step, )):
            return dict(loss=tf.identity(loss), count=count)
    else:
        with tf.control_dependencies(memory.updates):
            return dict(loss=tf.identity(loss),
                        count=count,
                        probs=tf.nn.sigmoid(logits))
