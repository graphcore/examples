# Copyright (c) 2022 Graphcore Ltd, All rights reserved.
from pathlib import Path
from typing import Callable

import tensorflow as tf
from tensorflow.python import ipu


def _attribute(
    op: str,
    dtype: tf.DType,
    n_groups: int,
    table_size: int,
    embedding_size: int,
    n_lookup: int,
) -> str:
    return " ".join(
        map(
            str,
            [
                "v0",
                op,
                {tf.dtypes.float32: "float", tf.dtypes.float16: "half"}[dtype],
                n_groups,
                table_size,
                embedding_size,
                n_lookup,
            ],
        )
    )


def grouped_gather(params: tf.Tensor, indices: tf.Tensor) -> (tf.Tensor, Callable):
    """Equivalent to tf.gather(params, indices, batch_dims=1, axis=1).

    params -- T[n_groups x table_size x embedding_size]

    indices -- int32[n_groups x n_lookup]

    returns -- T[n_groups x n_lookup x embedding_size]
    """
    if len(params.shape) != 3:
        raise ValueError(
            "grouped_gather expects `params` to have shape (n_groups, table_size, embedding_size)"
            f", actual shape {params.shape}"
        )
    if len(indices.shape) != 2:
        raise ValueError(
            "grouped_gather expects `indices` to have shape (n_groups, n_lookup)" f", actual shape {indices.shape}"
        )
    if params.shape[0] != indices.shape[0]:
        raise ValueError(
            f"grouped_gather given inconsistent `n_groups` (first) axis"
            f", between `params` (shape {params.shape}) and `indices` (shape {indices.shape})"
        )
    if indices.dtype != tf.dtypes.int32:
        raise ValueError(f"grouped_gather expects indices.dtype == int32 (actual {indices.dtype})")

    @tf.custom_gradient
    def _internal_grouped_gather(_params, _indices):
        n_groups, table_size, embedding_size = map(int, _params.shape)
        _, n_lookup = map(int, _indices.shape)

        (output,) = ipu.custom_ops.precompiled_user_op(
            [_params, _indices],
            library_path=str(Path(__file__).parent / "custom_grouped_gather_scatter.so"),
            attributes=_attribute(
                op="gather",
                dtype=_params.dtype,
                n_groups=n_groups,
                table_size=table_size,
                embedding_size=embedding_size,
                n_lookup=n_lookup,
            ),
            outs=dict(
                output_types=[_params.dtype],
                output_shapes=[tf.TensorShape([n_groups, n_lookup, embedding_size])],
            ),
            name="grouped_gather",
        )

        def grad(upstream: tf.Tensor) -> (tf.Tensor, None):
            return grouped_scatter_sum(upstream, _indices, table_size=table_size), None

        return output, grad

    return _internal_grouped_gather(params, indices)


def grouped_scatter_sum(data: tf.Tensor, indices: tf.Tensor, table_size: int) -> (tf.Tensor, Callable):
    """A grouped version of tf.math.unsorted_segment_sum.

    Equivalent to:

        tf.stack([tf.math.unsorted_segment_sum(data[i], indices[i], table_size)
                  for i in range(data.shape[0])])

    data -- T[n_groups x n_lookup x embedding_size]

    indices -- int32[n_groups x n_lookup] -- in range [0, table_size)

    table_size -- int -- number of table entries to return (for each group)

    returns -- T[n_groups x table_size x embedding_size]
    """
    if len(data.shape) != 3:
        raise ValueError(
            "grouped_scatter expects `data` to have shape (n_groups, n_lookup, embedding_size)"
            f", actual shape {data.shape}"
        )
    if len(indices.shape) != 2:
        raise ValueError(
            "grouped_scatter expects `indices` to have shape (n_groups, n_lookup)" f", actual shape {indices.shape}"
        )
    if data.shape[0] != indices.shape[0]:
        raise ValueError(
            f"grouped_scatter given inconsistent `n_groups` (first) axis"
            f", between `data` (shape {data.shape}) and `indices` (shape {indices.shape})"
        )
    if data.shape[1] != indices.shape[1]:
        raise ValueError(
            f"grouped_scatter given inconsistent `n_lookup` (second) axis"
            f", between `data` (shape {data.shape}) and `indices` (shape {indices.shape})"
        )
    if indices.dtype != tf.dtypes.int32:
        raise ValueError(f"grouped_scatter expects indices.dtype == int32 (actual {indices.dtype})")

    @tf.custom_gradient
    def _internal_grouped_scatter(_data, _indices):
        n_groups, n_lookup, embedding_size = map(int, _data.shape)
        (output,) = ipu.custom_ops.precompiled_user_op(
            [_data, _indices],
            library_path=str(Path(__file__).parent / "custom_grouped_gather_scatter.so"),
            attributes=_attribute(
                op="scatter_sum",
                dtype=_data.dtype,
                n_groups=n_groups,
                table_size=table_size,
                embedding_size=embedding_size,
                n_lookup=n_lookup,
            ),
            outs=dict(
                output_types=[_data.dtype],
                output_shapes=[tf.TensorShape([n_groups, table_size, embedding_size])],
            ),
            name="grouped_scatter",
        )

        def grad(upstream: tf.Tensor) -> (tf.Tensor, None):
            return grouped_gather(upstream, _indices), None

        return output, grad

    return _internal_grouped_scatter(data, indices)


def grouped_scatter_max(
    data: tf.Tensor, indices: tf.Tensor, table_size: int, backwards_mode: str = "sum"
) -> (tf.Tensor, Callable):
    """A grouped version of tf.math.unsorted_segment_sum.

    Equivalent to:

        tf.stack([tf.math.unsorted_segment_sum(data[i], indices[i], table_size)
                  for i in range(data.shape[0])])

    data -- T[n_groups x n_lookup x embedding_size]

    indices -- int32[n_groups x n_lookup] -- in range [0, table_size)

    table_size -- int -- number of table entries to return (for each group)

    returns -- T[n_groups x table_size x embedding_size]
    """
    if len(data.shape) != 3:
        raise ValueError(
            "grouped_scatter expects `data` to have shape (n_groups, n_lookup, embedding_size)"
            f", actual shape {data.shape}"
        )
    if len(indices.shape) != 2:
        raise ValueError(
            "grouped_scatter expects `indices` to have shape (n_groups, n_lookup)" f", actual shape {indices.shape}"
        )
    if data.shape[0] != indices.shape[0]:
        raise ValueError(
            f"grouped_scatter given inconsistent `n_groups` (first) axis"
            f", between `data` (shape {data.shape}) and `indices` (shape {indices.shape})"
        )
    if data.shape[1] != indices.shape[1]:
        raise ValueError(
            f"grouped_scatter given inconsistent `n_lookup` (second) axis"
            f", between `data` (shape {data.shape}) and `indices` (shape {indices.shape})"
        )
    if indices.dtype != tf.dtypes.int32:
        raise ValueError(f"grouped_scatter expects indices.dtype == int32 (actual {indices.dtype})")

    @tf.custom_gradient
    def _internal_grouped_scatter_max(_data, _indices):
        n_groups, n_lookup, embedding_size = map(int, _data.shape)
        (fwd_out,) = ipu.custom_ops.precompiled_user_op(
            [_data, _indices],
            library_path=str(Path(__file__).parent / "custom_grouped_gather_scatter.so"),
            attributes=_attribute(
                op="scatter_max",
                dtype=_data.dtype,
                n_groups=n_groups,
                table_size=table_size,
                embedding_size=embedding_size,
                n_lookup=n_lookup,
            ),
            outs=dict(
                output_types=[_data.dtype],
                output_shapes=[tf.TensorShape([n_groups, table_size, embedding_size])],
            ),
            name="grouped_scatter",
        )

        min_value = -65500 if fwd_out.dtype == tf.float16 else -3.402823e38
        fwd_out = tf.where(fwd_out <= min_value, tf.cast(0, fwd_out.dtype), fwd_out)

        fwd_temp = fwd_out

        def grad(upstream: tf.Tensor) -> (tf.Tensor, None):
            fwd_out = grouped_gather(fwd_temp, _indices)
            mask = tf.where(_data == fwd_out, tf.cast(1, upstream.dtype), tf.cast(0, upstream.dtype))

            if backwards_mode == "mean":
                # to do mean over equal max values must count values in mask and scale upstream
                max_counts = grouped_scatter_sum(mask, _indices, table_size)
                max_counts = tf.maximum(max_counts, tf.constant(1, max_counts.dtype))
                upstream /= max_counts

            upstream = grouped_gather(upstream, _indices)
            return upstream * mask, None

        return fwd_out, grad

    return _internal_grouped_scatter_max(data, indices)
