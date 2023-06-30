# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path

import tensorflow as tf
from tensorflow.python import ipu


class CustomOpsNotFoundException(Exception):
    """Raised when the custom ops .so file is not found."""

    pass


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


def grouped_gather(params: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
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

    custom_op_path = Path(__file__).parent.joinpath("build/custom_grouped_gather_scatter.so")
    if not custom_op_path.exists():
        raise CustomOpsNotFoundException(
            f"`{custom_op_path}` not found. Please run `make` in the `./static_ops` directory first."
        )

    n_groups, table_size, embedding_size = map(int, params.shape)
    _, n_lookup = map(int, indices.shape)

    (output,) = ipu.custom_ops.precompiled_user_op(
        [params, indices],
        library_path=str(custom_op_path),
        attributes=_attribute(
            op="gather",
            dtype=params.dtype,
            n_groups=n_groups,
            table_size=table_size,
            embedding_size=embedding_size,
            n_lookup=n_lookup,
        ),
        outs=dict(
            output_types=[params.dtype],
            output_shapes=[tf.TensorShape([n_groups, n_lookup, embedding_size])],
        ),
        name="grouped_gather",
    )
    return output


def grouped_scatter(data: tf.Tensor, indices: tf.Tensor, table_size: int) -> tf.Tensor:
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

    custom_op_path = Path(__file__).parent.joinpath("build/custom_grouped_gather_scatter.so")
    if not custom_op_path.exists():
        raise CustomOpsNotFoundException(
            f"`{custom_op_path}` not found. Please run `make` in the `./static_ops` directory first."
        )

    n_groups, n_lookup, embedding_size = map(int, data.shape)

    (output,) = ipu.custom_ops.precompiled_user_op(
        [data, indices],
        library_path=str(custom_op_path),
        attributes=_attribute(
            op="scatter",
            dtype=data.dtype,
            n_groups=n_groups,
            table_size=table_size,
            embedding_size=embedding_size,
            n_lookup=n_lookup,
        ),
        outs=dict(
            output_types=[data.dtype],
            output_shapes=[tf.TensorShape([n_groups, table_size, embedding_size])],
        ),
        name="grouped_scatter",
    )
    return output
