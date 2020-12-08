__all__ = [
    'DenseFcLayer',
    'SparseFcLayer',
    'SparseTiedEmbedding']

from ipu_sparse_ops.layers.fully_connected import (
    DenseFcLayer as DenseFcLayer,
    SparseFcLayer as SparseFcLayer)

from ipu_sparse_ops.layers.embedding import (
    SparseTiedEmbedding as SparseTiedEmbedding)
