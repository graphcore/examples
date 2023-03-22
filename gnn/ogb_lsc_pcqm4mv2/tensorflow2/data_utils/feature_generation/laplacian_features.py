# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright (c) 2022 Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Dominique Beaini
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file has been modified by Graphcore Ltd.

import numpy as np
import scipy

from data_utils.feature_generation.utils import get_laplacian

# This section has been adapted from the Graph-GPS code and converted to numpy


def eigvec_normalizer(eig_vecs, eig_vals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """
    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = np.linalg.norm(eig_vecs, ord=1, axis=0, keepdims=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = np.linalg.norm(eig_vecs, ord=2, axis=0, keepdims=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = np.max(np.abs(eig_vecs), axis=0)

    elif normalization == "wavelength":
        # These are placeholders that we can fill if needed.
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    elif normalization == "wavelength-asin":
        # These are placeholders that we can fill if needed.
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    elif normalization == "wavelength-soft":
        # These are placeholders that we can fill if needed.
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    else:
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    denom = np.tile(np.clip(denom, a_min=eps, a_max=None), (eig_vecs.shape[0], 1))
    eig_vecs = eig_vecs / denom

    return eig_vecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    num_nodes = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = np.clip(evals, a_min=0, a_max=None)

    # Normalize and pad eigen vectors.
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if num_nodes < max_freqs:
        eig_vecs = np.pad(evects, ((0, 0), (0, max_freqs - num_nodes)), "constant", constant_values=np.nan)
    else:
        eig_vecs = evects

    # Pad and save eigenvalues.
    if num_nodes < max_freqs:
        eig_vals = np.pad(evals, (0, max_freqs - num_nodes), "constant", constant_values=np.nan)
    else:
        eig_vals = evals

    eig_vals = np.tile(eig_vals, (num_nodes, 1))
    eig_vals = np.expand_dims(eig_vals, axis=-1)

    return eig_vals, eig_vecs


def get_laplacian_features(
    data, max_freqs=3, eigvec_norm="L2", eigval_inverse=False, eigval_norm=False, remove_first=False
):
    # Eigen values and vectors.
    evals, evects = None, None
    # Basic preprocessing of the input graph.
    num_nodes = data["num_nodes"]  # Explicitly given number of nodes
    edge_index = data["edge_index"]

    # If only one node or no edges, set eigenvalue/vector to zero
    if num_nodes == 1 or edge_index.shape[1] < 1:
        evals = np.zeros((num_nodes, max_freqs, 1))  # eigenvalue size (num_nodes, max_freqs, 1)
        evects = np.zeros((num_nodes, max_freqs))  # eigenvector size (num_nodes, max_freqs)
    else:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        edge_list, edge_attr = get_laplacian(edge_index, None, num_nodes=num_nodes)
        row, col = edge_list
        assert edge_attr.shape[0] == row.shape[0]
        laplacian = scipy.sparse.coo_matrix((edge_attr, (row, col)), (num_nodes, num_nodes))
        evals, evects = np.linalg.eigh(
            laplacian.astype(np.float32).toarray()
        )  # Hack to convert to float32 for linear algebra

        evals, evects = get_lap_decomp_stats(evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm)

    # If remove_first is true, remove the smallest eigen value (which is 0) and first eigen vector (which are all 1s)
    # the size then becomes [num_nodes, max_freq-1]
    if remove_first:
        evals = evals[:, 1:]
        evects = evects[:, 1:]

    # If eigval_inverse is true, take the inverse of the eigen values (apart from values close to 0, will stay the same)
    # recommend to use with the remove_first
    if eigval_inverse:
        # remove nans and fill with 0
        nan_mask = np.isnan(evals)
        evals[nan_mask] = 0.0
        evals[evals > 1e-10] = 1.0 / evals[evals > 1e-10]

    # If eigval_norm is true, normalize the eigen values, recommend to use with the eigval_inverse and remove_first
    if eigval_norm:
        # this is doing the Frobenius norm by default
        # basically sum(eval_i^2)^(1/2)
        norm = np.linalg.norm(evals[0])
        if norm == 0:
            evals = evals
        else:
            evals = evals / norm
    if remove_first:
        assert evals.shape == (num_nodes, max_freqs - 1, 1)
        assert evects.shape == (num_nodes, max_freqs - 1)
    else:
        assert evals.shape == (num_nodes, max_freqs, 1)
        assert evects.shape == (num_nodes, max_freqs)

    return (evals, evects)


def get_laplacian_features_from_dataset(dataset_item, item_options):
    lap_feats = get_laplacian_features(dataset_item, **item_options)
    return lap_feats
