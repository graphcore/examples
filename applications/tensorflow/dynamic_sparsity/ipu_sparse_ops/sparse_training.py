# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
This module provides host-side functionality for implementing sparse training
algorithms by manipulating sparse matrices/tensors represented as triplets/COO
format.
"""
import numpy as np
import os
import logging
from ipu_sparse_ops import sparse

logger = logging.getLogger(os.path.basename(__file__))


def prune_bottom_k_weights(row_indices, col_indices, nz_values, slot_values, k, debug_name: str):
    """
    Given triplets of sparse weights, find the bottom-k (by absolute value) and return new triplets with
    those low valued weights removed. If momentum_values are provided then remove the equivalent indices
    from that list of values.
    :param row_indices: List of row indices for the non-zero weight values.
    :param col_indices: List of column indices for the non-zero weight values.
    :param nz_values: List of non-zero values at the corresponding row/col indices.
    :param slot_values: Non-zero values for the sparse slots which share row/col coordinates with
                        the weights. (Optional - can be None)
    :param k: The number of weights to remove from the sparse set.
    :param debug_name: Name to uniquely identify the logging output for this call.
    """

    def bottom_k(a, k):
        return np.argpartition(a, k)[0:k]

    triplets = (row_indices, col_indices, nz_values)

    # Find the indices of the lowest magnitude weights:
    lowest_weight_idx = bottom_k(np.abs(triplets[2]), k)
    logger.debug(f"Bottom {k} value indices for layer {debug_name}: {lowest_weight_idx}")
    logger.debug(f"Average trained triplet weights for layer {debug_name}: {np.mean(triplets[2])}")
    prune_row_indices = triplets[0][lowest_weight_idx]
    prune_col_indices = triplets[1][lowest_weight_idx]
    logger.debug(f"Weight indices to prune: {debug_name}: {list(zip(prune_row_indices, prune_col_indices))}")
    logger.debug(f"Weight values to prune: {debug_name}: {triplets[2][lowest_weight_idx]}")
    if len(prune_row_indices) != k:
        raise RuntimeError(f"Pruned {len(prune_row_indices)} indices but expected to prune {k}")

    # Make new triplets with these indices removed:
    remaining_weight_triplets = [np.delete(t, lowest_weight_idx) for t in triplets]

    # Prune the same indices from the slot triplets:
    slot_values = slot_values or {}
    remaining_slot_values = {
        name: np.delete(values, lowest_weight_idx)
        for name, values in slot_values.items()
    }

    if len(remaining_weight_triplets[0]) + k != len(triplets[0]):
        raise RuntimeError(f"Remaining index count {len(remaining_weight_triplets[0])} is not the correct size: {len(triplets[0]) - k}")
    return remaining_weight_triplets, remaining_slot_values


def regrow_rigl(unpruned_triplets, dense_grad_w, new_value_gen, grow_count, debug_name):
    """
    Given an original set of sparse weights represented as triplets, the dense gradient of the loss wrt those weights,
    and a count of the number of triplets that have been or will be pruned, compute a new set of triplets at coordinates
    that have the highest absolute values of gradients. Also prunes the corresponding momentum values if necessary.
    :param unpruned_triplets: Triplets representing the sparse weights before pruning.
    :param dense_grad_w: Dense gradient of loss with respect to the weights.
    :param new_value_gen: A callable that will generate a specified set of new weight values for the new connections.
    :param grow_count: The number of weights to grow/number of new connections to return.
    :param debug_name: Name to uniquely identify the logging output for this call.
    """
    # Get flat indices for the original index set:
    original_flat_idx = np.ravel_multi_index((unpruned_triplets[0], unpruned_triplets[1]), dense_grad_w.shape)

    # We want to grow back weights at the positions with the highest gradient
    # magnitudes that are also not in the original set:
    abs_grad_flat = np.abs(dense_grad_w.flatten())
    abs_grad_flat[original_flat_idx] = -1  # This has the effect of excluding existing indices from the top-k
    argsorted = np.argsort(-abs_grad_flat)
    topk_flat_idx = argsorted[0:grow_count]
    common = np.intersect1d(topk_flat_idx, original_flat_idx)
    if len(common):
        raise RuntimeError("Intersection of new and original indices must be empty.")
    logger.debug(f"Final non intersecting indices: {topk_flat_idx}")

    # Check the indices are unique before we use them:
    unique = np.unique(topk_flat_idx)
    duplicates = len(topk_flat_idx) - len(unique)
    if duplicates != 0:
        print(f"New indices contain {duplicates} duplicates:\n{topk_flat_idx}")
        raise RuntimeError("New indices are not unique")

    top_k_idx = np.unravel_index(topk_flat_idx, dense_grad_w.shape)
    logger.debug(f"Layer {debug_name} weight grad top-k indices: {top_k_idx}")
    return (top_k_idx[0], top_k_idx[1], new_value_gen(size=grow_count))


def zero_values_generator(size=1):
    """
    Return a list of zeros of the specified length.
    :param size: Number of zeros to produce in one call.
    """
    return [0]*size


def join_triplets(kept_weight_triplets, new_weight_triplets, kept_slot_values, pruned_count):
    """
    Join two sets of weight triplets (typically ones we want to keep and some new ones we want to activate).
    Also adds slots for the new connections initialised to zero (Optional).
    :param kept_weight_triplets: Tuple of triplets (rows, cols, values) for the sparse weights being kept.
    :param new_weight_triplets: Tuple of triplets (rows, cols, values) for the new sparse weights being added.
    :param kept_slot_values: Non-zero slot values corresponding to the weights in 'kept_weight_triplets'.
    :param pruned_count:
    """
    # Join the triplets we kept with the newly generated ones:
    grown_rows = np.concatenate([kept_weight_triplets[0], new_weight_triplets[0]]).astype(int)
    grown_cols = np.concatenate([kept_weight_triplets[1], new_weight_triplets[1]]).astype(int)
    grown_values = np.concatenate([kept_weight_triplets[2], new_weight_triplets[2]])
    # Slots for new weights are set to zero:
    kept_slot_values = kept_slot_values or {}
    grown_slots = {
        name: np.concatenate([
            values,
            zero_values_generator(size=pruned_count)
        ])
        for name, values in kept_slot_values.items()
    }
    return (grown_rows, grown_cols, grown_values), grown_slots


def prune_and_grow(name, triplets, shape,
                   spec, max_non_zeros,
                   slot_triplets, prune_schedule,
                   prune_ratio: float, grad_w: np.array,
                   grow_method='rigl', random_gen=None):
    """
    Performs the pruning and growing of the weights to update the sparsity pattern, for the current fc layer
    :param name: A debug name
    :param triplets: Current triplets to prune and grow
    :param shape: Shape of the dense matrix
    :param spec: Specs of the sparse matmul
    :param max_non_zeros: Maximum number of non-zeros values
    :slot_triplets: Triplets for the current slots
    :param prune_schedule: a function which given max prune count returns the number of weights to update
    :param prune_ratio: the maximum percentage of this layer's weights to update
    :param grad_w: A numpy array containing the dense gradients for each sub-layer
    :param grow_method: Method used to regrow the weights, either rigl or random selection
    :param random_gen: Random number generator to be used if the grow_method is 'random'
    """
    if isinstance(grad_w, list):
        grad_w = grad_w[0]
    # Compute the prune count using the provides schedule, the max numbe rof zeros in the
    # layer and the pruning ratio
    prune_count = prune_schedule(max_pruned=int(np.ceil(prune_ratio * max_non_zeros)))

    # Prune bottom k weights
    slot_values = {name: triplet.values
                   for name, triplet in slot_triplets.items()}

    remaining_triplets, remaining_slot_values = prune_bottom_k_weights(
        *triplets, slot_values, prune_count, name)
    zero_values = zero_values_generator
    # regrow weights
    if grow_method == 'rigl':
        # Grow back new indices using Rig-L: (https://arxiv.org/abs/1911.11134)
        if shape != grad_w.shape:
            raise RuntimeError(f"Dense weight gradient has unexpected shape.Expected {shape}, got {grad_w.shape}")
        new_triplets = regrow_rigl(
            triplets, grad_w, zero_values, prune_count, name)
    if grow_method == 'random':
        # Random replacement strategy: add back random indices
        # Gen some replacement random indices excluding all the existing
        # ones then we will swap for the pruned ones:
        new_triplets = sparse.random_triplets(spec, indices_initialiser_gen=random_gen,
                                              value_generator=zero_values,
                                              excluded_indices=(triplets[0], triplets[1]), count=prune_count)

    grown_triplets, grown_slots = join_triplets(
        remaining_triplets, new_triplets, remaining_slot_values, prune_count)
    if len(grown_triplets[0]) != max_non_zeros:
        raise ValueError(f"Grown row count {len(grown_triplets[0])} does not match expected count {max_non_zeros}")
    if len(grown_triplets[1]) != max_non_zeros:
        raise ValueError(f"Grown col count {len(grown_triplets[1])} does not match expected count {max_non_zeros}")
    if len(grown_triplets[2]) != max_non_zeros:
        raise ValueError(f"Grown col count {len(grown_triplets[2])} does not match expected count {max_non_zeros}")
    for grown_slot in grown_slots.values():
        if len(grown_slot) != max_non_zeros:
            raise ValueError(f"Grown col count {len(grown_slot)} does not match expected count {max_non_zeros}")

    grown_triplets = sparse.Triplets(grown_triplets[0], grown_triplets[1], grown_triplets[2])
    grown_slots = {name: [grown_triplets[0], grown_triplets[1], grown_slot]
                   for name, grown_slot in grown_slots.items()}

    return {'gt': grown_triplets, 'nt': new_triplets, 'rt': remaining_triplets, 'gs': grown_slots}
