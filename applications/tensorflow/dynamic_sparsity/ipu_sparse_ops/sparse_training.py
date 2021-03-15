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


def plot_and_log_matrix(name, matrix):
    try:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        import wandb
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import sys
        plt.clf()
        fig = plt.figure(figsize=(4, 4))
        plt.axis('equal')
        sns.set(font_scale=0.8)
        sns.heatmap(matrix, cmap='magma', ax=plt.gca(), norm=LogNorm(vmin=sys.float_info.min, clip=True))
        wandb.log({name: wandb.Image(plt)}, commit=False)
        plt.close(fig)
    except Exception as e:
        # Don't lose a run because plotting went wrong!
        logger.warn(f"Plot logging failed: {e}")
        pass


def prune_bottom_k_weights(row_indices, col_indices, nz_values, slot_values, k, debug_name: str):
    """
    Given triplets of sparse weights, find the bottom-k (by absolute value) and return new triplets with
    those low valued weights removed. If momentum_values are provided then remove the equivalent indices
    from that list of values.
    :param row_indices: List of row indices for the non-zero weight values.
    :param col_indices: List of column indices for the non-zero weight values.
    :param nz_values: List of non-zero values at the corresponding row/col indices. Each value may be either a 2D block or a scalar.
    :param slot_values: Non-zero values for the sparse slots which share row/col coordinates with
                        the weights. Each value may be either a 2D block or a scalar. (Optional - can be None)
    :param k: The number of indices to remove from the sparse set.
    :param debug_name: Name to uniquely identify the logging output for this call.
    """

    def bottom_k(a, k):
        return np.argpartition(a, k)[0:k]

    triplets = (row_indices, col_indices, nz_values)

    # Find the indices of the lowest magnitude weights:
    weight_magnitude = np.abs(triplets[2])
    # In case of blocks, sum over all block values
    if len(weight_magnitude.shape) > 1:
        weight_magnitude = weight_magnitude.sum(-1).sum(-1)
        value_type = "block"
    else:
        value_type = "element"

    lowest_weight_idx = bottom_k(weight_magnitude, k)

    logger.info(f"Pruning {k} / {weight_magnitude.size} non-zero {value_type}s for layer {debug_name}.")
    logger.debug(f"Mean ABS {value_type} weight for layer {debug_name}: {np.mean(weight_magnitude)}")
    logger.debug(f"Median ABS {value_type} weight magnitude for {debug_name}: {np.median(weight_magnitude)}")
    logger.debug(f"Bottom k {value_type} ABS weight range for {debug_name}: "
                 f"{np.min(weight_magnitude[lowest_weight_idx])} to {np.max(weight_magnitude[lowest_weight_idx])}")
    prune_row_indices = [triplets[0][i] for i in lowest_weight_idx]
    prune_col_indices = [triplets[1][i] for i in lowest_weight_idx]
    weights_to_prune = np.array([triplets[2][i] for i in lowest_weight_idx])
    if weights_to_prune.size > 0:
        mean_abs_pruned_weight = np.mean(np.abs(weights_to_prune))
        logger.debug(f"Mean ABS pruned element weight for {debug_name}: {mean_abs_pruned_weight}")

    if len(prune_row_indices) != k:
        raise RuntimeError(f"Pruned {len(prune_row_indices)} indices but expected to prune {k}")

    # Make new triplets with these indices removed:
    remaining_weight_triplets = [np.delete(t, lowest_weight_idx, axis=0) for t in triplets]

    # Prune the same indices from the slot triplets:
    slot_values = slot_values or {}
    remaining_slot_values = {
        name: np.delete(values, lowest_weight_idx, axis=0)
        for name, values in slot_values.items()
    }

    if len(remaining_weight_triplets[0]) + k != len(triplets[0]):
        raise RuntimeError(f"Remaining index count {len(remaining_weight_triplets[0])} is not the correct size: {len(triplets[0]) - k}")
    return remaining_weight_triplets, remaining_slot_values


def block_pool(dense_tensor, block_size, pooling_type="SUM"):
    # Reshape the grad into block of values
    nx, ny = dense_tensor.shape[0]//block_size, dense_tensor.shape[1]//block_size
    strides = np.array(dense_tensor.strides)  # strides are in bytes
    strides = tuple(strides*block_size) + tuple(strides)
    abs_dense_tensor = np.abs(dense_tensor)
    blocked_abs_dense_tensor = np.lib.stride_tricks.as_strided(abs_dense_tensor, (nx, ny, block_size, block_size), strides)
    # Sum the gradient over the block [nx,ny,b,b] -> [nx,ny]
    if pooling_type == "SUM":
        pooled_tensor = np.copy(blocked_abs_dense_tensor).sum(-1).sum(-1)
    elif pooling_type == "MAX":
        pooled_tensor = np.copy(blocked_abs_dense_tensor).max(-1).max(-1)
    elif pooling_type == "AVG":
        pooled_tensor = np.copy(blocked_abs_dense_tensor).mean(-1).mean(-1)
    return pooled_tensor


def regrow_rigl(unpruned_triplets, dense_grad_w, new_value_gen, grow_count, pool, debug_name):
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
    # Get the block size
    weights_shape = np.array(unpruned_triplets[2]).shape
    block_size = 1 if len(weights_shape) == 1 else weights_shape[-1]

    # Perform sum pooling by block
    if block_size > 1 and pool:
        logger.debug(f"Pooling dense grad for {debug_name} on host.")
        dense_grad_w = block_pool(dense_grad_w, block_size, "SUM")

    logger.debug(f"Layer {debug_name} dense grad shape: {dense_grad_w.shape}")

    abs_dense_grad = np.abs(dense_grad_w)
    if block_size > 1 and logger.level <= logging.DEBUG:
        plot_and_log_matrix(debug_name + "/abs_pooled_dense_gradw", abs_dense_grad)

    # Get flat indices for the original index set:
    original_flat_idx = np.ravel_multi_index((unpruned_triplets[0], unpruned_triplets[1]), dense_grad_w.shape)

    # We want to grow back weights at the positions with the highest gradient
    # magnitudes that are also not in the original set:
    abs_grad_flat = abs_dense_grad.flatten()
    abs_grad_flat[original_flat_idx] = -1  # This has the effect of excluding existing indices from the top-k
    argsorted = np.argsort(-abs_grad_flat)
    topk_flat_idx = argsorted[0:grow_count]
    common = np.intersect1d(topk_flat_idx, original_flat_idx)
    if len(common):
        logger.info(f"Number of intersecting indices: {common.size}")
        logger.debug(f"Intersection with original indices: {common}")
        raise RuntimeError("Intersection of new and original indices must be empty.")

    logger.info(f"Number of final non-intersecting indices: {topk_flat_idx.size}")

    # Check the indices are unique before we use them:
    unique = np.unique(topk_flat_idx)
    duplicates = len(topk_flat_idx) - len(unique)
    if duplicates != 0:
        print(f"New indices contain {duplicates} duplicates:\n{topk_flat_idx}")
        raise RuntimeError("New indices are not unique")

    top_k_idx = np.unravel_index(topk_flat_idx, dense_grad_w.shape)
    logger.debug(f"Layer {debug_name} Median ABS grad value: {np.median(abs_grad_flat)}")
    logger.debug(f"Layer {debug_name} Range of ABS grad top-k values: "
                 f"{np.min(abs_grad_flat[topk_flat_idx])} to {np.max(abs_grad_flat[topk_flat_idx])}")
    return (top_k_idx[0], top_k_idx[1], new_value_gen(size=grow_count, block_size=block_size))


def zero_values_generator(size=1, block_size=1):
    """
    Return a list of zeros of the specified length.
    :param size: Number of zeros to produce in one call.
    :param block_size: The blocks_size of each produced value. If set to 1 (default), returned value
           is a list of scalars otherwise a list with shape [size, block_size, block_size]
    """
    logger.debug(f"Generating {size} zero values of block-size: {block_size}")
    if block_size == 1:
        return np.array([0]*size)
    else:
        return np.zeros([size, block_size, block_size])


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
    block_size = 1 if len(grown_values.shape) == 1 else grown_values.shape[-1]
    kept_slot_values = kept_slot_values or {}
    grown_slots = {
        name: np.concatenate([
            values,
            zero_values_generator(size=pruned_count, block_size=block_size)
        ])
        for name, values in kept_slot_values.items()
    }
    return (grown_rows, grown_cols, grown_values), grown_slots


def prune_and_grow(name, triplets, shape,
                   spec, max_non_zeros,
                   slot_triplets, prune_schedule,
                   prune_ratio: float, grad_w: np.array,
                   grow_method='rigl', random_gen=None,
                   ipu_pooling_type='NONE'):
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
    if isinstance(grad_w, dict):
        grad_w = [grad for grad in grad_w.values()][0]

    # Compute the prune count using the provides schedule, the max number of zeros in the
    # layer and the pruning ratio
    prune_count = prune_schedule(max_pruned=int(np.ceil(prune_ratio * max_non_zeros)))
    if prune_count == 0:
        logger.info("Nothing to prune according to prune schedule.")
        return None

    logger.info(f"Triplet stats before prune and grow for {name}: {sparse.triplet_stats(*triplets)}")

    if logger.level <= logging.DEBUG:
        abs_nz_values = np.abs(triplets[2])
        if len(abs_nz_values.shape) > 1:
            abs_nz_values = abs_nz_values.sum(-1).sum(-1)
            block_input_size = spec.input_size // spec.block_size
            block_output_size = spec.output_size // spec.block_size
            block_spec = sparse.MatmulSpec(
                block_size=1, input_size=block_input_size, output_size=block_output_size,
                num_groups=spec.num_groups, batch_size=spec.batch_size, data_type=spec.data_type,
                max_non_zero_blocks=spec.max_non_zero_blocks, pooling_type=spec.pooling_type)
            dense_abs_weights = sparse.dense_from_triplets(
                block_spec, triplets[0], triplets[1], abs_nz_values)
            plot_and_log_matrix(name + "/abs_block_weights", dense_abs_weights)

    # Prune bottom k weights
    logging.debug(f"Pruning and grow also applies to these slot vars: {slot_triplets.keys()}")
    slot_values = {name: triplet.values for name, triplet in slot_triplets.items()}

    remaining_triplets, remaining_slot_values = prune_bottom_k_weights(
        *triplets, slot_values, prune_count, name)

    # regrow weights
    logger.debug(f"Regrowing non-zeros for layer {name} using '{grow_method}' method.")
    if grow_method == 'rigl':
        weights_shape = np.array(triplets[2]).shape
        block_size = 1 if len(weights_shape) == 1 else weights_shape[-1]
        # Grow back new indices using Rig-L: (https://arxiv.org/abs/1911.11134)
        if (shape != grad_w.shape and (ipu_pooling_type == "NONE" or block_size == 1)):
            raise RuntimeError(f"Dense weight gradient has unexpected shape.Expected {shape}, got {grad_w.shape}")
        new_triplets = regrow_rigl(
            triplets, grad_w, zero_values_generator, prune_count, ipu_pooling_type == "NONE", name)
    if grow_method == 'random':
        # Random replacement strategy: add back random indices
        # Gen some replacement random indices excluding all the existing
        # ones then we will swap for the pruned ones:
        new_triplets = sparse.random_triplets(spec, indices_initialiser_gen=random_gen,
                                              value_generator=zero_values_generator,
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
    grown_slots = {name: sparse.Triplets(grown_triplets[0], grown_triplets[1], grown_slot)
                   for name, grown_slot in grown_slots.items()}

    logger.info(f"Triplet stats after prune and grow for {name}: {sparse.triplet_stats(*grown_triplets)}")

    return {'gt': grown_triplets, 'nt': new_triplets, 'rt': remaining_triplets, 'gs': grown_slots, 'name': name}


def cosine_prune_function(t, T, opts):
    """
    Utility to build a parematerised cosine pruning schedule. Returns the
    prune ratio for the specified step.
    :param t: Current step
    :param T: Total steps
    :opts: Dictionary of options that describe the cosine function:
        zero_steps: Retunr 0 for this many steps before starting the cosine schedule.
        phase_delay: Phase offset for the cosine function.
        period: Period of the cosine function. Typically > 0 and < 0.5 unless you want a cyclic schedule.
    """
    zero_steps = opts['zero_steps']
    delay = opts['phase_delay']
    period = opts['period']
    logger.debug(f"zero-steps: {zero_steps}, t: {t}/{T}")
    if t < zero_steps:
        return 0
    current = t - zero_steps
    end = T - zero_steps
    return 0.5 * (1 + np.cos(delay + period * (current * (2 * np.pi / end))))
