# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import glob
import struct
import random
import argparse
import numpy as np
from scipy import optimize
from itertools import repeat, chain
from functools import lru_cache, reduce
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from .pretraining_dataset import CachedDataLoader, packed_data_file_format


@lru_cache(maxsize=None)
def get_packing_strategies(start_length, minimum_increment, target_length, depth):
    """Recursively build a list of unique packing "strategies".

    These strategies represent the ways that up to "depth" many sequences can
    be packed together to produce a packed sequence of exactly "target_length"
    tokens in total. For example [1, 2, 509] represent the packing strategy of
    combining a sequence of length 1, a sequence of length 2, and sequence of
    length 509 to create one packed sequence of length 512.

    To ensure strategies are unique, each sequence added to the pack must be
    at least as long as the previous sequence added to the pack. This is
    tracked through the "minimum_increment" variable and results in the
    strategies containing sequence lengths in sorted order e.g. [1, 2, 509]
    but not [2, 1, 509]

    Parameters
    ----------
    start_length : int
      The current cumulative number of tokens in the pack.
      Typically initalized to 0.
    minimum_increment : int
      The minimum length of the next sequence can be added to the pack.
      Typically initialized to 1.
    target_length : int
      The target_length for a pack of sequences (e.g. 512).
    depth : int
      Remaining depth in the recursion (must be > 0).

    Returns
    -------
    strategies : list[list[int]]
      A list of strategies where each strategy is a list of integers
      representing sequence lengths of the components in the pack. Each
      strategy should have at most "depth" entries and sum up to "target_length".
    """
    gap = target_length - start_length
    strategies = []

    # Complete the packing with exactly 1 number
    if depth == 1:
        if gap >= minimum_increment:
            strategies.append([gap])

    # Complete the sample in "depth" steps, recursively
    else:
        for new in range(minimum_increment, gap + 1):
            new_gap = target_length - start_length - new
            if new_gap == 0:
                strategies.append([new])
            else:
                options = get_packing_strategies(start_length + new, new, target_length, depth - 1)
                for option in options:
                    if len(option) > 0:
                        strategies.append([new] + option)
    return strategies


def get_packing_matrix(strategy_set, max_sequence_length):
    """Construct a packing matrix from a set of packing strategies.

    The packing matrix "A" is of shape [max_sequence_length, len(strategy_set)].
    Each column of the matrix corresponds to a strategy and each row corrsponds
    to the usage of a particular sequence length across all strategies.

    This matrix is typically very sparse. For instance for packing depth 3,
    each strategy uses at most 3 sequences leading to 3 non-zero entires in
    that strategy's column in A. The density of the matrix is then only
    3/max_sequence_length. This sparsity can be exploited to further speed-up the
    packing algorithm.

    Parameters
    ----------
    strategy_set : list[list[int]]
        A list of unique strategies as returned by get_packing_strategies.
    max_sequence_length : int
        The target or maximum sequence length of the packing problem.

    Returns
    -------
    A : np.array of shape [max_sequence_length, len(strategy_set)]
        The packing matrix for the provided strategy set.
    """
    num_strategies = len(strategy_set)
    A = np.zeros((max_sequence_length, num_strategies), dtype=np.int32)
    for i, strategy in enumerate(strategy_set):
        for seq_len in strategy:
            A[seq_len - 1, i] += 1
    return A


def get_packing_recipe(args, sequence_lengths):
    """Given program arguments and a list of sequence lengths return the packing recipe.

    A "packing recipe" primarily consists of a set of strategies "strategy_set" and the "mixture"
    which states how many times each one of these strategies should be applied in order to pack
    the dataset. Additionally, we also return the "padding" vector which states how many sequences
    of a given sequence length need to be added to our dataset in order use the proposed mixture
    of strategies.

    Parameters
    ----------
    args : namedtuple containing the following attributes
        max_sequence_length : int
            The maximum sequence length to which the sequences will be packed. Used to generate the
            appropriate packing strategies.
        max_sequences_per_pack : int
            The maximum number of sequences that can ever be put into a pack. Used to generate the
            appropriate packing strategies.
        drop_unpacked_remainder : bool
            Whether to drop the sequences that could not be packed (usually a very small percentage)
            If false, then the unpacked sequences will be padded instead.
    sequence_lengths : list[int]
        A list containing the sequence length of each example in the un-packed dataset.
    Returns
    -------
    strategy_set : list[list[int]]
        The list of unique packing strategies with which the packing problem
        was solved.
    mixture : list[int] pf shape [len(strategy_set)]
        States how many times each of the strategies from the strategy set
        should be repeated to cover the entire dataset.
    padding : list[int] pf shape [max_sequence_length]
        For each sequence length how many padding sequence of that length
        need to be created to realize the packing mixture.
    """

    print("Entering packing solver".center(80, "_"))

    # List all unique ways of packing to the desired maximum sequence length
    strategy_set = get_packing_strategies(0, 1, args.max_sequence_length, args.max_sequences_per_pack)
    for strategy in strategy_set:
        assert(sum(strategy) == args.max_sequence_length)
    num_strategies = len(strategy_set)
    print(f"Packing will involve {num_strategies} unique packing strategies.",
          f"at a maximum {args.max_sequences_per_pack} sequences per pack.")

    # Get the packing matrix corresponding to this list of packing strategies
    A = get_packing_matrix(strategy_set, args.max_sequence_length)

    # To achieve more robust convergence of the packing problem we create
    # weights that penalize the residual on short sequences less.
    # In other words we allow short sequences (up to length padding_cutoff)
    # to be over-used to a larger degree than longer sequences
    padding_cutoff = 8
    w0 = np.ones([args.max_sequence_length])
    w0[:padding_cutoff] = padding_cutoff / (2 * args.max_sequence_length)
    w0 = np.sqrt(w0)

    # Histogram of sequence lengths
    histogram, bins = np.histogram(sequence_lengths, bins=np.arange(1, args.max_sequence_length + 2))

    # Solve the packing problem
    # A@mixture = histogram
    # i.e. find the non-negative "mixture" of strategies such that the
    # packing matches the distribution of sequences lengths (histogram) as
    # closely as possbile in the least squares sense
    print(f"Sequences to pack: ", histogram.sum())
    start = time.time()
    mixture, rnorm = optimize.nnls(np.expand_dims(w0, -1) * A, w0 * histogram)
    print(f"Solving non-negative least squares took {time.time() - start:3.2f} seconds.")

    # Round the floating point solution to integer).
    # The relative error introduced by this is relatively small since we are
    # dealing with millions of sequences while rounding introduces a residual
    # of around ~ max_sequence_length sequences.
    residual_float = histogram - A @ mixture
    mixture = np.rint(mixture)

    # Compute the residuals
    residual = histogram - A @ mixture
    rounding_residual = abs(residual_float - residual).sum()
    print(f"Total residual of packing mixture: {abs(residual).sum():3.1f}",
          f"Total residual introduced by rounding mixture to int: {rounding_residual:3.2f}",
          f"Residual on first 8 categories: {np.around(residual[:8], 4)}",
          f"Residual on last 8 categories:  {np.around(residual[-8:], 4)}", sep="\n")

    # Optionally add additional mixture entries for leftover sequences
    if not args.drop_unpacked_remainder:
        unpacked = np.where(residual > 0, residual, 0)
        assert unpacked[-1] == 0, "Max sequence length should always be fully used up."
        unpacked_seqlen = np.arange(1, args.max_sequence_length + 1)[unpacked > 0]
        # Update the mixture to also covered the unpacked sequences
        for l in unpacked_seqlen:
            # Get the depth 1 strategy
            strategy = sorted([l, args.max_sequence_length - l])
            strategy_index = strategy_set.index(strategy)
            mixture[strategy_index] += unpacked[l-1]

        # Recompute the residual for the mixture (should be < 0)
        residual = histogram - A @ mixture

    # Add padding based on deficit (negative residual)
    padding = np.where(residual < 0, -residual, 0)

    # End of solver, now printing out some properties of the packing mixture
    samples_dropped = (residual + padding).sum().astype(np.int32)
    new_number_of_samples = int(mixture.sum())
    compression = 1 - new_number_of_samples / (len(sequence_lengths) - samples_dropped)
    num_padding_tokens_original = (args.max_sequence_length - sequence_lengths).sum()
    num_padding_tokens_packed = (np.arange(1, args.max_sequence_length + 1) * padding).sum()
    speedup_upper_bound = 1.0 / (1 - ((1 - sequence_lengths / args.max_sequence_length).mean()))
    avg_sequences_per_sample = ((A.sum(0) * mixture).sum() - padding.sum()) / new_number_of_samples
    efficiency = 1 - num_padding_tokens_packed/(new_number_of_samples*args.max_sequence_length)
    print(f"Done solving for packing mixture".center(80, "_"),
          f"Packing efficiency (fraction of real tokens): {efficiency:3.4f}",
          f"Added {num_padding_tokens_packed:3.2e} padding tokens. Original dataset used {num_padding_tokens_original:3.2e} padding tokens",
          f"New number of samples: {new_number_of_samples}, original {len(sequence_lengths)}. A compression ratio of {compression:3.3f}",
          f"Average sequences/sample {avg_sequences_per_sample:3.5f}",
          f"Theoretical upper bound on speed-up: {speedup_upper_bound:3.3f}",
          f"The achieved speed-up from packing: {1/(1-compression):3.3f}",
          f"Number of strategies utilized: {np.count_nonzero(mixture)}",
          f"Number of sequences dropped:  {samples_dropped}",
          f"Top 8 strategies:", sep="\n")
    for i in np.argsort(-mixture)[:8]:
        print(f"\tstrategy {strategy_set[i]} which is used {int(mixture[i])} times")
    print("".center(80, "_"))

    mixture = mixture.astype(np.int64)
    padding = padding.astype(np.int64)
    return strategy_set, mixture, padding


def get_example_slicing(strategy_set, mixture, max_sequence_length):
    """
    Slice the bins of examples such that the strategies can be filled
    and written to disk in parallel. In essence this calculates of a
    cumulative sum of how many sequences of a given length are used by
    strategies that occur earlier in the strategy_set

    Returns
    -------
    slicing : np.array of shape [num_strategies, max_sequence_length]
        Provides the starting index for slicing a particular sequence length
        to fill a given strategy.
    """
    A = get_packing_matrix(strategy_set, max_sequence_length)

    # The first strategy should start its slice at 0, and all
    # other strategies should start at the cumulative sum of
    # sequences used up strategies which come earlier in the
    # strategy set
    slicing = np.zeros_like(A)
    slicing[:, 1:] = np.cumsum(A * mixture, axis=1)[:, :-1]
    slicing = slicing.T  # transposing [max_sequence_length, num_strategies]
    return slicing


def slice_examples(examples_by_length, slicing, strategy_set, mixture):
    """Divide the examples between strategies to enable parallel processing.

    Parameters
    ----------
    examples_by_length : dict
        A dictionary mapping from sequence_length to the bin of examples of that
        sequence length.
    slicing : np.array
        The slicing information obtained using get_example_slicing
    strategy_set : list[list[int]]
        The list of unique packing strategies with which the packing problem
        was solved.
    mixture : list[int] pf shape [len(strategy_set)]
        States how many times each of the strategies from the strategy set
        should be repeated to cover the entire dataset.

    Returns
    -------
    example_slices : list[multi_sequence]
        Each component of the list is a list of multiple sequences i.e. specifically sequences which
        are to be combined together to form a pack according to a strategy.
    strategies : list[int]
        A list containing the strategy for each slice i.e. strategy will appear multiple times if it the
        work to fill the strategy has been split into multiple parts. The repetition of strategies is
        the main difference with strategy_set, which is unique.
    part_idx : int
        Used to ensure uniqueness of output filenames i.e. if a strategy contains many examples
        the work is split into multiple parts and processed in parallel. Each process receives a
        different part_idx such that they write to different files.
    """

    chunk_limit = 50000
    example_slices = []
    strategies = []
    part_idx = []
    for strategy, slice_offsets, repeat_count in zip(strategy_set, slicing, mixture):
        if repeat_count == 0:
            continue
        # Slice out the sequences allocated to this strategy in increments of 50k
        num_parts = repeat_count // chunk_limit
        num_parts = num_parts + int(repeat_count != num_parts * chunk_limit)
        subcounts = (min(chunk_limit, repeat_count - chunk_limit * (i - 1)) for i in range(1, num_parts + 1))
        for part_id, part_count in enumerate(subcounts):
            examples = []
            for k, seq_len in enumerate(strategy):
                slice_start = int(slice_offsets[seq_len - 1])
                slice_end = slice_start + int(part_count)
                slice_offsets[seq_len - 1] = slice_end
                examples.append(examples_by_length[seq_len][slice_start:slice_end])

            example_slices.append(examples)
            strategies.append(strategy)
            part_idx.append(part_id)

    return example_slices, strategies, part_idx


def parallel_pack_according_to_strategy(args, part_idx, strategy, example_slices):
    """Pack "examples" according to "strategy" and write them to disk

    Parameters
    ----------
    args : namedtuple containing the following attributes
        output_dir : str
            The destination folder to which examples will be written. This
            directory should already exist.
        mlm_tokens : int
            The maximum number of masked lm predictions in each unpacked sequence.
        max_sequence_length : int
            The maximum sequence length to which the sequences will be packed. Used to generate the
            appropriate packing strategies.
        max_sequences_per_pack : int
            The maximum number of sequences that can ever be put into a pack. Used to generate the
            appropriate packing strategies.
    part_idx : int
        Used to ensure uniqueness of output filenames i.e. if a strategy contains many examples
        the work is split into multiple parts and processed in parallel. Each process receives a
        different part_idx such that they write to different files.
    strategy : list[int]
        A strategy is a list of integers representing the sequence lengths of the components
        in the pack.
    example_slices : list[multi_sequence]
        Each component of the list is a list of multiple sequences i.e. specifically sequences which
        are to be combined together to form a pack according to the provided strategy.

    Returns
    -------
    This function does not return anything, instead it writes its output directly to a file. This
    facilities packing different strategies in parallel.
    """
    base_filename = os.path.join(args.output_dir, "strategy_" + "_".join(map(str, strategy)))
    filename = base_filename + f"_part_{part_idx}"
    lines = []
    for i, multi_sequence in enumerate(zip(*example_slices)):
        lines.append(create_multi_sequence_example(multi_sequence, args.mlm_tokens,
                                                   args.max_sequence_length, args.max_sequences_per_pack))
    # Write to file
    with open(filename, "wb") as f:
        f.writelines(lines)


def create_multi_sequence_example(multi_sequence, mlm_tokens, max_sequence_length, max_sequences_per_pack):
    """Combines a list of sequences into a single pack

    Parameters
    ----------
    multi_sequence : list of sequences, each sequence is a tuple of numpy arrays
        The incoming sequences are also in the "packed" pretraining data format,
        contain just 1 sequence per pack. Entries which are "None" represent
        padding sequences.
    mlm_tokens : int
        The maximum number of masked lm predictions in each unpacked sequence.
    max_sequence_length : int
        The sequence length to which sequence in the multi_sequence will be packed.
    max_sequence_per_pack : int
        This value must be the same for all packs in the dataset as it determines the
        data format. Represents the maximum number of sequences that a pack may contain.

    Returns
    -------
    line : binary str
        The binary representation of the packed sequences, ready to be written to a file.
    """
    # SEQ
    packed_input_ids = np.zeros(max_sequence_length, dtype=np.int32)
    packed_input_mask = np.zeros(max_sequence_length, dtype=np.int32)
    packed_segment_ids = np.zeros(max_sequence_length, dtype=np.int32)
    packed_positions = np.zeros(max_sequence_length, dtype=np.int32)

    # MLM
    # It is assumed that each sequence in a pack has a (roughly) fixed percentage of MLM tokens.
    # To account for the cases where this percentage is rounded up, additional mlm tokens are added
    # e.g in a pack of sequence lengths [ 17  37 458], 15% represents [2.55 5.55 68.7] which is
    # then rounded to  [3 6 69] or 78 tokens in total. This is equivalent to
    # (mlm_tokens + max_sequences_per_pack - 1 = 76 + 3 - 1).
    # The "-1" term is omitted in implementation to provide a small margin for other effects.
    packed_masked_lm_positions = np.zeros(mlm_tokens + max_sequences_per_pack, dtype=np.int32)
    packed_masked_lm_ids = np.zeros(mlm_tokens + max_sequences_per_pack, dtype=np.int32)
    packed_masked_lm_weights = np.zeros(mlm_tokens + max_sequences_per_pack, dtype=np.int32)

    # NSP
    packed_next_sentence_positions = np.zeros(max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_labels = np.zeros(max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_weights = np.zeros(max_sequences_per_pack, dtype=np.int32)

    offset = 0
    mlm_offset = 0
    sequence_index = 1  # used in the input mask
    for sequence in multi_sequence:
        # Padding sequences are denoted with None
        if sequence is not None:
            input_ids, input_mask, segment_ids, positions, *sequence = sequence
            masked_lm_positions, masked_lm_ids, masked_lm_weights, *sequence = sequence
            next_sentence_positions, next_sentence_labels, next_sentence_weights = sequence
            seq_len = input_mask.sum()

            # SEQ
            packed_input_ids[offset:offset + seq_len] = input_ids[:seq_len]
            packed_input_mask[offset:offset + seq_len] = sequence_index
            packed_segment_ids[offset:offset + seq_len] = segment_ids[:seq_len]
            packed_positions[offset:offset + seq_len] = np.arange(0, seq_len)

            # MLM
            mlm_len = int(masked_lm_weights.sum())
            assert mlm_offset + mlm_len < mlm_tokens + max_sequences_per_pack, "Too many LM predictions per sequences"
            max_mlm = mlm_offset + mlm_len
            packed_masked_lm_positions[mlm_offset:max_mlm] = offset + masked_lm_positions[:mlm_len]
            packed_masked_lm_ids[mlm_offset:max_mlm] = masked_lm_ids[:mlm_len]
            packed_masked_lm_weights[mlm_offset:max_mlm] = sequence_index

            # NSP
            packed_next_sentence_positions[sequence_index - 1] = offset
            packed_next_sentence_labels[sequence_index - 1] = next_sentence_labels
            packed_next_sentence_weights[sequence_index - 1] = 1

            # Update offsets
            sequence_index += 1
            offset += seq_len
            mlm_offset = max_mlm

    # Pack into binary format and write it
    line = reduce(lambda accl, i: accl + struct.pack('<I', i),
                  chain(packed_input_ids,
                        packed_input_mask,
                        packed_segment_ids,
                        packed_positions,
                        packed_masked_lm_positions,
                        packed_masked_lm_ids,
                        packed_masked_lm_weights,
                        packed_next_sentence_positions,
                        packed_next_sentence_labels,
                        packed_next_sentence_weights), b'')
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", help="A glob expression for the input files to read in and pack", required=True, type=str)
    parser.add_argument("--output-dir", help="The destination folder for the output files", required=True)
    parser.add_argument("--random-seed", help="For shuffling the data", default=12345)
    parser.add_argument("--drop-unpacked-remainder", help="Whether to drop sequences that failed to pack", default=True, type=eval)
    parser.add_argument("--unpacked-dataset-duplication-factor",
                        help="The duplication factor with which create_pretraining_data.py was run to create the un-packed dataset. " +
                        "The output dataset will always have duplication factor 1", default=1, type=int)
    parser.add_argument("--max-sequence-length", help="The maximum number of tokens in an example", default=512, type=int)
    parser.add_argument("--mlm-tokens", help="The maximum number of masked tokens in an un-packed example", default=76, type=int)
    parser.add_argument("--max-sequences-per-pack", help="The maximum number of sequences per packed example.", choices=[2, 3], default=3, type=int)
    args = parser.parse_args()

    # Input files
    input_files = glob.glob(args.input_glob)
    assert len(input_files) > 0

    # Load un-packed dataset (1 sequence per pack)
    load_batch_size = 1024
    sample_sizes = packed_data_file_format(args.max_sequence_length, args.mlm_tokens, 1)
    dataset = CachedDataLoader(input_files, sample_sizes, duplication_factor=args.unpacked_dataset_duplication_factor, batch_size=load_batch_size)

    # Put examples into bins depending on their sequence lengths and extract the sequence length
    # as an array
    sequence_lengths = []
    examples_by_length = defaultdict(list)
    print("Looping through dataset to collect sequence length information...")
    for data in dataset:
        input_mask = data[1]
        batch_of_lengths = input_mask.sum(1).tolist()
        for i, length in enumerate(batch_of_lengths):
            examples_by_length[length].append([d[i] for d in data])
        sequence_lengths.extend(batch_of_lengths)
    sequence_lengths = np.array(sequence_lengths)
    assert len(sequence_lengths) > 1

    # Run the packing algorithm on these sequence lengths
    strategy_set, mixture, padding = get_packing_recipe(args, sequence_lengths)

    # Add the required number of padding (i.e. empty) sequences
    # The padding is added explicitly to enable parallel processing
    # while make sure that the probability of a padding sequence occuring
    # is equal across all strategies
    for i in range(1, args.max_sequence_length + 1):
        examples_by_length[i].extend([None] * int(padding[i - 1]))
    # Shuffle the data to fairly redistribute the padding among all
    # strategies that will slice the data
    random.seed(args.random_seed)
    for key in examples_by_length:
        random.shuffle(examples_by_length[key])

    print(f"\nPacking and writing packed dataset to {args.output_dir}.")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Slice the data into chunks of max 50k packed examples
    slicing = get_example_slicing(strategy_set, mixture, args.max_sequence_length)
    example_slices, strategies, part_idx = slice_examples(examples_by_length, slicing, strategy_set, mixture)
    print(f"Splitting work into {len(part_idx)} parts.")

    # Parallel packing and write of examples to disk
    start = time.time()
    with ProcessPoolExecutor() as executor:
        work = repeat(args), part_idx, strategies, example_slices
        for partial_result in executor.map(parallel_pack_according_to_strategy, *work):
            pass
    print(f"\nDone. Took: {time.time() - start:3.2f} seconds to pack and write dataset.")
