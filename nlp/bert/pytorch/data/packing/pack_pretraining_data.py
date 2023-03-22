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

import collections
import os
import time
import glob
import random
import argparse
import numpy as np
import psutil
from scipy import optimize
from itertools import chain, islice, repeat
from functools import lru_cache, partial
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required to generate data for this application. " "Please install with: 'pip install tensorflow'"
    )
from pretraining_data import TFRecordPretrainingDataset, _WorkerInit
from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
from transformers import BertConfig
from ipu_options import get_options
from args import parse_bert_args
from tqdm import tqdm
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


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
    start_length:int
      The current cumulative number of tokens in the pack.
      Typically initialized to 0.
    minimum_increment:int
      The minimum length of the next sequence can be added to the pack.
      Typically initialized to 1.
    target_length:int
      The target_length for a pack of sequences (e.g. 512).
    depth:int
      Remaining depth in the recursion (must be > 0).

    Returns
    -------
    strategies:list[list[int]]
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


def get_packing_matrix(strategy_set, sequence_length):
    """Construct a packing matrix from a set of packing strategies.

    The packing matrix "A" is of shape [sequence_length, len(strategy_set)].
    Each column of the matrix corresponds to a strategy and each row corresponds
    to the usage of a particular sequence length across all strategies.

    This matrix is typically very sparse. For instance for packing depth 3,
    each strategy uses at most 3 sequences leading to 3 non-zero entries in
    that strategy's column in A. The density of the matrix is then only
    3/sequence_length. This sparsity can be exploited to further speed-up the
    packing algorithm.

    Parameters
    ----------
    strategy_set:list[list[int]]
        A list of unique strategies as returned by get_packing_strategies.
    sequence_length:int
        The target or maximum sequence length of the packing problem.

    Returns
    -------
    A:np.array of shape [sequence_length, len(strategy_set)]
        The packing matrix for the provided strategy set.
    """
    num_strategies = len(strategy_set)
    A = np.zeros((sequence_length, num_strategies), dtype=np.int32)
    for i, strategy in enumerate(strategy_set):
        for seq_len in strategy:
            A[seq_len - 1, i] += 1
    return A


def get_packing_recipe(args, sequence_lengths, drop_unused_strategies=False):
    """Given program arguments and a list of sequence lengths return the packing recipe.

    A "packing recipe" primarily consists of a set of strategies "strategy_set" and the "mixture"
    which states how many times each one of these strategies should be applied in order to pack
    the dataset. Additionally, we also return the "padding" vector which states how many sequences
    of a given sequence length need to be added to our dataset in order use the proposed mixture
    of strategies.

    Parameters
    ----------
    args:namedtuple containing the following attributes
        sequence_length:int
            The maximum sequence length to which the sequences will be packed. Used to generate the
            appropriate packing strategies.
        max_sequences_per_pack:int
            The maximum number of sequences that can ever be put into a pack. Used to generate the
            appropriate packing strategies.
        drop_unpacked_remainder:bool
            Whether to drop the sequences that could not be packed (usually a very small percentage)
            If false, then the unpacked sequences will be padded instead.
    sequence_lengths:list[int]
        A list containing the sequence length of each example in the un-packed dataset.
    drop_unused_strategies:bool
        If True, filter out strategies that are to be used 0 times according to the mixture.
    Returns
    -------
    strategy_set:list[list[int]]
        The list of unique packing strategies with which the packing problem
        was solved.
    mixture:list[int] of shape [len(strategy_set)]
        States how many times each of the strategies from the strategy set
        should be repeated to cover the entire dataset.
    padding:list[int] of shape [sequence_length]
        For each sequence length how many padding sequence of that length
        need to be created to realize the packing mixture.
    """

    print("Entering packing solver".center(80, "_"))

    # List all unique ways of packing to the desired maximum sequence length
    strategy_set = get_packing_strategies(0, 1, args.sequence_length, args.max_sequences_per_pack)
    for strategy in strategy_set:
        assert sum(strategy) == args.sequence_length
    num_strategies = len(strategy_set)
    print(
        f"Packing will involve {num_strategies} unique packing strategies.",
        f"at a maximum {args.max_sequences_per_pack} sequences per pack.",
    )

    # Get the packing matrix corresponding to this list of packing strategies
    A = get_packing_matrix(strategy_set, args.sequence_length)

    # To achieve more robust convergence of the packing problem we create
    # weights that penalize the residual on short sequences less.
    # In other words we allow short sequences (up to length padding_cutoff)
    # to be over-used to a larger degree than longer sequences
    padding_cutoff = 8
    w0 = np.ones([args.sequence_length])
    w0[:padding_cutoff] = padding_cutoff / (2 * args.sequence_length)
    w0 = np.sqrt(w0)

    # Histogram of sequence lengths
    histogram, bins = np.histogram(sequence_lengths, bins=np.arange(1, args.sequence_length + 2))

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
    # of around ~ sequence_length sequences.
    residual_float = histogram - A @ mixture
    mixture = np.rint(mixture)

    # Compute the residuals
    residual = histogram - A @ mixture
    rounding_residual = abs(residual_float - residual).sum()
    print(
        f"Total residual of packing mixture: {abs(residual).sum():3.1f}",
        f"Total residual introduced by rounding mixture to int: {rounding_residual:3.2f}",
        f"Residual on first 8 categories: {np.around(residual[:8], 4)}",
        f"Residual on last 8 categories:  {np.around(residual[-8:], 4)}",
        sep="\n",
    )

    # Optionally add additional mixture entries for leftover sequences
    if not args.drop_unpacked_remainder:
        unpacked = np.where(residual > 0, residual, 0)
        assert unpacked[-1] == 0, "Max sequence length should always be fully used up."
        unpacked_seqlen = np.arange(1, args.sequence_length + 1)[unpacked > 0]
        # Update the mixture to also covered the unpacked sequences
        for l in unpacked_seqlen:
            # Get the depth 1 strategy
            strategy = sorted([l, args.sequence_length - l])
            strategy_index = strategy_set.index(strategy)
            mixture[strategy_index] += unpacked[l - 1]

        # Recompute the residual for the mixture (should be < 0)
        residual = histogram - A @ mixture

    # Add padding based on deficit (negative residual)
    padding = np.where(residual < 0, -residual, 0)

    # End of solver, now printing out some properties of the packing mixture
    samples_dropped = (residual + padding).sum().astype(np.int32)
    new_number_of_samples = int(mixture.sum())
    compression = 1 - new_number_of_samples / (len(sequence_lengths) - samples_dropped)
    num_padding_tokens_original = (args.sequence_length - sequence_lengths).sum()
    num_padding_tokens_packed = (np.arange(1, args.sequence_length + 1) * padding).sum()
    speedup_upper_bound = 1.0 / (1 - ((1 - sequence_lengths / args.sequence_length).mean()))
    avg_sequences_per_sample = ((A.sum(0) * mixture).sum() - padding.sum()) / new_number_of_samples
    efficiency = 1 - num_padding_tokens_packed / (new_number_of_samples * args.sequence_length)
    print(
        f"Done solving for packing mixture".center(80, "_"),
        f"Packing efficiency (fraction of real tokens): {efficiency:3.4f}",
        f"Added {num_padding_tokens_packed:3.2e} padding tokens. Original dataset used {num_padding_tokens_original:3.2e} padding tokens",
        f"New number of samples: {new_number_of_samples}, original {len(sequence_lengths)}. A compression ratio of {compression:3.3f}",
        f"Average sequences/sample {avg_sequences_per_sample:3.5f}",
        f"Theoretical upper bound on speed-up: {speedup_upper_bound:3.3f}",
        f"The achieved speed-up from packing: {1/(1-compression):3.3f}",
        f"Number of strategies utilized: {np.count_nonzero(mixture)}",
        f"Number of sequences dropped:  {samples_dropped}",
        f"Top 8 strategies:",
        sep="\n",
    )
    for i in np.argsort(-mixture)[:8]:
        print(f"\tstrategy {strategy_set[i]} which is used {int(mixture[i])} times")
    print("".center(80, "_"))

    mixture = mixture.astype(np.int64)
    padding = padding.astype(np.int64)
    if drop_unused_strategies:
        used_mixture_indices = np.nonzero(mixture)[0]
        strategy_set = [strategy_set[idx] for idx in used_mixture_indices]
        mixture = mixture[used_mixture_indices]
    return strategy_set, mixture, padding


def slice_examples(examples_by_length, strategy_set, mixture):
    """Divide the examples between strategies in order to (partially) fulfill the mixture

    Parameters
    ----------
    examples_by_length:dict
        A dictionary mapping from sequence_length to the bin of examples of that
        sequence length.
    strategy_set:list[list[int]]
        The list of unique packing strategies with which the packing problem
        was solved.
    mixture:list[int] pf shape [len(strategy_set)]
        States how many times each of the strategies from the strategy set
        should still be repeated to cover the entire dataset.

    Returns
    -------
    example_slices:list[multi_sequence]
        Each component of the list is a list of multiple sequences i.e. specifically sequences which
        are to be combined together to form a pack according to a strategy.
    strategies:list[int]
        A list containing the strategy for each slice i.e. strategy will appear multiple times if it the
        work to fill the strategy has been split into multiple parts. The repetition of strategies is
        the main difference with strategy_set, which is unique.
    new_mixture:list[int] pf shape [len(strategy_set)]
        States how many times each of the strategies from the strategy set
        should still be repeated to cover the entire dataset.
    """

    total_packs_to_be_written = 0
    example_slices = []
    strategies = []
    for i, (strategy, target_repeat_count) in enumerate(zip(strategy_set, mixture)):
        # Determine how much of the mixture can be fulfilled given the (partial)
        # examples by length
        feasible_repeat_count = target_repeat_count
        for k in set(strategy):
            feasible_repeat_count = min(feasible_repeat_count, len(examples_by_length[k]) // strategy.count(k))

        # IF nothing to do
        if feasible_repeat_count == 0:
            continue

        examples = []
        for k, seq_len in enumerate(strategy):
            examples.append(examples_by_length[seq_len][-feasible_repeat_count:])
            del examples_by_length[seq_len][-feasible_repeat_count:]
        example_slices.append(examples)
        strategies.append(strategy)
        total_packs_to_be_written += feasible_repeat_count
        mixture[i] -= feasible_repeat_count

    return example_slices, strategies, mixture, total_packs_to_be_written


def submit_example_slices_for_packing(args, executor, example_slices):
    """Submit example slices to a ProcessPoolExecutor for packing, and return associated futures.

    Parameters
    ----------
    args:namedtuple containing the following attributes
        mask_tokens:int
            The maximum number of masked lm predictions in each unpacked sequence.
        sequence_length:int
            The maximum sequence length to which the sequences will be packed. Used to generate the
            appropriate packing strategies.
        max_sequences_per_pack:int
            The maximum number of sequences that can ever be put into a pack. Used to generate the
            appropriate packing strategies.
        num_packing_workers:int
            The maximum number of worker subprocesses to be used for packing sequences.
        chunks_per_packing_worker:int
            Approximate number of chunks to divide the example slices due to be packed by each packing subprocess.
    executor:concurrent.futures.ProcessPoolExecutor
        A wrapper around the multiprocessing module enabling asynchronous execution to which we submit
        chunks of example_slices.
    example_slices:list[multi_sequence]
        Each component of the list is a list of multiple sequences i.e. specifically sequences which
        are to be combined together to form a pack.

    Returns
    -------
    futures:list[concurrent.futures.Future] of length len(example_slices)
        Each Future gives access to the results of the tasks submitted to the executor, as they become available.
    """
    # Shuffle to make the distribution of lengths of packs across packing workers more uniform.
    random.shuffle(example_slices)
    futures = []
    packs_per_worker = len(example_slices) // args.num_packing_workers + (
        len(example_slices) % args.num_packing_workers > 0
    )
    chunksize = max(1, packs_per_worker // args.chunks_per_packing_worker)
    work = example_slices, repeat(args.mask_tokens), repeat(args.sequence_length), repeat(args.max_sequences_per_pack)
    for chunk in get_chunks(zip(*work), chunksize=chunksize):
        futures.append(executor.submit(partial(process_chunk, create_multi_sequence_example), chunk))
    return futures


def get_chunks(iterable, chunksize):
    iterator = iter(iterable)
    while True:
        chunk = tuple(islice(iterator, chunksize))
        if not chunk:
            return
        yield chunk


def process_chunk(fn, chunk):
    return [fn(*args) for args in chunk]


def realise_futures(futures):
    return list(chain.from_iterable(f.result() for f in as_completed(futures)))


def create_multi_sequence_example(multi_sequence, mask_tokens, sequence_length, max_sequences_per_pack):
    """Combines a list of sequences into a single pack

    Parameters
    ----------
    multi_sequence:list of sequences, each sequence is a tuple of numpy arrays
        The incoming sequences are also in the "packed" pretraining data format,
        contain just 1 sequence per pack. Entries which are "None" represent
        padding sequences.
    mask_tokens:int
        The maximum number of masked lm predictions in each unpacked sequence.
    sequence_length:int
        The sequence length to which sequence in the multi_sequence will be packed.
    max_sequence_per_pack:int
        This value must be the same for all packs in the dataset as it determines the
        data format. Represents the maximum number of sequences that a pack may contain.

    Returns
    -------
    line:binary str
        The binary representation of the packed sequences, ready to be written to a file.
    """
    # SEQ
    packed_input_ids = np.zeros(sequence_length, dtype=np.int32)
    packed_input_mask = np.zeros(sequence_length, dtype=np.int32)
    packed_segment_ids = np.zeros(sequence_length, dtype=np.int32)
    packed_position_ids = np.zeros(sequence_length, dtype=np.int32)

    # MLM
    # It is assumed that each sequence in a pack has a (roughly) fixed percentage of MLM tokens.
    # To account for the cases where this percentage is rounded up, additional mlm tokens are added
    # e.g in a pack of sequence lengths [ 17  37 458], 15% represents [2.55 5.55 68.7] which is
    # then rounded to  [3 6 69] or 78 tokens in total. This is equivalent to
    # (mask_tokens + max_sequences_per_pack - 1 = 76 + 3 - 1).
    packed_masked_lm_positions = np.zeros(mask_tokens + max_sequences_per_pack - 1, dtype=np.int32)
    packed_masked_lm_ids = np.zeros(mask_tokens + max_sequences_per_pack - 1, dtype=np.int32)
    packed_masked_lm_mask = np.zeros(mask_tokens + max_sequences_per_pack - 1, dtype=np.float32)

    # NSP
    packed_next_sentence_labels = np.zeros(max_sequences_per_pack, dtype=np.int32)
    packed_next_sentence_mask = np.zeros(max_sequences_per_pack, dtype=np.float32)

    # An offset of where to start writing the tokens
    seq_offset = 0

    # Where to write the next MLM token
    mlm_offset = 0
    sequence_index = 1  # used in the input mask
    for sequence in multi_sequence:
        # Padding sequences are denoted with None
        if sequence is not None:
            input_ids, input_mask, segment_ids, *sequence = sequence
            masked_lm_positions, masked_lm_ids, next_sentence_labels = sequence
            # Use input_mask because input_ids could contain false "0"s
            seq_len = input_mask.sum()
            mask_tokens_mask_idx = (masked_lm_positions != 0).sum()

            # SEQ
            # This writes all the normal sequence tokens excluding the [CLS] tokens
            max_seq = seq_offset + seq_len - 1
            packed_input_ids[seq_offset:max_seq] = input_ids[1:seq_len]
            packed_input_mask[seq_offset:max_seq] = sequence_index
            packed_segment_ids[seq_offset:max_seq] = segment_ids[1:seq_len]
            packed_position_ids[seq_offset:max_seq] = np.arange(1, seq_len)

            # MLM
            max_mlm = mlm_offset + mask_tokens_mask_idx

            # Very rarely, (1/10M) it can occur that a multi-sequence has more than the allowed
            # number of cumulative mask_tokens, such multi-sequences are thrown away
            try:
                packed_masked_lm_positions[mlm_offset:max_mlm] = (
                    masked_lm_positions[:mask_tokens_mask_idx] - 1 + seq_offset
                )
            except ValueError:
                return b"", False

            packed_masked_lm_ids[mlm_offset:max_mlm] = masked_lm_ids[:mask_tokens_mask_idx]
            packed_masked_lm_mask[mlm_offset:max_mlm] = sequence_index

            # NSP
            # The NSP tokens are packed at the end of the sequence to enable slicing
            packed_input_ids[-sequence_index] = input_ids[0]
            packed_input_mask[-sequence_index] = sequence_index
            packed_segment_ids[-sequence_index] = segment_ids[0]
            packed_position_ids[-sequence_index] = 0
            packed_next_sentence_labels[-sequence_index] = next_sentence_labels
            packed_next_sentence_mask[-sequence_index] = 1

            # Update offsets
            sequence_index += 1
            seq_offset = max_seq
            mlm_offset = max_mlm

    # Pack into binary format and write it
    features = collections.OrderedDict()
    features["packed_input_ids"] = create_int_feature(packed_input_ids)
    features["packed_input_mask"] = create_int_feature(packed_input_mask)
    features["packed_segment_ids"] = create_int_feature(packed_segment_ids)
    features["packed_position_ids"] = create_int_feature(packed_position_ids)
    features["packed_masked_lm_positions"] = create_int_feature(packed_masked_lm_positions)
    features["packed_masked_lm_ids"] = create_int_feature(packed_masked_lm_ids)
    features["packed_masked_lm_mask"] = create_float_feature(packed_masked_lm_mask)
    features["packed_next_sentence_labels"] = create_int_feature(packed_next_sentence_labels)
    features["packed_next_sentence_mask"] = create_float_feature(packed_next_sentence_mask)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example.SerializeToString(), any([seq is not None for seq in multi_sequence])


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def get_dataloader(config, opts):
    dataset = TFRecordPretrainingDataset(config.input_files)
    loader = DataLoader(
        opts,
        dataset,
        batch_size=config.micro_batch_size,
        num_workers=config.dataloader_workers,
        drop_last=False,
        worker_init_fn=_WorkerInit(config.random_seed),
        mode=DataLoaderMode.AsyncRebatched if config.async_dataloader else DataLoaderMode.Sync,
    )
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files", help="A glob expression for the input files to read in and pack", required=True, type=str
    )
    parser.add_argument("--output-dir", help="The destination folder for the output files", required=True)
    parser.add_argument("--random-seed", help="For shuffling the data", default=12345)
    parser.add_argument(
        "--drop-unpacked-remainder", help="Whether to drop sequences that failed to pack", default=False, type=eval
    )
    parser.add_argument(
        "--unpacked-dataset-duplication-factor",
        help="The duplication factor with which create_pretraining_data.py was run to create the un-packed dataset. "
        + "The output dataset will always have duplication factor 1",
        default=1,
        type=int,
    )
    parser.add_argument("--sequence-length", help="The maximum number of tokens in an example", default=128, type=int)
    parser.add_argument(
        "--mask-tokens", help="The maximum number of masked tokens in an un-packed example", default=20, type=int
    )
    parser.add_argument(
        "--max-sequences-per-pack",
        help="The maximum number of sequences per packed example.",
        choices=[2, 3],
        default=3,
        type=int,
    )
    parser.add_argument(
        "--load-batch-size",
        help="The number of sequences to load at a time, set it to 1 to avoid"
        " dropping any sequences when loading the dataset (for eval)",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num-packing-workers",
        help="Max number of worker subprocesses to be used for packing sequences",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--chunks-per-packing-worker",
        help="Approximate number of chunks to be packed by each packing subprocess",
        default=8,
        type=int,
    )
    args = parser.parse_args()
    random.seed(args.random_seed)

    # Input files
    input_files = glob.glob(args.input_files)
    assert len(input_files) > 0
    print(f"\nInput files: {input_files}")

    # Load un-packed dataset (1 sequence per pack)
    # We borrow the config for pretrain_base_128 here
    data_args = """
    --config pretrain_base_128
    """.split()
    config = BertConfig(**(vars(parse_bert_args(data_args))))
    config.input_files = input_files
    config.compile_only = True
    opts = get_options(config)
    dataset = get_dataloader(config, opts)

    # Put examples into bins depending on their sequence lengths and extract the sequence length
    # as an array.
    sequence_lengths = []
    print("Looping through dataset to collect sequence length information...")
    start = time.time()
    for data in tqdm(dataset):
        # Use data[1] because data[0] could contain false "0"s
        real_tokens = (data[1] != 0).sum(1).tolist()
        sequence_lengths.extend(real_tokens)

    print(
        f"Done looping through dataset. Took {time.time() - start:3.3f} seconds to read {len(sequence_lengths)} sequences"
    )

    # Use the strategy_set and mixture to pack the dataset
    print(f"\nPacked dataset will be written to {args.output_dir}.")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    start = time.time()

    # Convert seq len info to np array
    sequence_lengths = np.array(sequence_lengths)
    assert len(sequence_lengths) > 1
    assert max(sequence_lengths) <= args.sequence_length

    # Run the packing algorithm on these sequence lengths
    strategy_set, mixture, padding = get_packing_recipe(args, sequence_lengths, drop_unused_strategies=True)
    target_num_sequences = int(mixture.sum())
    examples_by_length = defaultdict(list)
    print("Adding padding sequence to pack the remainder of sequences.")
    for i in range(1, args.sequence_length + 1):
        examples_by_length[i].extend([None] * int(padding[i - 1]))

    # Set the maximum number of sequences to write per output file.
    SEQUENCES_PER_FILE = 1000000
    PACKS_PER_FILE = SEQUENCES_PER_FILE // 2

    file_index = 0
    count = 0
    count_at_last_slice = 0
    write_count = 0
    example_slices_buffer = []
    packs_buffer = []
    packs_futures = []
    packing_executor = ProcessPoolExecutor(max_workers=args.num_packing_workers)

    print("Begin packing and writing.")
    for data in tqdm(dataset):
        data_as_arrays = [d.detach().numpy() for d in data]
        # Use data[1] because data[0] could contain false "0"s
        real_tokens = np.sum(data_as_arrays[1] != 0, axis=1).tolist()
        # Individual examples must be copied to ensure no references are held to the original (even if cloned) tensors in `data`.
        for i, length in enumerate(real_tokens):
            examples_by_length[length].append([np.array(d[i]) for d in data_as_arrays])
        del data_as_arrays
        count += len(real_tokens)

        seen_all_examples = force_pack = force_write = count == len(sequence_lengths)
        if count - count_at_last_slice >= SEQUENCES_PER_FILE or seen_all_examples:
            count_at_last_slice = count
            # Shuffle the data
            for key in examples_by_length:
                random.shuffle(examples_by_length[key])
            example_slices, strategies, mixture, total_packs_to_be_written = slice_examples(
                examples_by_length, strategy_set, mixture
            )
            for example_slice in example_slices:
                example_slices_buffer.extend(zip(*example_slice))

        while len(example_slices_buffer) >= PACKS_PER_FILE or force_pack:
            if len(packs_futures) > 0:
                packs_buffer.extend(realise_futures(packs_futures))
                packs_futures = []

            example_slices_to_pack, example_slices_buffer = (
                example_slices_buffer[:PACKS_PER_FILE],
                example_slices_buffer[PACKS_PER_FILE:],
            )
            force_pack = force_pack and len(example_slices_buffer) > 0
            packs_futures = submit_example_slices_for_packing(args, packing_executor, example_slices_to_pack)
            del example_slices_to_pack

        while len(packs_buffer) >= PACKS_PER_FILE or force_write:
            if force_write and len(packs_futures) > 0:
                packs_buffer.extend(realise_futures(packs_futures))
                packs_futures = []

            packs_to_write, packs_buffer = packs_buffer[:PACKS_PER_FILE], packs_buffer[PACKS_PER_FILE:]
            force_write = force_write and len(packs_buffer) > 0

            filename = os.path.join(args.output_dir, f"wiki_{file_index:03d}.tfrecord")
            write_start = time.time()
            writer = tf.io.TFRecordWriter(filename)
            for tf_example, is_usable in packs_to_write:
                if is_usable:
                    writer.write(tf_example)
            writer.close()
            file_index += 1
            num_packs_written = len(packs_to_write)
            write_count += num_packs_written
            del packs_to_write
            print(f"\nWrote {num_packs_written} packs into {filename}.")
            print("\nMemory usage:")
            print(psutil.virtual_memory())

    assert len(packs_futures) == 0
    packing_executor.shutdown(wait=True)

    print(f"\n-----------------------------------------------------------")
    print(f"Packing took: {time.time() - start:3.2f} seconds.", f"{mixture.sum()} packs left to fill.\n")

    assert target_num_sequences == write_count, (target_num_sequences, write_count)
