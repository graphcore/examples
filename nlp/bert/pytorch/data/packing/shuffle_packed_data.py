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
import math
import argparse
import psutil
from concurrent.futures import ProcessPoolExecutor
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required to generate data for this application. "
                      "Please install with: 'pip install tensorflow'")
from pretraining_data import TFRecordPretrainingDataset, _WorkerInit
from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
from transformers import BertConfig
from ipu_options import get_options
from args import parse_bert_args
from tqdm import tqdm
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def create_tf_example(packed_sequence):
    """Converts the list corresponding to a packed sequence into binary

    Parameters
    ----------
    packed_sequence:list [ints and floats]
        List with the data for one packed sequence, containing:
            - packed_input_ids
            - packed_input_mask
            - packed_segment_ids
            - packed_position_ids
            - packed_masked_lm_positions
            - packed_masked_lm_ids
            - packed_masked_lm_mask
            - packed_next_sentence_labels
            - packed_next_sentence_mask

    Returns
    -------
    line:binary str
        The binary representation of a packed sequence, ready to be written to a file.
    """
    features = collections.OrderedDict()
    features["packed_input_ids"] = create_int_feature(packed_sequence[0])
    features["packed_input_mask"] = create_int_feature(packed_sequence[1])
    features["packed_segment_ids"] = create_int_feature(packed_sequence[2])
    features["packed_position_ids"] = create_int_feature(packed_sequence[3])
    features["packed_masked_lm_positions"] = create_int_feature(packed_sequence[4])
    features["packed_masked_lm_ids"] = create_int_feature(packed_sequence[5])
    features["packed_masked_lm_mask"] = create_float_feature(packed_sequence[6])
    features["packed_next_sentence_labels"] = create_int_feature(packed_sequence[7])
    features["packed_next_sentence_mask"] = create_float_feature(packed_sequence[8])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example.SerializeToString()


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def write_to_file(args, executor, writer, packed_sequences):
    """Write data to files

    Parameters
    ----------
    args:namedtuple containing the following attributes
        num_workers:int
            The maximum number of worker subprocesses to be used for converting sequences.
        chunks_per_worker:int
            Approximate number of chunks to divide the example slices due to be packed by each conversion subprocess.
    executor:concurrent.futures.ProcessPoolExecutor
        A wrapper around the multiprocessing module enabling asynchronous execution to which we submit
        chunks of packed_sequences.
    writer:File
        Handle to a file open for binary writing.
    packed_sequences:list [ints and floats]
        Each component of the list is a list with the data for one packed sequence, containing:
            - packed_input_ids
            - packed_input_mask
            - packed_segment_ids
            - packed_position_ids
            - packed_masked_lm_positions
            - packed_masked_lm_ids
            - packed_masked_lm_mask
            - packed_next_sentence_labels
            - packed_next_sentence_mask

    Returns
    -------
    This function does not return anything, instead it writes its output directly to the file.
    """
    packs_per_worker = len(packed_sequences) // args.num_workers + (len(packed_sequences) % args.num_workers > 0)
    chunksize = max(1, packs_per_worker // args.chunks_per_worker)
    for tf_example in executor.map(create_tf_example, packed_sequences, chunksize=chunksize):
        writer.write(tf_example)


def get_dataloader(config, opts):
    dataset = TFRecordPretrainingDataset(config.input_files, packed_data=config.packed_data)
    loader = DataLoader(opts,
                        dataset,
                        batch_size=config.micro_batch_size,
                        num_workers=config.dataloader_workers,
                        drop_last=False,
                        worker_init_fn=_WorkerInit(config.random_seed),
                        mode=DataLoaderMode.AsyncRebatched if config.async_dataloader else DataLoaderMode.Sync)
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", help="A glob expression for the input files to read in and pack", required=True, type=str)
    parser.add_argument("--output-dir", help="The destination folder for the output files", required=True)
    parser.add_argument("--random-seed", help="For shuffling the data", default=12345)
    parser.add_argument("--num-workers", help="Max number of worker subprocesses to be used for converting sequences", default=16, type=int)
    parser.add_argument("--chunks-per-worker", help="Approximate number of chunks to be handled by each conversion subprocess", default=8, type=int)
    args = parser.parse_args()
    random.seed(args.random_seed)

    # Input files
    input_files = glob.glob(args.input_files)
    assert len(input_files) > 0
    print(f"\nInput files: {input_files}")

    # We borrow the config for pretrain_base_128_packed here
    data_args = """
    --config pretrain_base_128_packed
    """.split()
    config = BertConfig(**(vars(parse_bert_args(data_args))))
    config.input_files = input_files
    config.compile_only = True
    opts = get_options(config)
    dataset = get_dataloader(config, opts)

    # Check that the output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    start_total = time.time()

    PACKS_PER_FILE = 500000
    # Trading off peak memory consumption for shuffling quality. Value chosen to match previous scheme where
    # all packed sequences were split into 20 chunks.
    PACKS_BUFFER_SIZE = 5 * PACKS_PER_FILE

    file_index = 0
    packs_buffer = []
    executor = ProcessPoolExecutor(args.num_workers)

    # Read the data and append it into the packed_sequences list
    print(f"\nReading the data.")
    for idx, data in enumerate(tqdm(dataset)):
        data_as_arrays = [d.detach().numpy() for d in data]
        num_examples = len(data_as_arrays[0])
        for i in range(num_examples):
            packs_buffer.append([np.copy(d[i]) for d in data_as_arrays])
        del data_as_arrays

        force_write = idx == len(dataset) - 1
        while len(packs_buffer) >= PACKS_BUFFER_SIZE or force_write:
            packed_sequences, packs_buffer = packs_buffer[:PACKS_BUFFER_SIZE], packs_buffer[PACKS_BUFFER_SIZE:]
            force_write = force_write and len(packs_buffer) > 0

            print(f"\nThe number of packs being shuffled is {len(packed_sequences)}.")
            start = time.time()
            random.shuffle(packed_sequences)
            print(f"\nTime to shuffle: {time.time() - start:3.2f} seconds.")

            while len(packed_sequences) > 0:
                packs_to_write, packed_sequences = packed_sequences[:PACKS_PER_FILE], packed_sequences[PACKS_PER_FILE:]

                filename = os.path.join(args.output_dir, f"wiki_{file_index:03d}.tfrecord")
                start = time.time()
                print(f"\n-----------------------------------------------------------")
                print(f"\nWriting {len(packs_to_write)} packs into {filename}.")
                writer = tf.io.TFRecordWriter(filename)
                write_to_file(args, executor, writer, packs_to_write)
                writer.close()
                print("\nMemory usage:")
                print(psutil.virtual_memory())
                print(f"\nTime to write into {filename}: {time.time() - start:3.2f} seconds.")
                file_index += 1
    executor.shutdown(wait=True)

    print(f"\n-----------------------------------------------------------")
    print(f"\nTotal time to write data into files: {time.time() - start_total:3.2f} seconds.")
