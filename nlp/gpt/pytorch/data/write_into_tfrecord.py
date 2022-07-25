# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import collections
import os
import pdb
import pickle
from tqdm import tqdm
import argparse

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required to generate data for this application. "
                      "Please install with: 'pip install tensorflow==1.15.0'")


def write_instance_to_example_files(train_path, output_dir, max_seq_length, stride, num_output):
    """Create TF example files from `TrainingInstance`s."""

    def to_tfrecord(input_list, writers):
        writer_index = 0
        total_written = 0
        samples = []
        article = input_list
        # for article in tqdm(input_list):
        start_point = 0
        while start_point < len(article) - max_seq_length:
            samples.append(article[start_point: start_point + max_seq_length])
            start_point += stride
        if start_point < len(article) - (max_seq_length // 2):
            samples.append(article[start_point:])
        random.shuffle(samples)
        for (inst_index, input_ids) in enumerate(tqdm(samples)):
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(input_ids)

            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writers[writer_index].write(tf_example.SerializeToString())
            if total_written % 100 == 0:
                # update writer_index
                writer_index = (writer_index + 1) % len(writers)
            total_written += 1
        for writer in writers:
            writer.close()
        return total_written

    writers = []
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_files = [os.path.join(
        output_dir, 'data_{}.tfrecord'.format(i)) for i in range(num_output)]
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    total_written = to_tfrecord(input_list, writers)
    tf.compat.v1.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file-path', default='./wikicorpus_en_one_article_per_line2.pkl', required=False, type=str)
    parser.add_argument(
        '--output-file-path', default='./tfrecords_50264_128/', required=False, type=str)
    parser.add_argument('--seq-length', default=128, type=int,
                        required=False, help='sequence length of dataset')
    parser.add_argument('--stride', default=128, type=int,
                        required=False, help='stride window size to sample dataset')
    parser.add_argument('--num-output', type=int, default=4,
                        help="number of output files")
    args = parser.parse_args()
    write_instance_to_example_files(train_path=args.input_file_path, output_dir=args.output_file_path,
                                    max_seq_length=args.seq_length, stride=args.stride, num_output=args.num_output)
