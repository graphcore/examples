# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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

import tensorflow.compat.v1 as tf
import numpy as np
from .pretraining import synthetic_pretraining_dataset, get_pretraining_dataset
from .squad import get_squad_dataset
from .glue import get_glue_dataset

from log import logger


def load(opts, is_training=True):
    data_type = tf.float32 if opts["precision"] == '32' else tf.float16
    if opts['task'] == 'pretraining':
        tf.set_random_seed(int(opts['seed']))
        np.random.seed(int(opts['seed']))
        if opts['generated_data']:
            return synthetic_pretraining_dataset(opts)
        else:
            return get_pretraining_dataset(
                opts,
                data_type,
                is_training,
                num_cpu_threads=int(opts['parallel_io_threads']),
                use_static_mask=opts['static_mask'])
    elif opts['task'] == 'SQuAD':
        return get_squad_dataset(opts, is_training)
    elif opts['task'] == 'GLUE':
        return get_glue_dataset(opts, is_training)
    else:
        raise ValueError(
            "Conflict options between generated_data and input_file")


def get_dataset_files_count(opts, is_training=True):
    if opts['generated_data']:
        return 10000000
    if is_training:
        input_file = opts['train_file']
    else:
        input_file = opts['test_file']
    input_files = []
    for input_pattern in input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    total_samples = 0
    for each_input_file in input_files:
        total_samples += sum(
            1 for _ in tf.python_io.tf_record_iterator(each_input_file))
    return total_samples


def set_defaults(opts):
    # If the execution profile is on we set the data as generated
    if opts['execution_profile'] or opts['synthetic_data']:
        opts['generated_data'] = True
    # If the synthetic data is passed together with the train_file we want to use the synthetic data
    if opts['generated_data'] and opts['train_file']:
        logger.warning("generated-data flag passed, truning off the train_file flag")
        opts['train_file'] = None
        opts['test_file'] = None

    if opts['task'].lower() != 'glue':
        # If neither the synthetic data nor the train_file flags are passed we are going to proceed with synthetic data
        if not opts['generated_data'] and not opts['train_file'] and not opts['predict_file']:
            raise ValueError(
                'Neither generated-data nor train-file was passed, set at least one of them to run the code.')
    # In the case of synthetic-data training we set the duplication factor to 1
    if opts['generated_data']:
        opts['duplication_factor'] = 1
