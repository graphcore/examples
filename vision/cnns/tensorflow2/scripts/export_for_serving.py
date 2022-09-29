# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import argparse
import logging
import os
import sys

from tensorflow import keras
from tensorflow.python import ipu
from tensorflow.python.ipu import serving
import tensorflow as tf

from batch_config import BatchConfig
from configuration import terminal_argparse
from datasets.dataset_factory import DatasetFactory
from eight_bit_transfer import EightBitTransfer
from ipu_config import configure_ipu
from model.model_factory import ModelFactory
import precision
import seed
sys.path.append(os.getcwd())


def get_parser():
    parser = argparse.ArgumentParser(description='Exporting TF2 classification Models for Serving',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--export-dir', type=str, default=None,
                        help='Path to the directory where the SavedModel will be written.')
    parser.add_argument('--iterations', type=int, default=1,
                        help="Number of iterations for the model exported for TensorFlow Serving.")
    parser.add_argument('--pipeline-serving-model', type=terminal_argparse.str_to_bool, nargs="?", const=True, default=False,
                        help='Reuse the training pipeline splits for inference in the model exported for TensorFlow Serving')
    parser.add_argument('--checkpoint-file', type=str, default=None,
                        help='Path to a checkpoint file that will be loaded before exporting model for TensorFlow Serving. '
                             'If not set, the model will use randomly initialized parameters.')
    parser.add_argument('--use-serving-api', action='store_true',
                        help='Use TensorFlow serving API for model export')
    return parser


def validate_arguments(hparams):
    if hparams.export_dir is None:
        raise ValueError('Unspecified directory for the exported SavedModel. '
                         'Make sure --export-dir is defined.')
    elif os.path.exists(hparams.export_dir) and not os.path.isdir(hparams.export_dir):
        raise ValueError(
            f'--export-dir is set to "{hparams.export_dir}" which already exists and is not a directory.')
    elif os.path.isdir(hparams.export_dir) and os.listdir(hparams.export_dir):
        raise ValueError(f'--export-dir is set to "{hparams.export_dir}" which already exists, and is not empty. '
                         'Please choose a different export directory, or delete all the contents of the specified directory.')

    if not hparams.checkpoint_file:
        logging.warn('No checkpoint file provided, model\'s parameters will be initialized randomly.'
                     'Set --checkpoint-file to load trained parameters.')
    elif not os.path.exists(hparams.checkpoint_file):
        raise ValueError(
            f'--checkpoint-file is set to "{hparams.checkpoint_file}" which does not exists.')

    if (not len(hparams.pipeline_splits)) and hparams.pipeline_serving_model:
        logging.warn(
            'Pipeline splits have not been defined, turning off the pipeline-serving-model option.')
        hparams.pipeline_serving_model = False

    if hparams.iterations == 1:
        logging.warn('--iterations is set to 1 which is default value.')

    return hparams


if __name__ == '__main__':
    # configure logger
    logging.basicConfig(level=logging.INFO)

    parser = get_parser()
    hparams = terminal_argparse.handle_cmdline_arguments(parser)
    hparams = validate_arguments(hparams)

    hparams.seed = seed.set_host_seed(hparams.seed)
    hparams.num_replicas = 1

    fp_precision = precision.Precision(hparams.precision)
    fp_precision.apply()

    # Get eight bit transfer object
    eight_bit_transfer = EightBitTransfer(
        fp_precision.compute_precision) if hparams.eight_bit_transfer else None

    batch_config = BatchConfig(micro_batch_size=hparams.micro_batch_size,
                               num_replicas=1,
                               gradient_accumulation_count=1)

    # Get the validation dataset
    ds, accelerator_side_preprocess_inference_fn, hparams.pipeline_num_parallel = DatasetFactory.get_dataset(
        dataset_name=hparams.dataset,
        dataset_path=hparams.dataset_path,
        split='test',
        img_datatype=fp_precision.compute_precision,
        batch_config=batch_config,
        seed=hparams.seed,
        accelerator_side_preprocess=hparams.accelerator_side_preprocess,
        eight_bit_transfer=eight_bit_transfer,
        pipeline_num_parallel=hparams.pipeline_num_parallel,
        fused_preprocessing=hparams.fused_preprocessing,
        synthetic_data=hparams.synthetic_data)
    logging.debug(ds)

    if not hparams.pipeline_serving_model:
        hparams.num_ipus_per_replica = 1
    cfg = configure_ipu(hparams)
    seed.set_ipu_seed(hparams.seed)

    # Create an IPU distribution strategy
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        if hparams.use_serving_api:
            input_tensor = None
        else:
            input_tensor = keras.layers.Input(ds.image_shape,
                                              batch_size=hparams.micro_batch_size)
        # Create an instance of the model
        model = ModelFactory.create_model(model_name=hparams.model_name,
                                          input_shape=ds.image_shape,
                                          classes=ds.num_classes,
                                          accelerator_side_preprocessing_fn=accelerator_side_preprocess_inference_fn,
                                          eight_bit_transfer=eight_bit_transfer,
                                          norm_layer_params=hparams.norm_layer,
                                          input_tensor=input_tensor)

        if hparams.pipeline_serving_model:
            model = ModelFactory.configure_model(model=model,
                                                 gradient_accumulation_count=1,
                                                 pipeline_splits=hparams.pipeline_splits,
                                                 device_mapping=hparams.device_mapping,
                                                 pipeline_schedule=hparams.pipeline_schedule,
                                                 available_memory_proportion=hparams.available_memory_proportion,
                                                 optimizer_state_offloading=hparams.optimizer_state_offloading)
        else:
            model = ModelFactory.configure_model(model=model,
                                                 gradient_accumulation_count=1,
                                                 pipeline_splits=[],
                                                 device_mapping=None,
                                                 pipeline_schedule=None,
                                                 available_memory_proportion=hparams.available_memory_proportion,
                                                 optimizer_state_offloading=hparams.optimizer_state_offloading)

        if not hparams.use_serving_api:
            model.compile(steps_per_execution=hparams.iterations)
        model.build(input_shape=(hparams.micro_batch_size,
                                 ds.image_shape[0],
                                 ds.image_shape[1],
                                 ds.image_shape[2]))

        if hparams.checkpoint_file:
            logging.info(f'Loading checkpoint file {hparams.checkpoint_file}')
            model.load_weights(hparams.checkpoint_file)
        input_dtype = tf.uint8 if hparams.eight_bit_transfer else fp_precision.compute_precision

        if not hparams.use_serving_api:
            model.export_for_ipu_serving(
                hparams.export_dir, batch_size=hparams.micro_batch_size)
        else:
            input_signature = (tf.TensorSpec(shape=(hparams.micro_batch_size,
                                                    ds.image_shape[0],
                                                    ds.image_shape[1],
                                                    ds.image_shape[2]), dtype=input_dtype),)

            @tf.function(input_signature=input_signature)
            def predict_step(input):
                res = model(inputs=input)
                return res
            serving.export_single_step(
                predict_step, hparams.export_dir, hparams.iterations)
