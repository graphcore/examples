# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 The Google Research Authors.
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
"""An implementation of Cluster-GCN as described in:
  Cluster-GCN: An Efficient Algorithm for Training Deep and
  Large Graph Convolutional Networks
  https://arxiv.org/abs/1905.07953
"""
import argparse
from datetime import datetime
import logging

import tensorflow as tf
import wandb

from data_utils.clustering_utils import partition_graph
from data_utils.dataset_batch_generator import tf_dataset_generator
from data_utils.dataset_loader import Dataset
from keras_extensions.callbacks.callback_factory import CallbackFactory
from keras_extensions.optimization import get_optimizer
from model.loss_accuracy import get_loss_accuracy_f1score
from model.model import create_model
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from model.precision import Precision
from utilities.argparser import add_arguments, combine_config_file_with_args
from utilities.checkpoint_utility import load_checkpoint_into_model
from utilities.ipu_utils import create_ipu_strategy, set_random_seeds
from utilities.options import Options
from utilities.pipeline_stage_assignment import pipeline_model


def run(config):
    """Run training and validation on the model."""

    # Set a name for this run
    universal_run_name = (
        f"{config.name}-"
        f"{config.dataset_name}-"
        f"{datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y%m%d_%H%M%S')}"
    )
    logging.info(f"Universal name for run: {universal_run_name}")

    # Initialise Weights & Biases logging
    if config.wandb:
        wandb.init(entity="sw-apps",
                   project="TF2-Cluster-GCN",
                   name=universal_run_name,
                   config=config.dict())

    logging.info(f"Config: {config}")

    dataset = Dataset(
        dataset_path=config.data_path,
        dataset_name=config.dataset_name,
        precalculate_first_layer=config.model.first_layer_precalculation
    )

    # Set precision policy
    precision = Precision(config.precision)
    tf.keras.mixed_precision.set_global_policy(precision.policy)

    if config.do_training:
        logging.info(f"Running training on {config.training.device}...")

        partitions_training, max_nodes_per_batch_training = partition_graph(
            adj=dataset.adjacency_train,
            directed_graph=dataset.directed_graph,
            idx_nodes=dataset.train_data,
            num_clusters=config.training.num_clusters,
            clusters_per_batch=config.training.clusters_per_batch,
        )

        data_generator_training = tf_dataset_generator(
            adjacency=dataset.adjacency_train,
            partitions=partitions_training,
            features=dataset.features_train,
            labels=dataset.labels_train,
            mask=dataset.mask_train,
            num_clusters=config.training.num_clusters,
            clusters_per_batch=config.training.clusters_per_batch,
            max_nodes_per_batch=max_nodes_per_batch_training,
            seed=config.seed
        )

        # Calculate the number of pipeline stages and the number of required IPUs per replica.
        num_pipeline_stages_training = len(config.training.ipu_config.pipeline_device_mapping)
        num_ipus_per_replica_training = max(config.training.ipu_config.pipeline_device_mapping) + 1

        strategy_training_scope = create_ipu_strategy(
            num_ipus_per_replica=num_pipeline_stages_training,
            num_replicas=1,
            compile_only=config.compile_only,
        ).scope() if config.training.device == "ipu" else tf.device("/cpu:0")

        set_random_seeds(config.seed)

        num_nodes_processed_per_execution_training = (
                max_nodes_per_batch_training * config.training.steps_per_execution
        )
        steps_per_epoch_training = (
                config.training.num_clusters // config.training.clusters_per_batch
        )
        steps_per_epoch_training = (
                (steps_per_epoch_training // config.training.steps_per_execution) *
                config.training.steps_per_execution
        )

        with strategy_training_scope:
            model_training = create_model(
                num_labels=dataset.num_labels,
                num_features=dataset.num_features,
                max_nodes_per_batch=max_nodes_per_batch_training,
                hidden_size=config.model.hidden_size,
                num_layers=config.model.num_layers,
                dropout_rate=config.model.dropout,
                adjacency_params=config.model.adjacency.dict(),
                cast_model_inputs_to_dtype=precision.cast_model_inputs_to_dtype,
                first_layer_precalculation=config.model.first_layer_precalculation,
                use_ipu_layers=(config.training.device == "ipu")
            )
            model_training.summary(print_fn=logging.info)

            if num_pipeline_stages_training > 1:
                pipeline_model(model=model_training,
                               config=config.training,
                               pipeline_names=PIPELINE_NAMES,
                               pipeline_allocate_previous=PIPELINE_ALLOCATE_PREVIOUS,
                               num_ipus_per_replica=num_ipus_per_replica_training)

            loss, accuracy, f1_score = get_loss_accuracy_f1score(
                task=dataset.task,
                num_labels=dataset.num_labels,
                metrics_precision=precision.metrics_precision)

            optimizer = get_optimizer(
                gradient_accumulation_steps_per_replica=config.training.gradient_accumulation_steps_per_replica,
                num_replicas=1,
                learning_rate=tf.cast(config.training.lr, dtype=tf.float32),
                loss_scaling=config.training.loss_scaling,
                optimizer_compute_precision=precision.optimizer_compute_precision
            )
            model_training.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[accuracy, f1_score],
                steps_per_execution=config.training.steps_per_execution,
            )
            callbacks_training = CallbackFactory.get_callbacks(
                universal_run_name=universal_run_name,
                num_nodes_processed_per_execution=num_nodes_processed_per_execution_training,
                checkpoint_path=config.save_ckpt_path.joinpath(universal_run_name),
                config=config.dict(),
                executions_per_log=config.executions_per_log,
                executions_per_ckpt=config.executions_per_ckpt,
                outfeed_queues=[loss.outfeed_queue]
            )
            model_training.fit(
                data_generator_training,
                epochs=config.training.epochs,
                steps_per_epoch=steps_per_epoch_training,
                callbacks=callbacks_training,
                verbose=0
            )
            trained_weights = model_training.get_weights()
            logging.info("Training complete")

    if config.do_validation:
        logging.info(f"Running validation on {config.validation.device}...")

        partitions_validation, max_nodes_per_batch_validation = partition_graph(
            adj=dataset.adjacency_full,
            directed_graph=dataset.directed_graph,
            idx_nodes=dataset.visible_data_validation,
            num_clusters=config.validation.num_clusters,
            clusters_per_batch=config.validation.clusters_per_batch,
        )

        data_generator_validation = tf_dataset_generator(
            adjacency=dataset.adjacency_full,
            partitions=partitions_validation,
            features=dataset.features_test,
            labels=dataset.labels_validation,
            mask=dataset.mask_validation,
            num_clusters=config.validation.num_clusters,
            clusters_per_batch=config.validation.clusters_per_batch,
            max_nodes_per_batch=max_nodes_per_batch_validation,
            seed=config.seed
        )

        # Calculate the number of pipeline stages and the number of required IPUs per replica.
        num_pipeline_stages_validation = len(config.validation.ipu_config.pipeline_device_mapping)
        num_ipus_per_replica_validation = max(config.validation.ipu_config.pipeline_device_mapping) + 1

        strategy_validation_scope = create_ipu_strategy(
            num_ipus_per_replica=num_pipeline_stages_validation,
            num_replicas=1,
            compile_only=config.compile_only,
        ).scope() if config.validation.device == "ipu" else tf.device("/cpu:0")

        set_random_seeds(config.seed+1)

        steps_per_epoch_validation = (
                config.validation.num_clusters // config.validation.clusters_per_batch
        )
        steps_per_epoch_validation = (
                (steps_per_epoch_validation // config.validation.steps_per_execution) *
                config.validation.steps_per_execution
        )

        with strategy_validation_scope:
            model_validation = create_model(
                num_labels=dataset.num_labels,
                num_features=dataset.num_features,
                max_nodes_per_batch=max_nodes_per_batch_validation,
                hidden_size=config.model.hidden_size,
                num_layers=config.model.num_layers,
                dropout_rate=config.model.dropout,
                adjacency_params=config.model.adjacency.dict(),
                first_layer_precalculation=config.model.first_layer_precalculation,
                use_ipu_layers=(config.validation.device == "ipu")
            )

            if config.do_training:
                # Copy the weights from training
                model_validation.set_weights(trained_weights)
            else:
                # Load weights from a checkpoint file
                if not config.load_ckpt_path:
                    raise ValueError("Training has been skipped but no"
                                     " checkpoint has been provided.")
                load_checkpoint_into_model(model_validation, config.load_ckpt_path)

            if num_pipeline_stages_validation > 1 and config.validation.device == "ipu":
                pipeline_model(model=model_validation,
                               config=config.validation,
                               pipeline_names=PIPELINE_NAMES,
                               pipeline_allocate_previous=PIPELINE_ALLOCATE_PREVIOUS,
                               num_ipus_per_replica=num_ipus_per_replica_validation)

            _, accuracy, f1_score = get_loss_accuracy_f1score(
                task=dataset.task,
                num_labels=dataset.num_labels,
                metrics_precision=precision.metrics_precision)

            model_validation.compile(metrics=[accuracy, f1_score],
                                     steps_per_execution=config.validation.steps_per_execution)

            results = model_validation.evaluate(data_generator_validation,
                                                steps=steps_per_epoch_validation,
                                                verbose=0)
            logging.info(f"Validation Accuracy: {results[1]}, Validation F1: {results[2]}")
            if config.wandb:
                wandb.log({"validation_acc": results[1], "validation_f1": results[2]})
            logging.info("Validation complete")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # Prevent doubling of TF logs.
    tf.get_logger().propagate = False

    # Load arguments and config
    parser = argparse.ArgumentParser(description="TF2 Cluster-GCN")
    args = add_arguments(parser).parse_args()
    cfg = combine_config_file_with_args(args, Options)

    # Set log level based on config
    logging.getLogger().setLevel(cfg.logging)

    # Run training and validation
    run(cfg)
