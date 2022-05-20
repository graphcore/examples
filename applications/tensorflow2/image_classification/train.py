# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import glob
import shutil
import logging
import argparse
import warnings
from time import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu.horovod.popdist_strategy import PopDistStrategy
import popdist
import popdist.tensorflow

import precision
import time_to_train
from batch_config import BatchConfig
from ipu_config import configure_ipu
from callbacks.callback_factory import CallbackFactory
from callbacks.callbacks_periodicity import calculate_log_period
from configuration import file_argparse, terminal_argparse
from custom_exceptions import DimensionError, UnallowedConfigurationError
from data.dataset_factory import DatasetFactory
from eight_bit_transfer import EightBitTransfer
from losses.loss_enqueuer import (wrap_loss_in_allreduce_enqueuer,
                                  wrap_loss_in_enqueuer,
                                  wrap_loss_in_label_enqueuer,
                                  wrap_loss_in_pred_enqueuer)
from losses.smoothed_categorical_crossentropy import SmoothedCategoricalCrossentropy
from metrics.metric_enqueuer import (wrap_metric_in_allreduce_enqueuer,
                                     wrap_metric_in_enqueuer)
from model.model_factory import ModelFactory, replace_preprocess_layer_with_fn
from optimizers.optimizer_factory import OptimizerFactory
from schedules.scheduler_builder import get_lr_scheduler
import seed


if __name__ == '__main__':
    # configure logger
    logging.basicConfig(level=logging.INFO)

    # create an argument parser
    parser = argparse.ArgumentParser(description='TF2 classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = terminal_argparse.add_arguments(parser)
    hparams = parser.parse_args()
    hparams = file_argparse.parse_yaml_config(hparams, parser)

    hparams.validation_micro_batch_size = hparams.validation_micro_batch_size or hparams.micro_batch_size

    # check if the script has been called by poprun
    hparams.distributed_training = popdist.isPopdistEnvSet()

    if hparams.distributed_training:

        if hparams.num_replicas != popdist.getNumTotalReplicas():
            logging.warning(f'Replication factor given to poprun (=={popdist.getNumTotalReplicas()}) '
                            f'does not match the config (=={hparams.num_replicas}). Poprun will override the config.')
            hparams.num_replicas = popdist.getNumTotalReplicas()

        hvd.init()
        hparams.num_hosts = hvd.size() // hvd.local_size()

        logging.info(f'Total number of instances {popdist.getNumInstances()}')
        logging.info(f'Local number of instances {hvd.local_size()}')

        if not hparams.pipeline_validation_model and hparams.validation and hparams.training and len(hparams.pipeline_splits) > 0:
            raise ValueError(
                'pipeline-validation-model must be set to True when validation happens after training in distributed and pipelined mode')

        if popdist.getInstanceIndex() != 0:
            hparams.wandb = False

        if hvd.local_rank() != 0 and not hparams.ckpt_all_instances:
            hparams.checkpoints = False
            hparams.clean_dir = False

    else:
        if hparams.ckpt_all_instances:
            raise ValueError(
                'There is only one instance, --ckpt-all-instances cannot be enabled without distributed training.')

        if hparams.accelerator_side_reduction:
            logging.warning('--accelerator-side-reduction can only be enabled with distributed training strategy. '
                            'Overwritting the setting to False.')
            hparams.accelerator_side_reduction = False

    hparams.num_instances = popdist.getNumInstances() if hparams.distributed_training else 1

    # when neither option is specified, assume gradient accumulation count 1
    if hparams.gradient_accumulation_count is None and hparams.global_batch_size is None:
        hparams.gradient_accumulation_count = 1

    if hparams.recomputation and not len(hparams.pipeline_splits):
        raise ValueError('Recomputation requires a pipelined model. '
                         'Make sure "--pipeline-splits" is defined')

    if (not len(hparams.pipeline_splits)) and hparams.pipeline_validation_model:
        logging.warn('Pipeline splits have not been defined, turning off the pipeline-validation-model option')
        hparams.pipeline_validation_model = False

    if hparams.logs_per_epoch < 0:
        raise ValueError(f'--logs-per-epoch should be non-negative (>=0), it is {hparams.logs_per_epoch}')

    # check for partial logs, example --logs-per-epoch 0.5 and --epochs 5
    if (hparams.logs_per_epoch > 0) and (hparams.logs_per_epoch < 1) and (hparams.num_epochs % (1 / hparams.logs_per_epoch) != 0):
        raise ValueError(
            f'It is not possible to log {1/hparams.logs_per_epoch} epochs a time for {hparams.num_epochs} epochs')

    num_pipeline_stages = len(hparams.pipeline_splits) + 1

    if hparams.device_mapping:
        if len(hparams.device_mapping) != num_pipeline_stages:
            raise DimensionError(
                f'The number of device assignments {len(hparams.device_mapping)} is not equal to the number of pipeline splits + 1: {num_pipeline_stages}.')

        if len(set(hparams.device_mapping)) != max(hparams.device_mapping) + 1:
            raise DimensionError(
                f'The model is pipelined over {len(set(hparams.device_mapping))} different IPUs, but one or more stages are being assigned to IPU {max(hparams.device_mapping) + 1}')

    if hparams.eight_bit_transfer and not hparams.accelerator_side_preprocess:
        raise UnallowedConfigurationError(
            f'When eight bit transfer is enabled the normalisation must be done on the device. '
            f'If you want to keep 8bit io, set --accelerator-side-preprocess to True.')

    if len(hparams.available_memory_proportion) > 1 and num_pipeline_stages == 1:
        raise UnallowedConfigurationError(
            'Setting available memory proportion per pipeline stage, '
            'but no pipeline stages defined. Please use --pipeline-splits to define the pipeline stages')

    if len(hparams.available_memory_proportion) > 1 and len(hparams.available_memory_proportion) != 2 * num_pipeline_stages:
        raise DimensionError(
            'Define a single global value of available memory proportion or two values per pipeline stage. '
            f'There are {num_pipeline_stages} pipeline stages defined and {len(hparams.available_memory_proportion)} values of '
            'available memory proportion')

    if not hparams.checkpoints and hparams.ckpt_all_instances:
        raise ValueError('All instances cannot save weights when checkpointing is disabled. '
                         'Pass --checkpoints True to save weights for all instances, or disable --ckpt-all-instances otherwise.')

    hparams.num_ipus_per_replica = num_pipeline_stages if not hparams.device_mapping else max(
        hparams.device_mapping) + 1

    if hparams.dbn_replica_group_size > 1 and hparams.num_ipus_per_replica != 1:
        raise ValueError('Distributed Batch Norm can only be applied when model fits on a single ipu.')

    if hparams.dbn_replica_group_size > 1 and hparams.num_replicas % hparams.dbn_replica_group_size != 0:
        raise ValueError('Distributed Batch Norm can only be applied when model is replicated, '
                         'and replication factor is divisible by dbn-replica-group-size.')

    if hparams.fused_preprocessing is True and hparams.accelerator_side_preprocess is False:
        raise ValueError('Fused preprocessing can only be done in the IPU. '
                         'Set both --fused_preprocessing and --accelerator-side-preprocess to True')

    if hparams.norm_layer['name'] not in {'batch_norm', 'custom_batch_norm', 'group_norm'}:
        raise ValueError(f'Normalization layer {hparams.norm_layer["name"]} not supported.')

    wandb_params_keys = set(hparams.wandb_params.keys())
    possible_keys = {'entity', 'project_name', 'run_name', 'tags'}
    unexpected_keys = wandb_params_keys - possible_keys
    if len(unexpected_keys) > 0:
        raise ValueError(f'wandb params contains unexpected fields: {unexpected_keys}')

    logging.info(f'hyperparams = {hparams}')
    hparams.seed = seed.set_host_seed(hparams.seed)

    batch_config = BatchConfig(hparams.micro_batch_size,
                               hparams.num_replicas,
                               hparams.gradient_accumulation_count,
                               hparams.global_batch_size)

    hparams.gradient_accumulation_count = batch_config.gradient_accumulation_count
    hparams.global_batch_size = batch_config.global_batch_size

    if hparams.validation:

        if hparams.pipeline_validation_model:
            hparams.validation_num_replicas = hparams.validation_num_replicas or hparams.num_replicas
            validation_gradient_accumulation_count = 2 * (len(hparams.pipeline_splits) + 1)
            hparams.validation_ipus_per_replica = hparams.num_ipus_per_replica
        else:
            hparams.validation_num_replicas = hparams.validation_num_replicas or (
                hparams.num_replicas * hparams.num_ipus_per_replica)
            validation_gradient_accumulation_count = 1
            hparams.validation_ipus_per_replica = 1

        validation_batch_config = BatchConfig(micro_batch_size=hparams.validation_micro_batch_size,
                                              num_replicas=hparams.validation_num_replicas,
                                              gradient_accumulation_count=validation_gradient_accumulation_count,
                                              global_batch_size=None)

    fp_precision = precision.Precision(hparams.precision)
    fp_precision.apply()

    # get eight bit transfer object
    eight_bit_transfer = EightBitTransfer(fp_precision.compute_precision) if hparams.eight_bit_transfer else None

    hparams.num_local_instances = hvd.local_size() if hparams.distributed_training else 1

    # Get the training dataset
    ds, img_shape, dataset_size, num_classes, accelerator_side_preprocess_train_fn, hparams.pipeline_num_parallel = DatasetFactory.get_dataset(
        dataset_name=hparams.dataset,
        dataset_path=hparams.dataset_path,
        split='train',
        img_datatype=fp_precision.compute_precision,
        micro_batch_size=batch_config.micro_batch_size,
        shuffle=True,
        accelerator_side_preprocess=hparams.accelerator_side_preprocess,
        eight_bit_transfer=eight_bit_transfer,
        pipeline_num_parallel=hparams.pipeline_num_parallel,
        num_local_instances=hparams.num_local_instances,
        fused_preprocessing=hparams.fused_preprocessing,
        seed=hparams.seed,
        synthetic_data=hparams.synthetic_data)
    logging.debug(ds)

    # Get the validation dataset
    if hparams.validation:
        ds_valid, _, ds_valid_size, _, accelerator_side_preprocess_inference_fn, hparams.pipeline_num_parallel = DatasetFactory.get_dataset(
            dataset_name=hparams.dataset,
            dataset_path=hparams.dataset_path,
            split='test',
            img_datatype=fp_precision.compute_precision,
            micro_batch_size=validation_batch_config.micro_batch_size,
            accelerator_side_preprocess=hparams.accelerator_side_preprocess,
            eight_bit_transfer=eight_bit_transfer,
            pipeline_num_parallel=hparams.pipeline_num_parallel,
            num_local_instances=hparams.num_local_instances,
            fused_preprocessing=hparams.fused_preprocessing,
            seed=hparams.seed,
            synthetic_data=hparams.synthetic_data)
        logging.debug(ds_valid)

    cfg = configure_ipu(hparams)

    seed.set_ipu_seed(hparams.seed)

    if hparams.weight_updates_per_epoch == -1:
        hparams.weight_updates_per_epoch = dataset_size // batch_config.global_batch_size
    micro_batches_per_epoch = hparams.weight_updates_per_epoch * batch_config.num_micro_batches_per_weight_update

    micro_batches_per_log = calculate_log_period(
        hparams.weight_updates_per_epoch, hparams.num_epochs, hparams.logs_per_epoch, batch_config)
    logging.info(f'micro batches per log {micro_batches_per_log}')

    # steps_per_execution is the number of weight updates in term of micro batches before going back to the host
    if micro_batches_per_log != 0:
        micro_batches_per_execution = micro_batches_per_log
    else:
        # run training run in a single call
        logging.warn('The entire training run will be executed in a single call to the device.')
        micro_batches_per_execution = micro_batches_per_epoch * hparams.num_epochs

    # if we do more than one epoch per device call we need to adjust the number of epochs
    # and the number of micro batches processed in an epoch
    if micro_batches_per_epoch < micro_batches_per_execution:
        total_num_micro_batches = micro_batches_per_epoch * hparams.num_epochs
        hparams.num_epochs = int(total_num_micro_batches / micro_batches_per_execution)
        micro_batches_per_epoch = micro_batches_per_execution

    if (micro_batches_per_execution > micro_batches_per_epoch):
        warnings.warn(
            f'micro_batches_per_execution = {micro_batches_per_execution} > micro_batches_per_epoch = {micro_batches_per_epoch}')
        warnings.warn(
            f'This is not possible as micro_batches_per_epoch is a series of micro_batches_per_execution')
        warnings.warn(f'You might consider changing the number of micro_batches and / or weight_updates_per_execution')
        micro_batches_per_execution = micro_batches_per_epoch

    # micro_batches_per_epoch is the number of running micro batches per epoch which can be larger or smaller
    # than the actual number of steps per epoch ( = number of micro batches per epoch covering the whole dataset)
    if micro_batches_per_epoch % micro_batches_per_execution:
        raise ValueError(
            f'micro_batches_per_execution {micro_batches_per_execution} should divide micro_batches_per_epoch = {micro_batches_per_epoch}')

    time_to_train_timer = time_to_train.TimeToTrain()

    # Create an IPU distribution strategy
    train_strategy = PopDistStrategy() if hparams.distributed_training else ipu.ipu_strategy.IPUStrategy()

    with train_strategy.scope():

        # Create an instance of the model
        model = ModelFactory.create_model(model_name=hparams.model_name,
                                          input_shape=img_shape,
                                          classes=num_classes,
                                          accelerator_side_preprocessing_fn=accelerator_side_preprocess_train_fn,
                                          eight_bit_transfer=eight_bit_transfer,
                                          norm_layer_params=hparams.norm_layer)

        # model debugging
        debug_outfeeds = []
        layers_to_debug = []
        model, debug_outfeeds = ModelFactory.debug_layers(model, debug_layers_names=layers_to_debug)

        model = ModelFactory.configure_model(model=model,
                                             gradient_accumulation_count=batch_config.gradient_accumulation_count,
                                             pipeline_splits=hparams.pipeline_splits,
                                             device_mapping=hparams.device_mapping,
                                             pipeline_schedule=hparams.pipeline_schedule,
                                             available_memory_proportion=hparams.available_memory_proportion,
                                             optimizer_state_offloading=hparams.optimizer_state_offloading)

        if hparams.training:
            # prepare the learning rate scheduler
            lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
                outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)
            lr_scheduler = get_lr_scheduler(
                scheduler_name=hparams.lr_schedule,
                schedule_params=hparams.lr_schedule_params,
                warmup_params=hparams.lr_warmup_params,
                global_batch_size=batch_config.global_batch_size,
                weight_updates_per_epoch=hparams.weight_updates_per_epoch,
                staircase=hparams.lr_staircase,
                queue=lr_outfeed_queue if hparams.synthetic_data != 'ipu' else None
            )

            # get weight decay scheduler
            wd_scheduler = get_lr_scheduler(
                scheduler_name=hparams.lr_schedule,
                schedule_params=hparams.lr_schedule_params,
                warmup_params=hparams.lr_warmup_params,
                global_batch_size=batch_config.global_batch_size,
                weight_updates_per_epoch=hparams.weight_updates_per_epoch,
                staircase=hparams.lr_staircase,
                queue=None,
                factor=hparams.weight_decay
            )

            # prepare the optimizer
            optimizer = OptimizerFactory.get_optimizer(optimizer_name=hparams.optimizer,
                                                       optimizer_params=hparams.optimizer_params,
                                                       loss_scaling=hparams.loss_scaling,
                                                       l2_regularization=hparams.l2_regularization,
                                                       batch_config=batch_config,
                                                       lr_scheduler=lr_scheduler,
                                                       wd_scheduler=wd_scheduler,
                                                       distributed_training=hparams.distributed_training,
                                                       norm_layer_params=hparams.norm_layer)

            # prepare loss and metrics
            loss_kwargs = {'name': 'loss'}
            if hparams.label_smoothing is None:
                loss_class = tf.keras.losses.SparseCategoricalCrossentropy
            else:
                loss_class = SmoothedCategoricalCrossentropy
                loss_kwargs = dict(num_classes=num_classes, label_smoothing=hparams.label_smoothing)
            loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

            # debug predictions and labels
            if False:
                pred_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
                label_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
                loss_class = wrap_loss_in_pred_enqueuer(loss_class, pred_outfeed_queue)
                loss_class = wrap_loss_in_label_enqueuer(loss_class, label_outfeed_queue)
                debug_outfeeds.append(('prediction', pred_outfeed_queue))
                debug_outfeeds.append(('label', label_outfeed_queue))

            accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy
            accuracy_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

            if hparams.synthetic_data != 'ipu':
                if hparams.accelerator_side_reduction:
                    loss_class = wrap_loss_in_allreduce_enqueuer(
                        loss_class, loss_outfeed_queue, num_replicas=hparams.num_replicas)
                    accuracy_class = wrap_metric_in_allreduce_enqueuer(
                        accuracy_class, accuracy_outfeed_queue, num_replicas=hparams.num_replicas)
                else:
                    loss_class = wrap_loss_in_enqueuer(loss_class, loss_outfeed_queue)
                    accuracy_class = wrap_metric_in_enqueuer(accuracy_class, accuracy_outfeed_queue)

            loss = loss_class(**loss_kwargs)
            accuracy = accuracy_class(dtype=tf.float32, name='training_accuracy')

            # Compile the model
            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=[accuracy],
                          steps_per_execution=micro_batches_per_execution // batch_config.num_replicas)

            model.build(input_shape=(batch_config.micro_batch_size, img_shape[0], img_shape[1], img_shape[2]))
            model.summary(print_fn=logging.info)  # can't print summary until fit() or build() are invoked

            if num_pipeline_stages > 1:
                list_splits = ModelFactory.evaluate_splits(model, num_pipeline_stages)
                if hparams.pipeline_splits != list_splits:
                    logging.info(f'Recommended splits = {list_splits}')

            logging.info(f'weight_updates_per_epoch = {hparams.weight_updates_per_epoch}')
            logging.info(f'micro_batches_per_epoch = {micro_batches_per_epoch}')
            logging.info(f'micro_batches_per_execution = {micro_batches_per_execution}')
            logging.info(f'steps_per_execution = {micro_batches_per_execution // batch_config.num_replicas}')
            logging.info(f'num_epochs {hparams.num_epochs}')

            if hparams.checkpoint_dir is None:
                if hparams.distributed_training:
                    time_now = hvd.broadcast(tf.convert_to_tensor(value=time(), dtype=tf.float32), 0)
                else:
                    time_now = time()
                date_now = datetime.fromtimestamp(time_now).strftime("%d_%m_%Y_%H:%M:%S.%f")[:-3]
                hparams.checkpoint_dir = os.path.join('/tmp', 'checkpoints_' + date_now)

            if hparams.ckpt_all_instances:
                hparams.checkpoint_dir = os.path.join(hparams.checkpoint_dir, f'rank{hvd.rank()}')

            callbacks = CallbackFactory.get_callbacks(wandb=hparams.wandb,
                                                      wandb_params=hparams.wandb_params,
                                                      log_period=micro_batches_per_log // batch_config.num_replicas,
                                                      images_per_execution=micro_batches_per_execution * batch_config.micro_batch_size,
                                                      model=model,
                                                      outfeed_queues=None if hparams.synthetic_data == 'ipu' else [('lr', lr_outfeed_queue),
                                                                                                                   ('loss',
                                                                                                                    loss_outfeed_queue),
                                                                                                                   ('training_accuracy', accuracy_outfeed_queue)],
                                                      checkpoints=hparams.checkpoints,
                                                      checkpoint_dir=hparams.checkpoint_dir,
                                                      distributed_training=hparams.distributed_training and not hparams.accelerator_side_reduction,
                                                      hyperparams=vars(hparams),
                                                      debug_outfeed_queues=[] if hparams.synthetic_data == 'ipu' else debug_outfeeds)

            # start timer
            time_to_train_timer.start()

            # Train the model
            model.fit(ds,
                      steps_per_epoch=micro_batches_per_epoch // popdist.getNumInstances(),
                      epochs=hparams.num_epochs,
                      callbacks=callbacks)

        if hparams.validation:
            if not hparams.distributed_training:
                cfg.auto_select_ipus = hparams.validation_num_replicas * hparams.validation_ipus_per_replica
            else:
                if hparams.validation_num_replicas != popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica():
                    logging.warning(f'Validation replication factor given to poprun '
                                    f'(=={popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica()}) '
                                    f'does not match the config (=={hparams.validation_num_replicas}). Poprun will override the config.')

                if hparams.validation_ipus_per_replica != popdist.getNumIpusPerReplica():
                    raise ValueError(f'The number of ipus per replica in validation does not match the value provided to poprun'
                                     f'({hparams.validation_ipus_per_replica} != {popdist.getNumIpusPerReplica()})')
                popdist.tensorflow.set_ipu_config(
                    cfg, ipus_per_replica=hparams.validation_ipus_per_replica, configure_device=True)
            cfg.floating_point_behaviour.esr = ipu.config.StochasticRoundingBehaviour.from_bool(False)
            cfg.configure_ipu_system()
            seed.set_host_seed(hparams.seed)
            seed.set_ipu_seed(hparams.seed)

            # swap the training preprocess layer with inference preprocess layer
            model = replace_preprocess_layer_with_fn(model, fn=accelerator_side_preprocess_inference_fn)

            if hparams.pipeline_validation_model:
                # Gradient_accumulation_count must be changed again.
                # Configure model is also invoked to make sure the new layer has a device assignment
                model = ModelFactory.configure_model(model=model,
                                                     gradient_accumulation_count=validation_batch_config.gradient_accumulation_count,
                                                     pipeline_splits=hparams.pipeline_splits,
                                                     device_mapping=hparams.device_mapping,
                                                     pipeline_schedule=hparams.pipeline_schedule,
                                                     available_memory_proportion=hparams.available_memory_proportion,
                                                     optimizer_state_offloading=hparams.optimizer_state_offloading)

            else:
                # map all pipeline stages to one ipu and set pipeline schedule to sequential
                model.set_pipelining_options(device_mapping=[0 for _ in range(
                    len(hparams.pipeline_splits) + 1)], pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

            # Evaluate the number of steps per epoch
            validation_micro_batches_per_epoch = validation_batch_config.get_num_micro_batches_per_epoch(ds_valid_size)
            logging.info(f'validation micro batches per epoch {validation_micro_batches_per_epoch}')
            logging.info(f'validation micro batch size {validation_batch_config.micro_batch_size}')
            logging.info(f'validation global batch size {validation_batch_config.global_batch_size}')
            logging.info(f'validation num replicas {validation_batch_config.num_replicas}')
            logging.info(f'validation dataset size {ds_valid_size}')

            if validation_micro_batches_per_epoch == 0:
                raise ValueError(f'For validation, the number of replicas has been multiplied '
                                 f'by {hparams.num_ipus_per_replica} and then the number of validation micro batches should be '
                                 f'a multiple of {batch_config.num_replicas * hparams.num_ipus_per_replica}.')

            validation_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy
            validation_accuracy_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            if hparams.accelerator_side_reduction:
                validation_accuracy_class = wrap_metric_in_allreduce_enqueuer(
                    validation_accuracy_class,
                    validation_accuracy_outfeed_queue,
                    validation_batch_config.num_replicas
                )
            else:
                validation_accuracy_class = wrap_metric_in_enqueuer(
                    validation_accuracy_class,
                    validation_accuracy_outfeed_queue
                )
            validation_accuracy = validation_accuracy_class(name='validation_accuracy', dtype=tf.float32)

            # recompile the model for the validation
            model.compile(metrics=[validation_accuracy],
                          steps_per_execution=validation_micro_batches_per_epoch // validation_batch_config.num_replicas)

            validation_callbacks = CallbackFactory.get_callbacks(wandb=hparams.wandb,
                                                                 wandb_params=hparams.wandb_params,
                                                                 log_period=validation_micro_batches_per_epoch // validation_batch_config.num_replicas,
                                                                 images_per_execution=validation_micro_batches_per_epoch * validation_batch_config.micro_batch_size,
                                                                 model=model,
                                                                 outfeed_queues=None if hparams.synthetic_data == 'ipu' else [
                                                                     ('validation_accuracy', validation_accuracy_outfeed_queue)],
                                                                 distributed_training=hparams.distributed_training and not hparams.accelerator_side_reduction,
                                                                 hyperparams=vars(hparams),
                                                                 fields_to_remove=['loss'])

            ckpt_list = []
            if hparams.checkpoint_dir is not None:
                ckpt_list = glob.glob(os.path.join(hparams.checkpoint_dir, '*.h5'))
                if len(ckpt_list) == 0:
                    logging.warn(f'The directory {hparams.checkpoint_dir} doesn\'t contain checkpoint (*.h5) files')
            if len(ckpt_list) != 0:
                logging.info(f'number of checkpoints {len(ckpt_list)}')
                for ckpt_file in ckpt_list:
                    logging.info(f'checkpoint file {ckpt_file}')
                    model.load_weights(ckpt_file)
                    model.evaluate(ds_valid, steps=validation_micro_batches_per_epoch //
                                   popdist.getNumInstances(), callbacks=validation_callbacks)
                if hparams.clean_dir:
                    shutil.rmtree(hparams.checkpoint_dir)

            else:
                logging.warn(
                    'No checkpoint is used to evaluate, so it will be the last training run or random if training is false')
                metrics = model.evaluate(ds_valid, steps=validation_micro_batches_per_epoch //
                                         popdist.getNumInstances(), callbacks=validation_callbacks)

        # we only care about the TTT value if we ran both training and validation
        if hparams.training and hparams.validation:
            # stop timer
            time_to_train_timer.stop()
            time_to_train.log_time_to_train(time_to_train_timer, log_to_wandb=hparams.wandb)
