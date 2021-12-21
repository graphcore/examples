# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import logging
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.ops import pipelining_ops
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod.popdist_strategy import PopDistStrategy
import precision
from data.dataset_factory import DatasetFactory
from batch_config import BatchConfig
from model.model_factory import ModelFactory, replace_preprocess_layer_with_fn
from callbacks.callback_factory import CallbackFactory
from eight_bit_transfer import EightBitTransfer
from configuration import file_argparse, terminal_argparse
from custom_exceptions import DimensionError, UnallowedConfigurationError
import warnings
from callbacks.callbacks_periodicity import calculate_log_period
from schedules.scheduler_builder import get_lr_scheduler
from losses.loss_enqueuer import wrap_loss_in_enqueuer
from losses.smoothed_categorical_crossentropy import SmoothedCategoricalCrossentropy
from metrics.metric_enqueuer import wrap_metric_in_enqueuer
import os
from datetime import datetime
import glob
import shutil
import time_to_train
from optimizers.optimizer_factory import OptimizerFactory
from seed import set_seed
import popdist
import popdist.tensorflow


if __name__ == '__main__':
    # configure logger
    logging.basicConfig(level=logging.INFO)

    # create an argument parser
    parser = argparse.ArgumentParser(description='TF2 classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = terminal_argparse.add_arguments(parser)
    args = parser.parse_args()
    args = file_argparse.parse_yaml_config(args, parser)
    logging.info(f'args = {args}')

    # parse args
    dataset_name = args.dataset
    dataset_path = args.dataset_path
    model_name = args.model_name
    micro_batch_size = args.micro_batch_size
    validation_micro_batch_size = args.validation_micro_batch_size or micro_batch_size
    num_epochs = args.num_epochs
    weight_updates_per_epoch = args.weight_updates_per_epoch
    num_replicas = args.num_replicas
    validation_num_replicas = args.validation_num_replicas
    gradient_accumulation_count = args.gradient_accumulation_count
    global_batch_size = args.global_batch_size
    precision_type = args.precision
    pipeline_splits = args.pipeline_splits
    device_mapping = args.device_mapping
    pipeline_schedule = args.pipeline_schedule
    validation = args.validation
    training = args.training
    log_to_wandb = args.wandb
    eight_bit_transfer = args.eight_bit_transfer
    half_partials = args.half_partials
    recomputation = args.recomputation
    accelerator_side_preprocess = args.accelerator_side_preprocess
    logs_per_epoch = args.logs_per_epoch
    available_memory_proportion = args.available_memory_proportion
    stochastic_rounding = args.stochastic_rounding
    loss_scaling = args.loss_scaling
    weight_decay = args.weight_decay
    l2_regularization = args.l2_regularization
    optimizer_state_offloading = args.optimizer_state_offloading
    stable_norm = args.stable_norm
    fp_exceptions = args.fp_exceptions
    lr_schedule = args.lr_schedule
    lr_schedule_params = args.lr_schedule_params
    lr_warmup_params = args.lr_warmup_params
    lr_staircase = args.lr_staircase
    on_demand = args.on_demand
    dbn_replica_group_size = args.dbn_replica_group_size
    bn_momentum = args.bn_momentum
    checkpoints = args.checkpoints
    clean_dir = args.clean_dir
    checkpoint_dir = args.checkpoint_dir
    label_smoothing = args.label_smoothing
    optimizer_name = args.optimizer
    optimizer_params = args.optimizer_params
    seed = args.seed
    internal_exchange_optimization_target = args.internal_exchange_optimization_target
    max_cross_replica_buffer_size = args.max_cross_replica_buffer_size
    max_reduce_many_buffer_size = args.max_reduce_many_buffer_size
    gather_conv_output = args.gather_conv_output
    pipeline_num_parallel = args.pipeline_num_parallel

    # check if the script has been called by poprun
    distributed_training = popdist.isPopdistEnvSet()

    if distributed_training:
        if num_replicas != popdist.getNumTotalReplicas():
            logging.warning(f'Replication factor given to poprun (=={popdist.getNumTotalReplicas()}) '
                            f'does not match the config (=={num_replicas}). Poprun will override the config.')
            num_replicas = popdist.getNumTotalReplicas()

        max_threads_per_instance = os.cpu_count() // popdist.getNumInstances()
        if pipeline_num_parallel > max_threads_per_instance:
            logging.warning(f'The number of chosen threads {pipeline_num_parallel} is bigger than the total number of physical threads '
                            f'divided by the number of instances,  Poprun will override the config. ')
            # Limit the maximal number of threads to the total of physical threads divided by the number of instances
            pipeline_num_parallel = max_threads_per_instance

        if popdist.getInstanceIndex() != 0:
            checkpoints = False
            log_to_wandb = False

    # when neither option is specified, assume gradient accumulation count 1
    if gradient_accumulation_count is None and global_batch_size is None:
        gradient_accumulation_count = 1

    if recomputation and not len(pipeline_splits):
        raise ValueError('Recomputation requires a pipelined model. '
                         'Make sure "--pipeline-splits" is defined')

    if logs_per_epoch < 0:
        raise ValueError(f'--logs-per-epoch should be non-negative (>=0), it is {logs_per_epoch}')

    # check for partial logs, example --logs-per-epoch 0.5 and --epochs 5
    if (logs_per_epoch > 0) and (logs_per_epoch < 1) and (num_epochs % (1 / logs_per_epoch) != 0):
        raise ValueError(f'It is not possible to log {1/logs_per_epoch} epochs a time for {num_epochs} epochs')

    num_pipeline_stages = len(pipeline_splits) + 1

    if device_mapping:
        if len(device_mapping) != num_pipeline_stages:
            raise DimensionError(
                f'The number of device assignments {len(device_mapping)} is not equal to the number of pipeline splits + 1: {num_pipeline_stages}.')

        if len(set(device_mapping)) != max(device_mapping) + 1:
            raise DimensionError(
                f'The model is pipelined over {len(set(device_mapping))} different IPUs, but one or more stages are being assigned to IPU {max(device_mapping) + 1}')

    if eight_bit_transfer and not accelerator_side_preprocess:
        raise UnallowedConfigurationError(
            f'When eight bit transfer is enabled the normalisation must be done on the device. '
            f'If you want to keep 8bit io, set --accelerator-side-preprocess to True.')

    if (eight_bit_transfer or accelerator_side_preprocess) and 'cifar' in model_name:
        raise UnallowedConfigurationError(
            f'Currently eight bit transfer and on accelerator preprocessing are not available for cifar-resnet model family. '
            f'Either change the model or disable both --accelerator-side-preprocess and --eight-bit-transfer.')

    if len(available_memory_proportion) > 1 and num_pipeline_stages == 1:
        raise UnallowedConfigurationError(
            'Setting available memory proportion per pipeline stage, '
            'but no pipeline stages defined. Please use --pipeline-splits to define the pipeline stages')

    if len(available_memory_proportion) > 1 and len(available_memory_proportion) != 2 * num_pipeline_stages:
        raise DimensionError(
            'Define a single global value of available memory proportion or two values per pipeline stage. '
            f'There are {num_pipeline_stages} pipeline stages defined and {len(available_memory_proportion)} values of '
            'available memory proportion')

    num_ipus_per_replica = num_pipeline_stages if not device_mapping else max(device_mapping) + 1

    if dbn_replica_group_size > 1 and num_ipus_per_replica != 1:
        raise ValueError('Distributed Batch Norm can only be applied when model fits on a single ipu.')

    if dbn_replica_group_size > 1 and num_replicas % dbn_replica_group_size != 0:
        raise ValueError('Distributed Batch Norm can only be applied when model is replicated, '
                         'and replication factor is divisible by dbn-replica-group-size.')

    # configure IPU for training
    cfg = ipu.config.IPUConfig()
    cfg.allow_recompute = recomputation
    cfg.optimizations.merge_infeed_io_copies = True
    cfg.optimizations.maximum_cross_replica_sum_buffer_size = max_cross_replica_buffer_size
    cfg.optimizations.maximum_reduce_many_buffer_size = max_reduce_many_buffer_size
    cfg.floating_point_behaviour.esr = stochastic_rounding
    cfg.norms.use_stable_statistics = stable_norm
    cfg.norms.experimental.distributed_batch_norm_replica_group_size = dbn_replica_group_size
    if on_demand:
        cfg.device_connection.enable_remote_buffers = True
        cfg.device_connection.type = ipu.utils.DeviceConnectionType.ON_DEMAND
    if half_partials:
        cfg.convolutions.poplar_options['partialsType'] = 'half'
        cfg.matmuls.poplar_options['partialsType'] = 'half'
    if len(available_memory_proportion) == 1:
        cfg.matmuls.poplar_options['availableMemoryProportion'] = str(available_memory_proportion[0] / 100)
        cfg.convolutions.poplar_options['availableMemoryProportion'] = str(available_memory_proportion[0] / 100)
    if gather_conv_output:
        cfg.convolutions.poplar_options['gatherConvOutput'] = 'true'
    cfg.floating_point_behaviour.inv = fp_exceptions
    cfg.floating_point_behaviour.div0 = fp_exceptions
    cfg.floating_point_behaviour.oflo = fp_exceptions
    cfg.compilation_poplar_options['target.deterministicWorkers'] = 'false' if seed is None else 'portable'
    if internal_exchange_optimization_target is not None:
        cfg.compilation_poplar_options['opt.internalExchangeOptimisationTarget'] = internal_exchange_optimization_target

    if distributed_training:
        popdist.tensorflow.set_ipu_config(cfg, ipus_per_replica=num_ipus_per_replica, configure_device=True)
        hvd.init()
    else:
        cfg.auto_select_ipus = num_ipus_per_replica * num_replicas

    cfg.configure_ipu_system()

    set_seed(seed)

    batch_config = BatchConfig(micro_batch_size,
                               num_replicas,
                               gradient_accumulation_count,
                               global_batch_size)

    logging.info(f'micro batch size {batch_config.micro_batch_size}')
    logging.info(f'global batch size {batch_config.global_batch_size}')
    logging.info(f'gradient accumulation {batch_config.gradient_accumulation_count}')
    logging.info(f'num replicas {batch_config.num_replicas}')

    if validation:

        validation_num_replicas = validation_num_replicas or (num_replicas * num_ipus_per_replica)
        validation_batch_config = BatchConfig(micro_batch_size=validation_micro_batch_size,
                                              num_replicas=validation_num_replicas,
                                              gradient_accumulation_count=1,
                                              global_batch_size=None)

    fp_precision = precision.Precision(precision_type)
    fp_precision.apply()

    # get eight bit transfer object
    eight_bit_transfer = EightBitTransfer(fp_precision.compute_precision) if eight_bit_transfer else None

    # Get the training dataset
    ds, img_shape, dataset_size, num_classes, accelerator_side_preprocess_train_fn = DatasetFactory.get_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        split='train',
        img_datatype=fp_precision.compute_precision,
        micro_batch_size=batch_config.micro_batch_size,
        shuffle=True,
        accelerator_side_preprocess=accelerator_side_preprocess,
        eight_bit_transfer=eight_bit_transfer,
        pipeline_num_parallel=pipeline_num_parallel,
        seed=seed)
    logging.debug(ds)

    # Get the validation dataset
    if validation:
        ds_valid, _, ds_valid_size, _, accelerator_side_preprocess_inference_fn = DatasetFactory.get_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split='test',
            img_datatype=fp_precision.compute_precision,
            micro_batch_size=batch_config.micro_batch_size,
            accelerator_side_preprocess=accelerator_side_preprocess,
            eight_bit_transfer=eight_bit_transfer,
            seed=seed)
        logging.debug(ds_valid)

    if weight_updates_per_epoch == -1:
        weight_updates_per_epoch = dataset_size // batch_config.global_batch_size
    micro_batches_per_epoch = weight_updates_per_epoch * batch_config.num_micro_batches_per_weight_update

    micro_batches_per_log = calculate_log_period(weight_updates_per_epoch, num_epochs, logs_per_epoch, batch_config)
    logging.info(f'micro batches per log {micro_batches_per_log}')

    # steps_per_execution is the number of weight updates in term of micro batches before going back to the host
    if micro_batches_per_log != 0:
        steps_per_execution = micro_batches_per_log
    else:
        # run training run in a single call
        logging.warn('The entire training run will be executed in a single call to the device.')
        steps_per_execution = micro_batches_per_epoch * num_epochs

    # if we do more than one epoch per device call we need to adjust the number of epochs
    # and the number of micro batches processed in an epoch
    if micro_batches_per_epoch < steps_per_execution:
        total_num_micro_batches = micro_batches_per_epoch * num_epochs
        num_epochs = int(total_num_micro_batches / steps_per_execution)
        micro_batches_per_epoch = steps_per_execution

    if (steps_per_execution > micro_batches_per_epoch):
        warnings.warn(
            f'steps_per_execution = {steps_per_execution} > micro_batches_per_epoch = {micro_batches_per_epoch}')
        warnings.warn(
            f'This is not possible as micro_batches_per_epoch is a series of steps_per_execution in term of micro_batches')
        warnings.warn(f'You might consider changing the number of micro_batches and / or weight_updates_per_execution')
        steps_per_execution = micro_batches_per_epoch

    # micro_batches_per_epoch is the number of running micro batches per epoch which can be larger or smaller
    # than the actual number of steps per epoch ( = number of micro batches per epoch covering the whole dataset)
    if micro_batches_per_epoch % steps_per_execution:
        raise ValueError(
            f'steps_per_execution {steps_per_execution} should divide micro_batches_per_epoch = {micro_batches_per_epoch}')

    time_to_train_timer = time_to_train.TimeToTrain()

    # Create an IPU distribution strategy
    train_strategy = PopDistStrategy() if distributed_training else ipu.ipu_strategy.IPUStrategy()

    with train_strategy.scope():

        # Create an instance of the model
        model = ModelFactory.create_model(model_name=model_name,
                                          input_shape=img_shape,
                                          classes=num_classes,
                                          accelerator_side_preprocessing_fn=accelerator_side_preprocess_train_fn,
                                          eight_bit_transfer=eight_bit_transfer)

        model = ModelFactory.configure_model(model=model, gradient_accumulation_count=batch_config.gradient_accumulation_count,
                                             pipeline_splits=pipeline_splits, device_mapping=device_mapping, pipeline_schedule=pipeline_schedule,
                                             available_memory_proportion=available_memory_proportion, optimizer_state_offloading=optimizer_state_offloading)

        if training:
            # prepare the learning rate scheduler
            lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
                outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)
            lr_scheduler = get_lr_scheduler(
                scheduler_name=lr_schedule,
                schedule_params=lr_schedule_params,
                warmup_params=lr_warmup_params,
                global_batch_size=batch_config.global_batch_size,
                weight_updates_per_epoch=weight_updates_per_epoch,
                staircase=lr_staircase,
                queue=lr_outfeed_queue
            )

            # get weight decay scheduler
            wd_scheduler = get_lr_scheduler(
                scheduler_name=lr_schedule,
                schedule_params=lr_schedule_params,
                warmup_params=lr_warmup_params,
                global_batch_size=batch_config.global_batch_size,
                weight_updates_per_epoch=weight_updates_per_epoch,
                staircase=lr_staircase,
                queue=None,
                factor=weight_decay
            )

            # prepare the optimizer
            optimizer = OptimizerFactory.get_optimizer(optimizer_name=optimizer_name,
                                                       optimizer_params=optimizer_params,
                                                       loss_scaling=loss_scaling,
                                                       l2_regularization=l2_regularization,
                                                       bn_momentum=bn_momentum,
                                                       batch_config=batch_config,
                                                       lr_scheduler=lr_scheduler,
                                                       wd_scheduler=wd_scheduler,
                                                       distributed_training=distributed_training)

            loss_kwargs = {'name': 'loss'}
            if label_smoothing is None:
                loss_class = tf.keras.losses.SparseCategoricalCrossentropy
            else:
                loss_class = SmoothedCategoricalCrossentropy
                loss_kwargs = dict(num_classes=num_classes, label_smoothing=label_smoothing)
            loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            loss_class = wrap_loss_in_enqueuer(loss_class, loss_outfeed_queue)

            loss = loss_class(**loss_kwargs)

            accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy
            accuracy_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            accuracy_class = wrap_metric_in_enqueuer(accuracy_class, accuracy_outfeed_queue)
            accuracy = accuracy_class(dtype=tf.float32, name='training_accuracy')

            # Compile the model
            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=[accuracy],
                          steps_per_execution=steps_per_execution)

            model.build(input_shape=(batch_config.micro_batch_size, img_shape[0], img_shape[1], img_shape[2]))
            model.summary(print_fn=logging.info)  # can't print summary until fit() or build() are invoked
            if num_pipeline_stages > 1:
                list_splits = ModelFactory.evaluate_splits(model, num_pipeline_stages)
                if pipeline_splits != list_splits:
                    logging.info(f'Recommended splits = {list_splits}')

            logging.info(f'weight_updates_per_epoch = {weight_updates_per_epoch}')
            logging.info(f'micro_batches_per_epoch = {micro_batches_per_epoch}')
            logging.info(f'steps_per_execution = {steps_per_execution}')
            logging.info(f'num_epochs {num_epochs}')

            if checkpoints:
                if checkpoint_dir is None:
                    now = datetime.now()
                    root_path = '/tmp'
                    checkpoint_dir = os.path.join(root_path, 'checkpoints_' + now.strftime("%d_%m_%Y_%H:%M:%S.%f")[:-3])
                logging.info(
                    f'Checkpoint directory = {checkpoint_dir}, will this directory will be cleaned ? {clean_dir}')

            callbacks = CallbackFactory.get_callbacks(wandb=log_to_wandb,
                                                      log_period=micro_batches_per_log,
                                                      steps_per_execution=steps_per_execution,
                                                      micro_batch_size=batch_config.micro_batch_size,
                                                      model=model,
                                                      outfeed_queues=[('lr', lr_outfeed_queue),
                                                                      ('loss', loss_outfeed_queue),
                                                                      ('training_accuracy', accuracy_outfeed_queue)],
                                                      checkpoints=checkpoints,
                                                      checkpoint_dir=checkpoint_dir,
                                                      distributed_training=distributed_training,
                                                      args=vars(args))

            # start timer
            time_to_train_timer.start()

            # Train the model
            model.fit(ds, steps_per_epoch=micro_batches_per_epoch, epochs=num_epochs, callbacks=callbacks)

        if validation:
            if not distributed_training:
                cfg.auto_select_ipus = validation_num_replicas
                cfg.configure_ipu_system()
            else:
                if validation_num_replicas != popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica():
                    logging.warning(f'Validation replication factor given to poprun '
                                    f'(=={popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica()}) '
                                    f'does not match the config (=={validation_num_replicas}). Poprun will override the config.')

            set_seed(seed)

            # swap the training preprocess layer with inference preprocess layer
            model = replace_preprocess_layer_with_fn(model, fn=accelerator_side_preprocess_inference_fn)

            # map all pipeline stages to one ipu and set pipeline schedule to sequential
            model.set_pipelining_options(device_mapping=[0 for _ in range(
                len(pipeline_splits) + 1)], pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential)

            # Evaluate the number of steps per epoch
            validation_steps_per_epoch = validation_batch_config.get_num_micro_batches_per_epoch(ds_valid_size)
            logging.info(f'validation steps per epoch {validation_steps_per_epoch}')
            logging.info(f'validation micro batch size {validation_batch_config.micro_batch_size}')
            logging.info(f'validation global batch size {validation_batch_config.global_batch_size}')
            logging.info(f'validation num replicas {validation_batch_config.num_replicas}')
            logging.info(f'validation dataset size {ds_valid_size}')

            if validation_steps_per_epoch == 0:
                raise ValueError(f'For validation, the number of replicas has been multiplied '
                                 f'by {num_ipus_per_replica} and then the number of validation steps should be '
                                 f'a multiple of {batch_config.num_replicas * num_ipus_per_replica}.')

            validation_accuracy_class = tf.keras.metrics.SparseCategoricalAccuracy
            validation_accuracy_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            validation_accuracy_class = wrap_metric_in_enqueuer(
                validation_accuracy_class, validation_accuracy_outfeed_queue)
            validation_accuracy = validation_accuracy_class(name='validation_accuracy', dtype=tf.float32)

            # recompile the model for the validation
            model.compile(metrics=[validation_accuracy],
                          steps_per_execution=validation_steps_per_epoch)

            validation_callbacks = CallbackFactory.get_callbacks(wandb=log_to_wandb,
                                                                 log_period=validation_steps_per_epoch,
                                                                 steps_per_execution=validation_steps_per_epoch,
                                                                 micro_batch_size=validation_batch_config.micro_batch_size,
                                                                 model=model,
                                                                 outfeed_queues=[
                                                                     ('validation_accuracy', validation_accuracy_outfeed_queue)],
                                                                 distributed_training=distributed_training,
                                                                 args=vars(args),
                                                                 fields_to_remove=['loss'])

            if checkpoint_dir is not None:
                ckpt_list = glob.glob(os.path.join(checkpoint_dir, '*.h5'))
                if len(ckpt_list) == 0:
                    raise FileNotFoundError(f'The directory {checkpoint_dir} doesn\'t contain checkpoint (*.h5) files')
                logging.info(f'number of checkpoints {len(ckpt_list)}')
                for ckpt_file in ckpt_list:
                    logging.info(f'checkpoint file {ckpt_file}')
                    model.load_weights(ckpt_file)
                    model.evaluate(ds_valid, steps=validation_steps_per_epoch, callbacks=validation_callbacks)
                if clean_dir:
                    shutil.rmtree(checkpoint_dir)

            else:
                logging.warn(
                    'No checkpoint is used to evaluate, so it will be the last training run or random if training is false')
                metrics = model.evaluate(ds_valid, steps=validation_steps_per_epoch, callbacks=validation_callbacks)

        # we only care about the TTT value if we ran both training and validation
        if training and validation:
            # stop timer
            time_to_train_timer.stop()
            time_to_train.log_time_to_train(time_to_train_timer, log_to_wandb=log_to_wandb)
