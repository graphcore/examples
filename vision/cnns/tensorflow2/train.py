# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import glob
import logging
import os
import shutil
from copy import deepcopy
from datetime import datetime
from time import time

import horovod.tensorflow as hvd
import popdist
import popdist.tensorflow
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import distributed
from tensorflow.python.ipu.distributed.popdist_strategy import PopDistStrategy
from tensorflow.python.ipu.ops import pipelining_ops

import precision
import seed
import time_to_train
from batch_config import BatchConfig
from callbacks.callback_factory import CallbackFactory
from configuration import terminal_argparse
from datasets.dataset_factory import DatasetFactory
from eight_bit_transfer import EightBitTransfer
from ipu_config import configure_ipu, reconfigure_for_validation, select_ipus
from losses.loss_enqueuer import (wrap_loss_in_allreduce_enqueuer,
                                  wrap_loss_in_enqueuer,
                                  wrap_loss_in_label_enqueuer,
                                  wrap_loss_in_pred_enqueuer)
from losses.smoothed_categorical_crossentropy import \
    SmoothedCategoricalCrossentropy
from metrics.metric_enqueuer import (wrap_metric_in_allreduce_enqueuer,
                                     wrap_metric_in_enqueuer)
from model.model_factory import ModelFactory
from optimizers.optimizer_factory import OptimizerFactory
from program_steps import calculate_program_steps
from schedules.scheduler_factory import get_lr_scheduler


if __name__ == '__main__':
    # configure logger
    logging.basicConfig(level=logging.INFO)

    hparams = terminal_argparse.handle_cmdline_arguments()
    hparams.seed = seed.set_host_seed(hparams.seed, hparams.deterministic)

    # Create an IPU distribution strategy
    ipu_strategy = PopDistStrategy() if hparams.distributed_training else ipu.ipu_strategy.IPUStrategy()

    batch_config = BatchConfig(hparams.micro_batch_size,
                               hparams.num_replicas,
                               hparams.gradient_accumulation_count,
                               hparams.global_batch_size)

    hparams.gradient_accumulation_count = batch_config.gradient_accumulation_count
    hparams.global_batch_size = batch_config.global_batch_size
    fp_precision = precision.Precision(hparams.precision)
    fp_precision.apply()

    # get eight bit transfer object
    eight_bit_transfer = EightBitTransfer(fp_precision.compute_precision) if hparams.eight_bit_transfer else None

    cfg = configure_ipu(hparams)

    # prepare for training
    if hparams.training:

        time_to_train_timer = time_to_train.TimeToTrain()

        # select ipus before defining the training model
        cfg = select_ipus(hparams.distributed_training,
                          hparams.num_replicas,
                          hparams.num_ipus_per_replica,
                          cfg=cfg)
        seed.set_ipu_seed(hparams.seed)

        # Get the training dataset
        train_app_dataset, accelerator_side_preprocess_train_fn, hparams.pipeline_num_parallel = DatasetFactory.get_dataset(
            dataset_name=hparams.dataset,
            dataset_path=hparams.dataset_path,
            split='train',
            img_datatype=fp_precision.compute_precision,
            batch_config=batch_config,
            seed=hparams.seed,
            shuffle=hparams.shuffle,
            deterministic=hparams.deterministic,
            accelerator_side_preprocess=hparams.accelerator_side_preprocess,
            eight_bit_transfer=eight_bit_transfer,
            pipeline_num_parallel=hparams.pipeline_num_parallel,
            fused_preprocessing=hparams.fused_preprocessing,
            synthetic_data=hparams.synthetic_data)
        logging.debug(train_app_dataset.pipeline)

        (micro_batches_per_epoch,
         micro_batches_per_execution,
         micro_batches_per_log,
         micro_batches_per_ckpt) = calculate_program_steps(hparams, batch_config, train_app_dataset.size)

        with ipu_strategy.scope():
            # Create an instance of the model
            model = ModelFactory.create_model(model_name=hparams.model_name,
                                              input_shape=train_app_dataset.image_shape,
                                              classes=train_app_dataset.num_classes,
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

            model.set_infeed_queue_options(prefetch_depth=hparams.prefetch_depth)

            # prepare the learning rate scheduler
            lr_scheduler = get_lr_scheduler(
                scheduler_name=hparams.lr_schedule,
                schedule_params=hparams.lr_schedule_params,
                warmup_params=hparams.lr_warmup_params,
                global_batch_size=batch_config.global_batch_size,
                weight_updates_per_epoch=hparams.weight_updates_per_epoch,
                staircase=hparams.lr_staircase
            )

            # get weight decay scheduler
            wd_scheduler = get_lr_scheduler(
                scheduler_name=hparams.lr_schedule,
                schedule_params=hparams.lr_schedule_params,
                warmup_params=hparams.lr_warmup_params,
                global_batch_size=batch_config.global_batch_size,
                weight_updates_per_epoch=hparams.weight_updates_per_epoch,
                staircase=hparams.lr_staircase,
                factor=hparams.weight_decay
            )

            # prepare the optimizer
            optimizer = OptimizerFactory.get_optimizer(
                optimizer_name=hparams.optimizer,
                optimizer_params=hparams.optimizer_params,
                loss_scaling=hparams.loss_scaling,
                auto_loss_scaling=hparams.auto_loss_scaling,
                l2_regularization=hparams.l2_regularization,
                batch_config=batch_config,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                distributed_training=hparams.distributed_training,
                norm_layer_params=hparams.norm_layer,
                model=model
            )

            # prepare loss and metrics
            loss_kwargs = {'name': 'loss'}
            if hparams.label_smoothing is None:
                loss_class = tf.keras.losses.SparseCategoricalCrossentropy
            else:
                loss_class = SmoothedCategoricalCrossentropy
                loss_kwargs = dict(num_classes=train_app_dataset.num_classes, label_smoothing=hparams.label_smoothing)
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

            model.build(input_shape=(batch_config.micro_batch_size,
                                     train_app_dataset.image_shape[0],
                                     train_app_dataset.image_shape[1],
                                     train_app_dataset.image_shape[2]))
            model.summary(print_fn=logging.info)  # can't print summary until fit() or build() are invoked

            if hparams.num_pipeline_stages > 1:
                list_splits = ModelFactory.evaluate_splits(model, hparams.num_pipeline_stages)
                if hparams.pipeline_splits != list_splits:
                    logging.info(f'Recommended splits = {list_splits}')

            logging.info(f'weight_updates_per_epoch = {hparams.weight_updates_per_epoch}')
            logging.info(f'micro_batches_per_epoch = {micro_batches_per_epoch}')
            logging.info(f'micro_batches_per_execution = {micro_batches_per_execution}')
            logging.info(f'steps_per_execution = {micro_batches_per_execution // batch_config.num_replicas}')
            logging.info(f'num_epochs {hparams.num_epochs}')

            if hparams.checkpoint_output_dir is None:
                if hparams.distributed_training:
                    time_now = distributed.broadcast(tf.convert_to_tensor(value=time(), dtype=tf.float32), 0).numpy()
                else:
                    time_now = time()
                date_now = datetime.fromtimestamp(time_now).strftime("%d_%m_%Y_%H:%M:%S.%f")[:-3]
                hparams.checkpoint_output_dir = os.path.join('/tmp', 'checkpoints_' + date_now)

            ckpt_period = micro_batches_per_ckpt // batch_config.num_replicas
            if hparams.distributed_training:
                if hparams.ckpt_all_instances:
                    hparams.checkpoint_output_dir = os.path.join(hparams.checkpoint_output_dir, f'rank{hvd.rank()}')
                elif hvd.local_rank() != 0:
                    ckpt_period = 0

            # organize the outfeed queues
            debug_outfeed_queues = [] if hparams.synthetic_data == 'ipu' else debug_outfeeds
            outfeed_queues = None if hparams.synthetic_data == 'ipu' else [('loss', loss_outfeed_queue),
                                                                           ('training_accuracy', accuracy_outfeed_queue)]

            callbacks = CallbackFactory.get_callbacks(
                model=model,
                hyperparams=hparams,
                checkpoint_period=ckpt_period,
                checkpoint_phase=hparams.first_ckpt_epoch * micro_batches_per_epoch // batch_config.num_replicas,
                checkpoint_dir=hparams.checkpoint_output_dir,
                log_period=micro_batches_per_log // batch_config.num_replicas,
                images_per_execution=micro_batches_per_execution * batch_config.micro_batch_size,
                micro_batches_per_epoch=micro_batches_per_epoch // batch_config.num_replicas,
                debug_outfeed_queues=debug_outfeed_queues,
                outfeed_queues=outfeed_queues
            )

    # prepare for validation
    if hparams.validation:
        # select ipus before defining the validation model
        cfg = select_ipus(hparams.distributed_training,
                          hparams.validation_num_replicas,
                          hparams.validation_ipus_per_replica,
                          cfg=cfg)
        seed.set_ipu_seed(hparams.seed)

        validation_batch_config = BatchConfig(micro_batch_size=hparams.validation_micro_batch_size,
                                              num_replicas=hparams.validation_num_replicas,
                                              gradient_accumulation_count=hparams.validation_gradient_accumulation_count,
                                              global_batch_size=None)

        # Get the validation dataset
        validation_app_dataset, accelerator_side_preprocess_inference_fn, hparams.pipeline_num_parallel = DatasetFactory.get_dataset(
            dataset_name=hparams.dataset,
            dataset_path=hparams.dataset_path,
            split='test',
            img_datatype=fp_precision.compute_precision,
            batch_config=validation_batch_config,
            seed=hparams.seed,
            deterministic=hparams.deterministic,
            accelerator_side_preprocess=hparams.accelerator_side_preprocess,
            eight_bit_transfer=eight_bit_transfer,
            pipeline_num_parallel=hparams.pipeline_num_parallel,
            fused_preprocessing=hparams.fused_preprocessing,
            synthetic_data=hparams.synthetic_data)
        logging.debug(validation_app_dataset.pipeline)

        with ipu_strategy.scope():
            # Create an instance of the model
            validation_model = ModelFactory.create_model(model_name=hparams.model_name,
                                                         input_shape=validation_app_dataset.image_shape,
                                                         classes=validation_app_dataset.num_classes,
                                                         accelerator_side_preprocessing_fn=accelerator_side_preprocess_inference_fn,
                                                         eight_bit_transfer=eight_bit_transfer,
                                                         norm_layer_params=hparams.norm_layer)

            if hparams.pipeline_validation_model:
                # Gradient_accumulation_count must be changed again.
                # Configure model is also invoked to make sure the new layer has a device assignment
                validation_model = ModelFactory.configure_model(
                    model=validation_model,
                    gradient_accumulation_count=validation_batch_config.gradient_accumulation_count,
                    pipeline_splits=hparams.pipeline_splits,
                    device_mapping=hparams.device_mapping,
                    pipeline_schedule=hparams.pipeline_schedule,
                    available_memory_proportion=hparams.available_memory_proportion,
                    optimizer_state_offloading=hparams.optimizer_state_offloading
                )

            else:
                # map all pipeline stages to one ipu and set pipeline schedule to sequential
                validation_model.set_pipelining_options(
                    device_mapping=[0 for _ in range(len(hparams.pipeline_splits) + 1)],
                    pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential
                )

            validation_model.set_infeed_queue_options(prefetch_depth=hparams.prefetch_depth)

            validation_dataset_size = validation_app_dataset.padded_size
            accuracy_metric_name = 'validation_accuracy'
            correct_accuracy_metric = None

            if validation_app_dataset.padded_size != validation_app_dataset.size:
                logging.info(f'padded dataset size {validation_app_dataset.padded_size}')
                logging.info(f'original dataset size {validation_app_dataset.size}')
                correct_accuracy_metric = (accuracy_metric_name, validation_app_dataset.padded_size / validation_app_dataset.size)
                logging.info(f'correction factor {correct_accuracy_metric[1]}')

            # Evaluate the number of steps per epoch
            validation_micro_batches_per_epoch = validation_batch_config.get_num_micro_batches_per_epoch(
                validation_dataset_size)

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
            validation_accuracy = validation_accuracy_class(name=accuracy_metric_name, dtype=tf.float32)

            # recompile the model for the validation
            validation_model.compile(
                metrics=[validation_accuracy],
                steps_per_execution=validation_micro_batches_per_epoch // validation_batch_config.num_replicas
            )

            validation_outfeed_queues = None if hparams.synthetic_data == 'ipu' else [
                (accuracy_metric_name, validation_accuracy_outfeed_queue)]

            validation_callbacks = CallbackFactory.get_callbacks(
                model=validation_model,
                hyperparams=hparams,
                checkpoint_dir=hparams.checkpoint_input_dir,
                log_period=validation_micro_batches_per_epoch // validation_batch_config.num_replicas,
                images_per_execution=validation_micro_batches_per_epoch * validation_batch_config.micro_batch_size,
                micro_batches_per_epoch=validation_micro_batches_per_epoch * hparams.ckpts_per_epoch / validation_batch_config.num_replicas,
                outfeed_queues=validation_outfeed_queues,
                correct_metric=correct_accuracy_metric,
                fields_to_remove=['loss']
            )

    # run training
    if hparams.training:
        cfg = select_ipus(hparams.distributed_training,
                          hparams.num_replicas,
                          hparams.num_ipus_per_replica,
                          cfg=cfg)

        with ipu_strategy.scope():
            # When mlperf_logging is enabled loading the executable is excluded from TTT
            if hparams.mlperf_logging:
                # copy the model and optimiser weights
                optimizer._create_all_weights(model.trainable_variables)
                init_model_params = deepcopy(model.get_weights())
                init_optimiser_params = deepcopy(optimizer.get_weights())

                logging.info('Loading training binary into IPU')
                model.fit(train_app_dataset.pipeline,
                          steps_per_epoch=micro_batches_per_epoch // popdist.getNumInstances(),
                          epochs=1)

                # reset the model and optimiser weights
                model.set_weights(init_model_params)
                optimizer.set_weights(init_optimiser_params)

            # start timer
            time_to_train_timer.start()

            # Train the model
            logging.info('Starting training')
            model.fit(train_app_dataset.pipeline,
                      steps_per_epoch=micro_batches_per_epoch // popdist.getNumInstances(),
                      epochs=hparams.num_epochs,
                      callbacks=callbacks)

    # run validation
    if hparams.validation:
        cfg = reconfigure_for_validation(cfg)
        cfg = select_ipus(hparams.distributed_training,
                          hparams.validation_num_replicas,
                          hparams.validation_ipus_per_replica,
                          cfg=cfg)
        hparams.seed = seed.set_host_seed(hparams.seed)
        seed.set_ipu_seed(hparams.seed)

        logging.info(f'validation micro batches per epoch {validation_micro_batches_per_epoch}')
        logging.info(f'validation micro batch size {validation_batch_config.micro_batch_size}')
        logging.info(f'validation global batch size {validation_batch_config.global_batch_size}')
        logging.info(f'validation num replicas {validation_batch_config.num_replicas}')
        logging.info(f'validation dataset size {validation_dataset_size}')

        with ipu_strategy.scope():
            ckpt_list = []
            if hparams.checkpoint_input_dir is not None:
                ckpt_list = glob.glob(os.path.join(hparams.checkpoint_input_dir, '*.h5'))
                if len(ckpt_list) == 0:
                    logging.warn(f'The directory {hparams.checkpoint_input_dir} doesn\'t contain checkpoint (*.h5) files')
            validation_callbacks = CallbackFactory.set_validation_only_callbacks(
                callbacks=validation_callbacks,
                ckpt_list=ckpt_list,
                sweep=hparams.sweep,
                target_field=validation_accuracy.name,
                target_value=hparams.target_accuracy)
            if len(ckpt_list) != 0:
                logging.info(f'number of checkpoints {len(ckpt_list)}')
                for ckpt_file in ckpt_list:
                    logging.info(f'checkpoint file {ckpt_file}')
                    validation_model.load_weights(ckpt_file)
                    validation_model.evaluate(
                        validation_app_dataset.pipeline,
                        steps=validation_micro_batches_per_epoch // popdist.getNumInstances(),
                        callbacks=validation_callbacks
                    )
                if hparams.clean_dir:
                    shutil.rmtree(hparams.checkpoint_output_dir)

            else:
                if hparams.training:
                    logging.warn('No checkpoints to evaluate, evaluating final training weights.')
                    validation_model.set_weights(model.get_weights())
                else:
                    logging.warn('No checkpoints to evaluate, evaluating random weights.')
                metrics = validation_model.evaluate(
                    validation_app_dataset.pipeline,
                    steps=validation_micro_batches_per_epoch // popdist.getNumInstances(),
                    callbacks=validation_callbacks
                )

            if hparams.sweep and hparams.wandb:
                CallbackFactory.log_optimization_metric(validation_callbacks)

    # we only care about the TTT value if we ran both training and validation
    if hparams.training and hparams.validation:
        # stop timer
        time_to_train_timer.stop()
        time_to_train.log_time_to_train(time_to_train_timer, log_to_wandb=hparams.wandb)
