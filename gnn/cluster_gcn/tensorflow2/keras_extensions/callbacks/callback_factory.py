# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging

import popdist.tensorflow

from keras_extensions.callbacks.allreduce_metrics_callback import AllReduceMetricsCallback
from keras_extensions.callbacks.checkpoint_callback import CheckpointCallback
from keras_extensions.callbacks.custom_wandb_callback import TrainingCustomWandbCallback, ValidationCustomWandbCallback
from keras_extensions.callbacks.statistics_callback import BatchStatisticsCallback
from keras_extensions.callbacks.logging_callback import TrainingLoggingCallback, ValidationLoggingCallback
from keras_extensions.callbacks.outfeed_queue_callback import OutFeedQueueCallback


class CallbackFactory:

    @staticmethod
    def get_callbacks(universal_run_name,
                      num_nodes_processed_per_execution,
                      real_over_padded_ratio,
                      total_num_epochs,
                      checkpoint_path,
                      config={},
                      executions_per_log=1,
                      executions_per_ckpt=0,
                      outfeed_queues=None,
                      distributed_training=False):

        callbacks = []
        log_period = executions_per_log

        logging.info("Creating callback to report batch statistics.")
        callbacks.append(BatchStatisticsCallback(num_nodes_processed_per_execution, real_over_padded_ratio, total_num_epochs))

        if outfeed_queues is not None:
            logging.info(f"Creating callbacks to read outfeed queues: {outfeed_queues}")
            for outfeed_queue in outfeed_queues:
                callbacks.append(OutFeedQueueCallback(outfeed_queue))

        # For distributed training, peform allreduce on metrics
        if distributed_training:
            callbacks.append(AllReduceMetricsCallback())

        if log_period > 0:
            logging.info("Creating callback for logging to terminal with"
                         f" a period of 1 log every {log_period} executions.")
            callbacks.append(TrainingLoggingCallback(log_period=log_period))
            callbacks.append(ValidationLoggingCallback(log_period=log_period))
            if config["wandb"] and popdist.getInstanceIndex() == 0:
                logging.info("Creating callback for logging to weights and biases.")
                callbacks.append(TrainingCustomWandbCallback(name=universal_run_name,
                                                             config=config))
                callbacks.append(ValidationCustomWandbCallback(name=universal_run_name,
                                                               config=config))
        if popdist.getInstanceIndex() == 0:
            logging.info("Creating callback for creating checkpoints. Checkpoints"
                         f" will be saved at a period of 1 every {executions_per_ckpt}"
                         f" executions to path: {checkpoint_path}")
            callbacks.append(CheckpointCallback(checkpoint_dir=checkpoint_path,
                                                executions_per_ckpt=executions_per_ckpt))

        return callbacks
