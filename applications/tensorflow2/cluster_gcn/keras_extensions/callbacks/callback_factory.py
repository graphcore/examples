# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging

from keras_extensions.callbacks.checkpoint_callback import CheckpointCallback
from keras_extensions.callbacks.custom_wandb_callback import CustomWandbCallback
from keras_extensions.callbacks.statistics_callback import BatchStatisticsCallback
from keras_extensions.callbacks.logging_callback import LoggingCallback
from keras_extensions.callbacks.outfeed_queue_callback import OutFeedQueueCallback


class CallbackFactory:

    @staticmethod
    def get_callbacks(universal_run_name,
                      num_nodes_processed_per_execution,
                      checkpoint_path,
                      config={},
                      executions_per_log=1,
                      executions_per_ckpt=0,
                      outfeed_queues=None):

        callbacks = []
        log_period = executions_per_log

        logging.info("Creating callback to report batch statistics.")
        callbacks.append(BatchStatisticsCallback(num_nodes_processed_per_execution))

        if outfeed_queues is not None:
            logging.info(f"Creating callbacks to read outfeed queues: {outfeed_queues}")
            for outfeed_queue in outfeed_queues:
                callbacks.append(OutFeedQueueCallback(outfeed_queue))

        if log_period > 0:
            logging.info("Creating callback for logging to terminal with"
                         f" a period of 1 log every {log_period} executions.")
            callbacks.append(LoggingCallback(log_period=log_period))
            if config["wandb"]:
                logging.info("Creating callback for logging to weights and biases.")
                callbacks.append(CustomWandbCallback(name=universal_run_name,
                                                     config=config))

        logging.info("Creating callback for creating checkpoints. Checkpoints"
                     f" will be saved at a period of 1 every {executions_per_ckpt}"
                     f" executions to path: {checkpoint_path}")
        callbacks.append(CheckpointCallback(checkpoint_dir=checkpoint_path,
                                            executions_per_ckpt=executions_per_ckpt))

        return callbacks
