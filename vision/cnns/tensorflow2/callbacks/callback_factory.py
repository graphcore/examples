# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.python import ipu

from .allreduce_metrics_callback import AllReduceMetricsCallback
from .checkpoint_callback import CheckpointCallback
from .compilation_time_callback import CompilationTimeCallback
from .cpu_memory_callback import CPUMemoryCallback
from .custom_wandb_callback import CustomWandbCallback
from .data_remover_callback import DataRemoverCallback
from .debug_callback import DebugCallback
from .logging_callback import LoggingCallback
from .outfeed_queue_callback import OutFeedQueueCallback
from .throughput_estimator_callback import ThroughputEstimatorCallback
from .correction_metric_callback import CorrectionMetricCallback
from .epoch_calculation_callback import EpochCalculationCallback
from .lr_logger_callback import LRLoggerCallback


class CallbackFactory:

    @staticmethod
    def get_callbacks(log_period: int,
                      images_per_execution: int,
                      model: tf.keras.Model,
                      micro_batches_per_epoch: int,
                      outfeed_queues: Optional[List[Tuple[str, ipu.ipu_outfeed_queue.IPUOutfeedQueue]]] = None,
                      debug_outfeed_queues: List[Tuple[str, ipu.ipu_outfeed_queue.IPUOutfeedQueue]] = [],
                      checkpoints: bool = False,
                      checkpoint_dir: str = '/tmp/checkpoints',
                      hyperparams: dict = {},
                      correct_metric: Optional[Tuple[str, float]] = None,
                      fields_to_remove: List[str] = []):

        callbacks = []

        callbacks.append(ThroughputEstimatorCallback(images_per_execution=images_per_execution))
        callbacks.append(CompilationTimeCallback(hyperparams.compile_only))
        callbacks.append(EpochCalculationCallback(micro_batches_per_epoch))
        callbacks.append(CPUMemoryCallback())
        callbacks.append(LRLoggerCallback())

        if outfeed_queues is not None:
            for name, outfeed_queue in outfeed_queues:
                callbacks.append(OutFeedQueueCallback(queue=outfeed_queue, name=name))

        if len(fields_to_remove) > 0:
            callbacks.append(DataRemoverCallback(fields_to_remove))

        if len(debug_outfeed_queues) > 0:
            for name, outfeed_queue in debug_outfeed_queues:
                callbacks.append(DebugCallback(queue=outfeed_queue, name=name))

        # For distributed validation peform all reduce on metrics
        if hyperparams.distributed_training and not hyperparams.accelerator_side_reduction:
            callbacks.append(AllReduceMetricsCallback())

        if checkpoints:
            callbacks.append(CheckpointCallback(ckpt_period=log_period,
                                                checkpoint_dir=checkpoint_dir))


        if correct_metric is not None:
            metric_name, correction_factor = correct_metric
            callbacks.append(CorrectionMetricCallback(metric_name, correction_factor))

        # Add log callbacks
        logging_log_period = log_period
        if logging_log_period > 0:
            callbacks.append(LoggingCallback(log_period=logging_log_period))

        if hyperparams.wandb:
            wandb_log_period = log_period
            if wandb_log_period > 0:
                callbacks.append(CustomWandbCallback(log_period=wandb_log_period,
                                                     hyperparams=vars(hyperparams),
                                                     model=model))

        return callbacks
