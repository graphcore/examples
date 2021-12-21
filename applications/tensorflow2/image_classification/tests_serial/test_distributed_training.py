# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
import subprocess
import os

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
import popdist
import popdist.tensorflow
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import popdist_strategy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from losses.loss_enqueuer import wrap_loss_in_enqueuer
from callbacks.logging_callback import LoggingCallback
from callbacks.outfeed_queue_callback import OutFeedQueueCallback
from callbacks.allreduce_metrics_callback import AllReduceMetricsCallback


class PopDistStrategyEquivalenceToIPUStrategy(unittest.TestCase):

    def call_program(self,
                     num_instances: int,
                     num_replicas: int,
                     gradient_accumulation: int,
                     test_all_reduce: bool = False):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        working_path = os.getcwd()

        cmd_prefix = [] if num_instances == 0 else ['poprun',
                                                    '--num-instances', f'{num_instances}',
                                                    '--num-replicas', f'{num_replicas}',
                                                    '--only-output-from-instance', '0']

        cmd_suffix = ['--num-replicas', f'{num_replicas}'] if num_instances == 0 else []
        cmd_suffix += ['--gradient-accumulation-count', f'{gradient_accumulation}']

        cmd = cmd_prefix + ['python3', 'test_distributed_training.py'] + cmd_suffix

        try:
            completed = subprocess.run(args=cmd,
                                       cwd=working_path,
                                       shell=False, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, check=True)
            decoded_output = str(completed.stdout, 'utf-8')
            if not test_all_reduce:
                output_last_line = decoded_output.splitlines()[-1]
                return float(output_last_line.split(':')[-1] if '<stdout>' in output_last_line else output_last_line)
            else:
                # parse to get the loss
                last_logging_callback_line = [
                    line for line in decoded_output.splitlines() if 'INFO:logging_callback' in line][-1]
                import re
                matches = re.findall(r'(\{[^{}]+\})', last_logging_callback_line)
                import json
                d = json.loads(matches[-1].replace("'", '"'))
                return d['average_loss']

        except subprocess.CalledProcessError as e:
            print(f"The following command failed: {cmd}\n"
                  f"Working path: {working_path}\n"
                  f"Output of failed command:\n{e.output}")
            return 0.0

    def test_allreduce_callback(self):
        loss_with_allreduce = self.call_program(num_instances=2, num_replicas=4,
                                                gradient_accumulation=1, test_all_reduce=True)
        loss_single_instance = self.call_program(
            num_instances=0, num_replicas=4, gradient_accumulation=1, test_all_reduce=True)
        assert loss_single_instance == loss_with_allreduce

    def test_ipu_strategy_replication(self):
        assert self.call_program(num_instances=0, num_replicas=4, gradient_accumulation=1) == -186.0

    def test_ipu_strategy_accumulation(self):
        assert self.call_program(num_instances=0, num_replicas=1, gradient_accumulation=4) == -186.0

    def test_popdist_strategy_multiple_instances(self):
        assert self.call_program(num_instances=2, num_replicas=4, gradient_accumulation=1) == -186.0

    def test_popdist_strategy_multiple_instances_and_accumulation(self):
        assert self.call_program(num_instances=2, num_replicas=4, gradient_accumulation=4) == -186.0


if __name__ == '__main__':

    def ipu_prog(num_replicas, gradient_accumulation):
        import logging
        import sys
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        popdist_on = popdist.isPopdistEnvSet()

        num_global_replicas = popdist.getNumTotalReplicas() if popdist_on else num_replicas
        num_instances = popdist.getNumInstances() if popdist_on else 1

        dataset_size = global_batch_size = 16
        micro_batch_size = int(global_batch_size / num_global_replicas / gradient_accumulation)

        X = np.arange(1, dataset_size + 1, 1, dtype=float)
        Y = [0] * dataset_size
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if popdist_on:
            ds = ds.shard(num_instances, index=popdist.getInstanceIndex())
        ds = ds.batch(micro_batch_size, drop_remainder=True)
        ds = ds.repeat()

        cfg = ipu.config.IPUConfig()
        if popdist_on:
            cfg = popdist.tensorflow.set_ipu_config(
                cfg, ipus_per_replica=popdist.getNumIpusPerReplica(), configure_device=True)
            hvd.init()
        else:
            cfg.auto_select_ipus = num_global_replicas
        cfg.configure_ipu_system()

        strategy = popdist_strategy.PopDistStrategy() if popdist_on else ipu.ipu_strategy.IPUStrategy()

        with strategy.scope():

            def get_model():
                input_layer = tf.keras.Input(shape=1)
                kernel_initializer = tf.keras.initializers.Constant(1)
                x = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=kernel_initializer)(input_layer)
                return tf.keras.Model(input_layer, x)

            model = get_model()
            model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=gradient_accumulation)
            model.build(input_shape=(micro_batch_size, 1))

            if popdist_on:
                def gradient_normalizer(grads_and_vars):
                    return [(grad / gradient_accumulation, var) for grad, var in grads_and_vars]
            else:
                def gradient_normalizer(grads_and_vars):
                    return [(grad / num_global_replicas / gradient_accumulation, var) for grad, var in grads_and_vars]

            optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, gradient_transformers=[gradient_normalizer])

            loss_class = tf.keras.losses.MeanSquaredError
            loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
            loss_class = wrap_loss_in_enqueuer(loss_class, loss_outfeed_queue)
            loss = loss_class()

            micro_batches_per_weight_update = num_global_replicas * gradient_accumulation
            steps_per_execution = dataset_size // (micro_batch_size *
                                                   micro_batches_per_weight_update) * micro_batches_per_weight_update

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=[tf.keras.losses.MSE],
                          steps_per_execution=steps_per_execution)

            callbacks = [OutFeedQueueCallback(queue=loss_outfeed_queue, name='average_loss')]
            if num_instances > 1:
                callbacks += [AllReduceMetricsCallback()]
            callbacks += [LoggingCallback(1)]

            model.fit(ds, steps_per_epoch=steps_per_execution, callbacks=callbacks)

            return model.get_weights()[0][0][0]

    import argparse

    def add_arguments(parser):
        parser.add_argument('--num-replicas', type=int, default=1)
        parser.add_argument('--gradient-accumulation-count', type=int, default=1)
        return parser

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_arguments(parser)
    args = parser.parse_args()
    print(args)

    print(ipu_prog(num_replicas=args.num_replicas,
                   gradient_accumulation=args.gradient_accumulation_count))
