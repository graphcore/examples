# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
import subprocess
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
import popdist
import popdist.tensorflow
from tensorflow.python.ipu import distributed
from tensorflow.python.ipu.distributed import popdist_strategy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from losses.loss_enqueuer import wrap_loss_in_enqueuer, wrap_loss_in_allreduce_enqueuer
from metrics.metric_enqueuer import wrap_metric_in_enqueuer, wrap_metric_in_allreduce_enqueuer
from callbacks.logging_callback import LoggingCallback
from callbacks.outfeed_queue_callback import OutFeedQueueCallback
from callbacks.allreduce_metrics_callback import AllReduceMetricsCallback


class PopDistStrategyEquivalenceToIPUStrategy(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)

    def call_program(self,
                     num_instances: int,
                     num_replicas: int,
                     weight_updates: int,
                     gradient_accumulation: int,
                     test_reduction: bool = False,
                     accelerator_side_reduction: bool = False):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        working_path = os.getcwd()

        cmd_prefix = [] if num_instances == 0 else ['poprun',
                                                    '--num-instances', f'{num_instances}',
                                                    '--num-replicas', f'{num_replicas}',
                                                    '--only-output-from-instance', '0']

        cmd_suffix = ['--num-replicas', f'{num_replicas}'] if num_instances == 0 else []
        cmd_suffix += ['--weight-updates', f'{weight_updates}']
        cmd_suffix += ['--gradient-accumulation-count', f'{gradient_accumulation}']
        cmd_suffix += ['--accelerator-side-reduction'] if accelerator_side_reduction else []

        cmd = cmd_prefix + ['python3', 'test_distributed_training.py'] + cmd_suffix

        try:
            completed = subprocess.run(args=cmd,
                                       cwd=working_path,
                                       shell=False, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, check=True)
            decoded_output = str(completed.stdout, 'utf-8')
            if not test_reduction:
                matches = re.findall(r'weight = \d*\.\d*', decoded_output)[0]
                return float(matches.split(' = ')[-1])
            else:
                d = self.extract_logging_data_from_output(decoded_output)
                return d['average_loss'], d['average_accuracy']

        except subprocess.CalledProcessError as e:
            print(f"The following command failed: {cmd}\n"
                  f"Working path: {working_path}\n"
                  f"Output of failed command:\n{e.output}")
            return 0.0

    @staticmethod
    def extract_logging_data_from_output(output):
        # parse to get the loss
        last_logging_callback_line = [
            line for line in output.splitlines() if 'INFO:logging_callback' in line][-1]
        matches = re.findall(r'(\[[^\[\]]+\])', last_logging_callback_line)
        import json
        d = str(json.loads(matches[-1].replace("'", '"')))
        vals = (re.findall('\d*\.\d*', d))
        keys = (re.findall('[^\'\"]*:', d))
        keys = [key[:-1] for key in keys]

        d = dict()
        for i in range(len(keys)):
            d[keys[i]] = float(vals[i])
        return d

    def test_multireplica_outfeed_reduction(self):
        loss_allreduce_on_host, metrics_allreduce_on_host = self.call_program(
            num_instances=2,
            num_replicas=4,
            weight_updates=2,
            gradient_accumulation=1,
            test_reduction=True,
            accelerator_side_reduction=False
        )
        loss_allreduce_on_device, metrics_allreduce_on_device = self.call_program(
            num_instances=2,
            num_replicas=4,
            weight_updates=2,
            gradient_accumulation=1,
            test_reduction=True,
            accelerator_side_reduction=True
        )
        loss_single_instance, metrics_single_instance = self.call_program(
            num_instances=0,
            num_replicas=4,
            weight_updates=2,
            gradient_accumulation=1,
            test_reduction=True,
        )

        # verify losses
        self.assertAlmostEqual(loss_single_instance, loss_allreduce_on_host, places=4)
        self.assertAlmostEqual(loss_single_instance, loss_allreduce_on_device, places=4)
        # verify metrics
        self.assertAlmostEqual(metrics_single_instance, metrics_allreduce_on_host, places=4)
        self.assertAlmostEqual(metrics_single_instance, metrics_allreduce_on_device, places=4)

    def test_ipu_strategy_replication(self):
        self.assertAlmostEqual(
            first=self.call_program(num_instances=0, num_replicas=4, weight_updates=1, gradient_accumulation=1),
            second=0.3111111,
            places=4
        )

    def test_ipu_strategy_accumulation(self):
        self.assertAlmostEqual(
            first=self.call_program(num_instances=0, num_replicas=1, weight_updates=1, gradient_accumulation=4),
            second=0.3111111,
            places=4
        )

    def test_popdist_strategy_multiple_instances(self):
        self.assertAlmostEqual(
            first=self.call_program(num_instances=2, num_replicas=4, weight_updates=1, gradient_accumulation=1),
            second=0.3111111,
            places=4
        )

    def test_popdist_strategy_multiple_instances_and_accumulation(self):
        self.assertAlmostEqual(
            first=self.call_program(num_instances=2, num_replicas=4, weight_updates=1, gradient_accumulation=4),
            second=0.3111111,
            places=4
        )


if __name__ == '__main__':

    def ipu_prog(num_replicas, gradient_accumulation, accelerator_side_reduction, weight_updates=1):

        import logging

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        popdist_on = popdist.isPopdistEnvSet()

        num_global_replicas = popdist.getNumTotalReplicas() if popdist_on else num_replicas
        num_instances = popdist.getNumInstances() if popdist_on else 1

        global_batch_size = 16
        dataset_size = global_batch_size * weight_updates
        micro_batch_size = int(global_batch_size / num_global_replicas / gradient_accumulation)

        X = np.linspace(0, 1, num=dataset_size, dtype=float)
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
            popdist.init()
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

            micro_batches_per_weight_update = num_global_replicas * gradient_accumulation
            steps_per_execution = dataset_size // global_batch_size * micro_batches_per_weight_update

            print(f'weight_updates {weight_updates}')
            print(f'step_per_execution {steps_per_execution}')

            # wrap loss and metrics in enqueuer
            loss_class = tf.keras.losses.MeanSquaredError
            loss_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

            accuracy_class = tf.keras.metrics.MeanSquaredError
            accuracy_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

            if num_instances > 1 and accelerator_side_reduction:
                loss_class = wrap_loss_in_allreduce_enqueuer(
                    loss_class,
                    outfeed_queue=loss_outfeed_queue,
                    num_replicas=num_global_replicas
                )
                accuracy_class = wrap_metric_in_allreduce_enqueuer(
                    accuracy_class,
                    outfeed_queue=accuracy_outfeed_queue,
                    num_replicas=num_global_replicas
                )
            else:
                loss_class = wrap_loss_in_enqueuer(loss_class, loss_outfeed_queue)
                accuracy_class = wrap_metric_in_enqueuer(accuracy_class, accuracy_outfeed_queue)

            loss = loss_class()
            accuracy = accuracy_class()

            steps_per_execution_per_replica = steps_per_execution // num_global_replicas

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=[accuracy],
                          steps_per_execution=steps_per_execution_per_replica)

            callbacks = [OutFeedQueueCallback(queue=loss_outfeed_queue, name='average_loss'),
                         OutFeedQueueCallback(queue=accuracy_outfeed_queue, name='average_accuracy')]
            if num_instances > 1 and not accelerator_side_reduction:
                callbacks += [AllReduceMetricsCallback()]
            callbacks += [LoggingCallback(1)]

            steps_per_epoch_per_instance = steps_per_execution * weight_updates // num_instances

            model.fit(ds, steps_per_epoch=steps_per_epoch_per_instance, callbacks=callbacks)

            return model.get_weights()[0][0][0]

    import argparse

    def add_arguments(parser):
        parser.add_argument('--num-replicas', type=int, default=1)
        parser.add_argument('--gradient-accumulation-count', type=int, default=1)
        parser.add_argument('--accelerator-side-reduction', action='store_true', default=False)
        parser.add_argument('--weight-updates', type=int, default=1)
        return parser

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_arguments(parser)
    args = parser.parse_args()
    print(args)

    weight = ipu_prog(num_replicas=args.num_replicas,
                   gradient_accumulation=args.gradient_accumulation_count,
                   accelerator_side_reduction=args.accelerator_side_reduction,
                   weight_updates=args.weight_updates)
    print(f"weight = {weight}")
