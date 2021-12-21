# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import unittest
from pathlib import Path
import sys
import tensorflow as tf
sys.path.append(str(Path(__file__).absolute().parent.parent))
from callbacks.custom_wandb_callback import CustomWandbCallback
from custom_exceptions import MissingArgumentException


class Mock_wandb:
    args = dict()

    @staticmethod
    def log(args):
        Mock_wandb.log_invoked = True
        Mock_wandb.args = {**Mock_wandb.args, **args}

    @staticmethod
    def init(**kwargs):
        pass

    @staticmethod
    def clear():
        Mock_wandb.log_invoked = False
        Mock_wandb.args = {}


class Mock_time:
    time_step = 10
    counter = 0

    @staticmethod
    def time():
        Mock_time.counter += Mock_time.time_step
        return Mock_time.counter


# class below is to prevent logging graph to wandb
class Mock_run:
    summary = {'graph': 0}


# class below is to prevent logging graph to wandb
class Mock_Graph:
    def from_keras(model):
        return model


class WandBCallbackMissingArgumentTest(unittest.TestCase):
    def test_missing_model_name(self):
        with self.assertRaises(MissingArgumentException):
            callback = CustomWandbCallback(log_period=10, model=tf.keras.Sequential(),
                                           args={'dataset': 'random'})

    def test_missing_dataset(self):
        with self.assertRaises(MissingArgumentException):
            callback = CustomWandbCallback(log_period=10, model=tf.keras.Sequential(),
                                           args={'model_name': 'toy_example'})


class LogPeriodTest(unittest.TestCase):

    @unittest.mock.patch('callbacks.custom_wandb_callback.wandb.log', Mock_wandb.log)
    @unittest.mock.patch('callbacks.custom_wandb_callback.wandb.init', Mock_wandb.init)
    @unittest.mock.patch('callbacks.custom_wandb_callback.wandb.run', Mock_run)
    @unittest.mock.patch('callbacks.custom_wandb_callback.wandb.Graph', Mock_Graph)
    def test_log_period(self):

        log_period = 3
        args = {'dataset': 'random', 'model_name': 'toy_example', 'micro_batch_size': 1}
        callback = CustomWandbCallback(log_period=log_period, model=tf.keras.Sequential(), args=args)

        callback.on_train_begin()
        callback.on_train_batch_begin(0)
        callback.on_train_batch_end(0, logs={'Compilation Time': 3, 'dummy_metric': 0})
        self.assertTrue(Mock_wandb.log_invoked)
        self.assertEqual(Mock_wandb.args, {'Compilation Time': 3})

        for i, expected_value in zip(range(1, 3), [False, True]):
            Mock_wandb.clear()
            callback.on_train_batch_begin(i)
            callback.on_train_batch_end(i, logs={'dummy_metric': i})
            self.assertEqual(Mock_wandb.log_invoked, expected_value, f'failed on repeated call {i}')
            if expected_value:
                self.assertEqual(Mock_wandb.args, {'dummy_metric': 1})
