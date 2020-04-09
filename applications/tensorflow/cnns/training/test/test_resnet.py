# Copyright 2020 Graphcore Ltd.
"""
Tests covering various CNN training options using the ResNet model.
"""

import glob
import json
import os
import portpicker
import shutil
import statistics
import subprocess
import tempfile
import tensorflow as tf
import time
import unittest
import pytest

from test_common import get_csv, parse_csv, run_train, cifar10_data_dir


@pytest.mark.category1
class TestBasicFunctionality(unittest.TestCase):
    """ Test that the help option works"""
    def test_help(self):
        help_out = run_train(**{'--help': ''})
        self.assertNotEqual(help_out.find("usage: train.py"), -1)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestMisc(unittest.TestCase):
    """Some miscellaneous options"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--name-suffix': 'penguin',
                           '--log-dir': 'logs/walrus',
                           '--iterations': 10,
                           '--batches-per-step': 10})
        cls.logdir = None
        for line in out.split('\n'):
            if line.find('Saving to ') != -1:
                cls.logdir = line[11:]
                break
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_logdir(self):
        self.assertEqual(self.logdir[:12], 'logs/walrus/')

    def test_name_suffix(self):
        self.assertNotEqual(self.logdir.find('penguin'), -1)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestCifar10Training(unittest.TestCase):
    """Testing some basic training parameters"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--epochs': 10,
                           '--warmup-epochs': 0,
                           '--learning-rate-decay': '0.1',
                           '--learning-rate-schedule': '0.5,0.75,0.875'})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_final_validation_accuracy(self):
        final_acc = self.validation['val_acc'][-1]
        self.assertGreater(final_acc, 81)
        self.assertLess(final_acc, 87)

    def test_final_training_accuracy(self):
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 82)
        self.assertLess(final_acc, 87)

    def test_learning_rates(self):
        self.assertEqual(self.training['lr'][0], 0.5)
        self.assertEqual(self.training['lr'][-1], 0.0005)

    def test_epochs_completed(self):
        self.assertEqual(round(self.training['epoch'][-1]), 10)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestCifar10FullTraining(unittest.TestCase):
    """Fast training of Cifar-10 to good accuracy"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--epochs': 50,
                           '--batch-size': 48,
                           '--warmup-epochs': 2,
                           '--lr-schedule': 'cosine',
                           '--label-smoothing': '0.05',
                           '--base-learning-rate': -5,
                           '--precision': '16.32'})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_final_validation_accuracy(self):
        final_acc = statistics.median(self.validation['val_acc'][-3:-1])
        self.assertGreater(final_acc, 91.0)
        self.assertLess(final_acc, 93.0)

    def test_final_training_accuracy(self):
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 96)
        self.assertLess(final_acc, 98)

    def test_final_loss(self):
        self.assertLess(self.training['loss_batch'][-1], 0.45)
        self.assertGreater(self.training['loss_batch'][-1], 0.35)

    def test_epochs_completed(self):
        self.assertEqual(round(self.training['epoch'][-1]), 50)


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestResNet50SingleIPUTraining(unittest.TestCase):
    """ResNet-50 example on a single IPU.
        This is differs from the command line in the README:
        here we are testing with synthetic data and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--synthetic-data': '',
                           '--dataset': 'ImageNet',
                           '--model-size': 50,
                           '--batch-size': 1,
                           '--available-memory-proportion': 0.1,
                           '--iterations': 10,
                           '--batches-per-step': 10})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.category2
@pytest.mark.ipus(2)
class TestResNet50Pipelining2IPUs(unittest.TestCase):
    """Pipelined ResNet-50 from the README but with synthetic data
        and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 10,
                           '--batches-per-step': 10,
                           '--dataset': 'imagenet',
                           '--synthetic-data': '',
                           '--model-size': 50,
                           '--shards': 2,
                           '--pipeline-depth': 256,
                           '--batch-size': 2,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--available-memory-proportion': 0.1,
                           '--pipeline-splits': 'b3/1/relu'})
        cls.training = get_csv(out, 'training.csv')
        print(cls.training)

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.category2
@pytest.mark.ipus(4)
class TestResNet50Pipelining2IPUs2Replicas(unittest.TestCase):
    """Pipelined and replicated ResNet-50 from the README but with synthetic
       data and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 10,
                           '--batches-per-step': 10,
                           '--dataset': 'imagenet',
                           '--synthetic-data': '',
                           '--model-size': 50,
                           '--shards': 2,
                           '--replicas': 2,
                           '--pipeline-depth': 128,
                           '--pipeline-schedule': 'Grouped',
                           '--batch-size': 2,
                           '--no-validation': '',
                           '--xla-recompute': '',
                           '--available-memory-proportion': 0.1,
                           '--pipeline-splits': 'b3/0/relu'})
        cls.training = get_csv(out, 'training.csv')
        print(cls.training)

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)


@pytest.mark.category1
@pytest.mark.ipus(2)
class TestReplicatedTraining(unittest.TestCase):
    """Using replicas for data parallelism"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--data-dir': cifar10_data_dir,
                           '--model': 'resnet',
                           '--lr-schedule': 'stepped',
                           '--learning-rate-decay': 0.5,
                           '--learning-rate-schedule': '0.5,0.9',
                           '--epochs': 20,
                           '--replicas': 2})
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_final_training_accuracy(self):
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 85)
        self.assertLess(final_acc, 95)

    def test_epochs_completed(self):
        self.assertEqual(round(self.training['epoch'][-1]), 20)


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestLotsOfOptions(unittest.TestCase):
    """Testing lots of other options to check they are still available"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--dataset': 'cifar-10',
                           '--epochs': 10,
                           '--model-size': 14,
                           '--batch-norm': '',
                           '--pipeline-num-parallel': 8,
                           '--synthetic-data': '',
                           '--batch-size': 16,
                           '--base-learning-rate': -4,
                           '--precision': '32.32',
                           '--seed': 1234,
                           '--warmup-epochs': 0,
                           '--no-stochastic-rounding': '',
                           '--batches-per-step': 100
                           })
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    # We're mostly just testing that training still runs with all the above options.

    def test_learning_rate(self):
        self.assertEqual(self.training['lr'][0], 1.0)

    def test_epoch(self):
        self.assertEqual(int(self.validation['epoch'][-1] + 0.5), 10)


@pytest.mark.category1
@pytest.mark.ipus(4)
class TestDistributedTraining(unittest.TestCase):
    """Testing distributed training with multiple processes on a single machine."""

    NUM_WORKERS = 2
    WORKER_TIMEOUT_SECONDS = 10 * 60

    @classmethod
    def setUpClass(cls):
        cls.logdir = tempfile.mkdtemp()

        cmd = [
            'python3', 'train.py',
            '--dataset=cifar-10',
            '--synthetic-data',
            '--model-size=20',
            '--batch-size=1',
            '--iterations=100',
            '--batches-per-step=1',
            '--pipeline-depth=4',
            '--pipeline-splits', 'b2/0/relu',
            '--xla-recompute',
            '--shards=2',
            '--distributed',
            '--no-validation',
            '--no-stochastic-rounding',
            '--log-dir', cls.logdir,
        ]

        cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        worker_ports = cls._pick_unique_unused_ports(cls.NUM_WORKERS)
        cluster_spec = {
            'worker': ['localhost:%s' % port for port in worker_ports]
        }

        processes = cls._start_processes_with_tf_config(cmd, cwd, cluster_spec)
        cls._wait_for_processes(processes, cls.WORKER_TIMEOUT_SECONDS)

        cls.worker_log_dirs = cls._find_worker_log_dirs()
        cls.training_logs = [parse_csv(os.path.join(d, "training.csv")) for d in cls.worker_log_dirs]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.logdir)

    @staticmethod
    def _start_processes_with_tf_config(cmd, cwd, cluster_spec):
        processes = []

        for i in range(len(cluster_spec['worker'])):
            env = os.environ.copy()
            env["TF_CONFIG"] = json.dumps({
                "cluster": cluster_spec,
                "task": {
                    "type": "worker",
                    "index": i
                }
            })
            p = subprocess.Popen(cmd + ["--name-suffix", f"worker{i}"], cwd=cwd, env=env)
            processes.append(p)

        return processes

    @staticmethod
    def _pick_unique_unused_ports(num_ports):
        ports = set()
        while len(ports) < num_ports:
            ports.add(portpicker.pick_unused_port())
        return list(ports)

    @staticmethod
    def _wait_for_processes(processes, timeout):
        start_time = time.monotonic()
        remaining = processes[:]
        try:
            while remaining:
                p = remaining[0]
                elapsed = time.monotonic() - start_time
                returncode = p.wait(timeout - elapsed)
                # Only pop after a successful wait to allow for clean up.
                remaining.pop(0)
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd=" ".join(p.args))
        finally:
            # Try to clean up by killing any processes still alive.
            for p in remaining:
                p.kill()

    @classmethod
    def _find_worker_log_dirs(cls):
        worker_log_dirs = []

        for i in range(cls.NUM_WORKERS):
            logdirs = glob.glob(f"{cls.logdir}/*_worker{i}_*")
            if len(logdirs) != 1:
                raise RuntimeError(f"Expected 1 worker dir, found {len(logdirs)}: {logdirs}")
            worker_log_dirs.append(logdirs[0])

        return worker_log_dirs

    def test_final_train_acc_avg_is_same_on_workers(self):
        for i in range(1, self.NUM_WORKERS):
            self.assertEqual(
                self.training_logs[0]['train_acc_avg'][-1],
                self.training_logs[i]['train_acc_avg'][-1])

    def test_final_loss_avg_is_same_on_workers(self):
        for i in range(1, self.NUM_WORKERS):
            self.assertEqual(
                self.training_logs[0]['loss_avg'][-1],
                self.training_logs[i]['loss_avg'][-1])

    def test_variables_are_equal_on_workers(self):
        var_names_and_shapes = tf.train.list_variables(self.worker_log_dirs[0])

        for var_name, _ in var_names_and_shapes:
            value_worker_0 = tf.train.load_variable(self.worker_log_dirs[0], var_name)

            for i in range(1, self.NUM_WORKERS):
                value_worker_i = tf.train.load_variable(self.worker_log_dirs[i], var_name)
                self.assertListEqual(value_worker_0.tolist(), value_worker_i.tolist())
