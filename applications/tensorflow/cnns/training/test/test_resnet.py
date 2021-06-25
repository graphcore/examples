# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the ResNet model.
"""

import glob
import json
import os
import portpicker
import statistics
import subprocess
import sys
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


@pytest.mark.category2
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

    def test_results(self):
        # test_logdir
        self.assertEqual(self.logdir[:12], 'logs/walrus/')
        # test_name_suffix
        self.assertNotEqual(self.logdir.find('penguin'), -1)


@pytest.mark.category2
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

    def test_results(self):
        # test_final_validation_accuracy
        final_acc = self.validation['val_acc'][-1]
        self.assertGreater(final_acc, 80)
        self.assertLess(final_acc, 87)

        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 80)
        self.assertLess(final_acc, 87)

        # test_learning_rates
        self.assertEqual(self.training['lr'][0], 0.5)
        self.assertEqual(self.training['lr'][-1], 0.0005)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 10)


@pytest.mark.category2
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

    def test_results(self):
        # test_final_validation_accuracy
        final_acc = statistics.median(self.validation['val_acc'][-3:-1])
        self.assertGreater(final_acc, 89.0)
        self.assertLess(final_acc, 91.0)

        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 96)
        self.assertLess(final_acc, 98)

        # test_final_loss
        self.assertLess(self.training['loss_batch'][-1], 0.45)
        self.assertGreater(self.training['loss_batch'][-1], 0.35)

        # test_epochs_completed
        self.assertEqual(round(self.training['epoch'][-1]), 50)


@pytest.mark.category2
@pytest.mark.ipus(1)
class TestResNet50SingleIPUTraining(unittest.TestCase):
    """ResNet-50 example on a single IPU.
        This is differs from the command line in the README:
        here we are testing with generated random data and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--generated-data': '',
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
    """Pipelined ResNet-50 from the README but with generated random data
        and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 10,
                           '--batches-per-step': 10,
                           '--dataset': 'imagenet',
                           '--generated-data': '',
                           '--model-size': 50,
                           '--shards': 2,
                           '--pipeline': '',
                           '--gradient-accumulation-count': 256,
                           '--batch-size': 2,
                           '--no-validation': '',
                           '--enable-recomputation': '',
                           '--available-memory-proportion': 0.1,
                           '--pipeline-splits': 'b3/1/relu'})
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)
        self.assertGreater(self.training['loss_batch'][-1], 0)


@pytest.mark.category2
@pytest.mark.ipus(4)
class TestResNet50Pipelining2IPUs2Replicas(unittest.TestCase):
    """Pipelined and replicated ResNet-50 from the README but with generated random
       data and only 10 iterations.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--iterations': 10,
                           '--batches-per-step': 10,
                           '--dataset': 'imagenet',
                           '--generated-data': '',
                           '--model-size': 50,
                           '--shards': 2,
                           '--replicas': 2,
                           '--pipeline': '',
                           '--gradient-accumulation-count': 128,
                           '--pipeline-schedule': 'Grouped',
                           '--batch-size': 2,
                           '--no-validation': '',
                           '--enable-recomputation': '',
                           '--available-memory-proportion': 0.1,
                           '--pipeline-splits': 'b3/0/relu'})
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 10)
        self.assertGreater(self.training['loss_batch'][-1], 0)


@pytest.mark.category2
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

    def test_results(self):
        # test_final_training_accuracy
        final_acc = self.training['train_acc_avg'][-1]
        self.assertGreater(final_acc, 85)
        self.assertLess(final_acc, 95)
        # test_epochs_completed
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
                           '--generated-data': '',
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
    def test_results(self):
        # test_learning_rate
        self.assertEqual(self.training['lr'][0], 1.0)

        # test_epoch
        self.assertEqual(int(self.validation['epoch'][-1] + 0.5), 10)


@pytest.mark.category1
@pytest.mark.ipus(16)
class TestPopdist(unittest.TestCase):
    """Testing training with popdist launched using poprun."""

    def test_resnet8(self):

        NUM_TOTAL_REPLICAS = 4
        NUM_INSTANCES = 2
        NUM_LOCAL_REPLICAS = NUM_TOTAL_REPLICAS // NUM_INSTANCES

        with tempfile.TemporaryDirectory() as logdir:
            # The buildbot runs as root, so let's allow that.
            cmd = [
                'poprun',
                '--mpi-global-args=--tag-output --allow-run-as-root',
                '--num-replicas=' + str(NUM_TOTAL_REPLICAS),
                '--num-instances=' + str(NUM_INSTANCES),
                '--vipu-server-timeout=600',
                sys.executable,
                'train.py',
                '--dataset=cifar-10',
                '--generated-data',
                '--model-size=8',
                '--batch-size=1',
                '--batches-per-step=10',
                '--gradient-accumulation-count=10',
                '--no-validation',
                '--no-stochastic-rounding',
                '--iterations=100',
                '--warmup-epochs=0',
                '--log-dir', logdir,
                '--name-suffix', 'popdist_instance',
                '--ckpt-all-instances', "true",
                '--log-all-instances', "true",
                '--on-demand'
            ]

            # Add some debug logging.
            extra_env = {
                'POPRUN_LOG_LEVEL': 'TRACE',
                'TF_CPP_VMODULE': 'poplar_compiler=1,poplar_executor=1',
            }

            cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            env = os.environ.copy()
            env.update(extra_env)
            subprocess.check_call(cmd, cwd=cwd, env=env)

            instance_logdirs = glob.glob(f"{logdir}/*_popdist_instance_*")
            self.assertEqual(len(instance_logdirs), NUM_INSTANCES)

            training_logs = []

            for instance_logdir in instance_logdirs:
                # Check that each instance got the correct number of replicas from popdist.
                with open(os.path.join(instance_logdir, 'arguments.json'), 'r') as f:
                    argument_log = json.load(f)
                self.assertEqual(argument_log['replicas'], NUM_LOCAL_REPLICAS)

                # Check that the final accuracy is decent.
                training_log = parse_csv(os.path.join(instance_logdir, 'training.csv'))
                self.assertGreater(training_log['train_acc_avg'][-1], 95)
                training_logs.append(training_log)

            # The final training accuracy should be the same on all instances.
            for i in range(1, NUM_INSTANCES):
                self.assertEqual(
                    training_logs[0]['train_acc_avg'][-1],
                    training_logs[i]['train_acc_avg'][-1])

            # The final training loss should be the same on all instances.
            for i in range(1, NUM_INSTANCES):
                self.assertEqual(
                    training_logs[0]['loss_avg'][-1],
                    training_logs[i]['loss_avg'][-1])

            # The final weights should be the same on all instances.
            var_names_and_shapes = tf.train.list_variables(instance_logdirs[0])

            for var_name, _ in var_names_and_shapes:
                value_instance_0 = tf.train.load_variable(instance_logdirs[0], var_name)

                for i in range(1, NUM_INSTANCES):
                    value_instance_i = tf.train.load_variable(instance_logdirs[i], var_name)
                    self.assertListEqual(value_instance_0.tolist(), value_instance_i.tolist())


@pytest.mark.category3
@pytest.mark.ipus(16)
class TestDistributedTraining(unittest.TestCase):
    """Testing distributed training with multiple processes on a single machine with 16 IPUs."""

    def test_resnet_50_from_readme(self):

        NUM_WORKERS = 2
        WORKER_TIMEOUT_SECONDS = 60 * 60

        with tempfile.TemporaryDirectory() as logdir:
            cmd = [
                'python3', 'train.py',
                '--dataset=imagenet',
                '--generated-data',
                '--model-size=50',
                '--batch-size=4',
                '--batches-per-step=1',
                '--shards=4',
                '--pipeline',
                '--gradient-accumulation-count=64',
                '--pipeline-splits', 'b1/2/relu', 'b2/3/relu', 'b3/5/relu',
                '--enable-recomputation',
                '--replicas=2',  # Instead of 4 to make two processes fit on one machine.
                '--distributed',
                '--no-stochastic-rounding',
                '--no-validation',
                '--iterations=100',
                '--learning-rate-schedule=1',
                '--base-learning-rate=-14',
                '--log-dir', logdir,
                '--ckpt-all-instances', "true",
                '--log-all-instances', "true",
                '--on-demand'
            ]

            extra_env = {
                'POPLAR_ENGINE_OPTIONS': '{"opt.maxCopyMergeSize": 8388608}',
            }

            cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            worker_ports = self._pick_unique_unused_ports(NUM_WORKERS)
            cluster_spec = {
                'worker': ['localhost:%s' % port for port in worker_ports]
            }

            processes = self._start_processes_with_tf_config(cmd, cwd, extra_env, cluster_spec)
            self._wait_for_processes(processes, WORKER_TIMEOUT_SECONDS)

            worker_log_dirs = self._find_worker_log_dirs(NUM_WORKERS, logdir)
            training_logs = [parse_csv(os.path.join(d, "training.csv")) for d in worker_log_dirs]

            # The final training accuracy should be the same on all workers.
            for i in range(1, NUM_WORKERS):
                self.assertEqual(
                    training_logs[0]['train_acc_avg'][-1],
                    training_logs[i]['train_acc_avg'][-1])

            # The final training loss should be the same on all workers.
            for i in range(1, NUM_WORKERS):
                self.assertEqual(
                    training_logs[0]['loss_avg'][-1],
                    training_logs[i]['loss_avg'][-1])

            # The final weights should be the same on all workers.
            var_names_and_shapes = tf.train.list_variables(worker_log_dirs[0])

            for var_name, _ in var_names_and_shapes:
                value_worker_0 = tf.train.load_variable(worker_log_dirs[0], var_name)

                for i in range(1, NUM_WORKERS):
                    value_worker_i = tf.train.load_variable(worker_log_dirs[i], var_name)
                    self.assertListEqual(value_worker_0.tolist(), value_worker_i.tolist())

    @staticmethod
    def _start_processes_with_tf_config(cmd, cwd, extra_env, cluster_spec):
        processes = []

        for i in range(len(cluster_spec['worker'])):
            env = os.environ.copy()
            env.update(extra_env)
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

    @staticmethod
    def _find_worker_log_dirs(num_workers, logdir):
        worker_log_dirs = []

        for i in range(num_workers):
            logdirs = glob.glob(f"{logdir}/*_worker{i}_*")
            if len(logdirs) != 1:
                raise RuntimeError(f"Expected 1 worker dir, found {len(logdirs)}: {logdirs}")
            worker_log_dirs.append(logdirs[0])

        return worker_log_dirs


@pytest.mark.category1
@pytest.mark.ipus(1)
class TestConfig(unittest.TestCase):
    """Testing lots of other options to check they are still available"""

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--config': 'mk2_resnet8_test',
                           '--data-dir': cifar10_data_dir,
                           })
        cls.training = get_csv(out, 'training.csv')


    def test_results(self):
        # test the cmd line arg overrode config arg
        self.assertEqual(int(self.training['epoch'][-1]), 10)


@pytest.mark.category2
@pytest.mark.ipus(2)
@pytest.mark.ipu_version("ipu2")
class TestResNet50RecomputeDbnTraining(unittest.TestCase):
    """ResNet-50 example on two IPUs with distributed batch norm and recompute.
    """

    @classmethod
    def setUpClass(cls):
        out = run_train(**{'--generated-data': '',
                           '--dataset': 'ImageNet',
                           '--model-size': 50,
                           '--batch-size': 8,
                           '--available-memory-proportion': 0.1,
                           '--iterations': 10,
                           '--BN-span': 2,
                           '--internal-exchange-optimisation-target': 'memory',
                           '--pipeline': '',
                           '--gradient-accumulation-count': 2,
                           '--pipeline-schedule': 'Sequential',
                           '--enable-recomputation': '',
                           '--pipeline-splits': 'b1/0/relu',
                           '--eight-bit': '',
                           '--replicas': 2,
                           '--enable-half-partials': '',
                           '--disable-variable-offloading': '',
                           '--batch-norm': '',
                           '--normalise-input': ''
                           })
        cls.validation = get_csv(out, 'validation.csv')
        cls.training = get_csv(out, 'training.csv')

    def test_iterations_completed(self):
        self.assertEqual(self.training['iteration'][-1], 500)
