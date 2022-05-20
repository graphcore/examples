# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""
Tests covering various CNN training options using the ResNet model.
"""

import glob
import json
import os
import portpicker
import subprocess
import sys
import tempfile
import tensorflow as tf
import time
import unittest
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))

from test_common import parse_csv
from examples_tests.test_util import SubProcessChecker


@pytest.mark.ipus(16)
class TestPopdist(SubProcessChecker):
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
                '--micro-batch-size=1',
                '--batches-per-step=10',
                '--gradient-accumulation-count=10',
                '--no-validation',
                '--stochastic-rounding=OFF',
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
            print(f"Command: {cmd}")
            self.run_command(cmd, cwd, [], env=env)

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


@pytest.mark.ipus(16)
@pytest.mark.ipu_version("ipu2")
class TestDistributedTraining(unittest.TestCase):
    """Testing distributed training with multiple processes on a single machine with 16 IPUs."""

    def test_resnet_50_from_readme(self):

        NUM_WORKERS = 2
        WORKER_TIMEOUT_SECONDS = 60 * 60 * 3

        with tempfile.TemporaryDirectory() as logdir:
            cmd = [
                'python3', 'train.py',
                '--dataset=imagenet',
                '--generated-data',
                '--model-size=50',
                '--micro-batch-size=4',
                '--batches-per-step=1',
                '--shards=4',
                '--pipeline',
                '--gradient-accumulation-count=64',
                '--pipeline-splits', 'stage1/unit3/relu', 'stage2/unit4/relu', 'stage3/unit6/relu',
                '--enable-recomputation',
                '--replicas=2',  # Instead of 4 to make two processes fit on one machine.
                '--distributed',
                '--stochastic-rounding=OFF',
                '--no-validation',
                '--iterations=10',
                '--learning-rate-schedule=1',
                '--base-learning-rate-exponent=-14',
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
