# Copyright 2020 Graphcore Ltd
import os
import subprocess
import unittest

import pytest


def run_profiling(**kwargs):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['gc-profile', '-d', 'testdir', '--', 'python3', 'train.py']
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    return subprocess.check_output(cmd).decode('utf-8')


def get_profile_csv(out, name):
    log_dir = None
    logfile = open('./testdir/stdout.txt', 'r')
    log = logfile.read().splitlines()
    for line in log:
        if line.find('Saving to ') != -1:
            log_dir = line[11:]
            break
    if not log_dir:
        raise ValueError("Couldn't find log directory from output")

    with open(os.path.join(log_dir, name)) as csv:
        lines = csv.read().split('\n')
        items = [line.split(',') for line in lines if line]
        results = {}  # dict with headers of csv as keys
        for i in range(len(items[0])):
            values = [float(v[i]) for v in items[1:]]
            results[items[0][i]] = values
    return results


@pytest.mark.ipus(1)
@pytest.mark.category1
class TestProfiling(unittest.TestCase):
    """Testing some basic training parameters"""

    @classmethod
    def setUpClass(cls):
        out = run_profiling(**{
                           '--dataset': 'cifar-10',
                           '--warmup-epochs': 0,
                           '--synthetic-data': '',
                           '--iterations': 1,
                           '--batches-per-step': 1,
                           '--gc-profile': ''
                           })
        cls.training = get_profile_csv(out, 'training.csv')

    def test_results(self):
        # test_archive_exists
        archive = './testdir/archive.a'
        self.assertTrue(os.path.isfile(archive))

        # test_execution_report
        execution = './testdir/execution.json'
        self.assertTrue(os.path.isfile(execution))
