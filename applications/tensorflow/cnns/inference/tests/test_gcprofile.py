# Copyright 2020 Graphcore Ltd
import os
import subprocess
import unittest
import pytest


def get_images():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['./get_images.sh']
    return subprocess.check_output(cmd).decode('utf-8')


def run_profiling():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('..')
    cmd = ['gc-profile', '-d', 'testdir', '--', 'python', 'run_benchmark.py', 'resnet50', 'images',
           '--batch-size', '1', '--num-iterations', '1', '--batches-per-step', '1', '--gc-profile']
    return subprocess.check_output(cmd).decode('utf-8')


@pytest.mark.ipus(1)
@pytest.mark.category1
class TestProfiling(unittest.TestCase):
    """Testing --gc-profile option"""

    @classmethod
    def setUpClass(cls):
        get_images()
        run_profiling()

    def test_archive_exists(self):
        """Testing if the archive containing the tiles memory is created"""

        archive = './testdir/archive.a'
        self.assertTrue(os.path.isfile(archive))

    def test_execution_report(self):
        """Testing if the execution profile has been retrieved"""

        execution = './testdir/execution.json'
        self.assertTrue(os.path.isfile(execution))
