# Copyright 2020 Graphcore Ltd.
import unittest
import os
import sys
import subprocess


def run_resnet(**kwargs):
    cmd = ["python" + str(sys.version_info[0]),
           'resnet_synthetic_benchmark.py']
    # Flatten kwargs and convert to strings
    args = [str(item) for sublist in kwargs.items() for item in sublist if item != '']
    cmd.extend(args)
    out = subprocess.check_output(cmd, cwd=os.path.dirname(__file__)).decode("utf-8")
    print(out)
    return out


class TestPopARTResNetSyntheticBenchmarks(unittest.TestCase):
    """Tests for ResNet popART synthetic benchmarks"""

    @classmethod
    def setUpClass(cls):
        pass

    # Resnet inference
    def test_resnet_20_inference_batch_size_32(self):
        out = run_resnet(**{'--size': 20,
                            '--batch-size': 32,
                            '--norm-type': 'BATCH',
                            '--shortcut-type': 'B',
                            '--use-data': ''})

    def test_resnet_18_inference_batch_size_1(self):
        out = run_resnet(**{'--size': 18,
                            '--batch-size': 1,
                            '--norm-type': 'GROUP'})

    def test_resnet_18_inference_batch_size_16(self):
        out = run_resnet(**{'--size': 18,
                            '--batch-size': 16,
                            '--norm-type': 'BATCH'})

    def test_resnet_50_inference_batch_size_8(self):
        out = run_resnet(**{'--size': 50,
                            '--batch-size': 8,
                            '--norm-type': 'NONE'})

    def test_resnet_50_inference_batch_size_8_pipelined_4ipus(self):
        out = run_resnet(**{'--size': 50,
                            '--batch-size': 8,
                            '--norm-type': 'NONE',
                            '--shards': 4,
                            '--pipeline': ''})

    # ResNet training
    def test_resnet_18_train_batch_size_4(self):
        out = run_resnet(**{'--size': 18,
                            '--batch-size': 4,
                            '--norm-type': 'GROUP',
                            '--mode': 'train'})

    def test_resnet_18_train_batch_size_4_pipelined_2ipus(self):
        out = run_resnet(**{'--size': 18,
                            '--batch-size': 4,
                            '--norm-type': 'GROUP',
                            '--mode': 'train',
                            '--shards': 2,
                            '--pipeline': ''})

    def test_resnet_50_train_batch_size_1(self):
        out = run_resnet(**{'--size': 50,
                            '--batch-size': 1,
                            '--norm-type': 'GROUP',
                            '--mode': 'train'})

    def test_resnet_50_train_sharded(self):
        out = run_resnet(**{'--size': 50,
                            '--batch-size': 2,
                            '--norm-type': 'GROUP',
                            '--mode': 'train',
                            '--shards': 2})

    def test_resnet_20_train_batch_size_32(self):
        out = run_resnet(**{'--size': 20,
                            '--batch-size': 16,
                            '--norm-type': 'BATCH',
                            '--mode': 'train'})
