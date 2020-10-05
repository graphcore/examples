# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from examples_tests.test_util import SubProcessChecker
from pathlib import Path
import os
import pytest

build_dir = Path(__file__).parent.parent.parent


class TestBuildAndRun(SubProcessChecker):

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_ffn_block_fp32(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = ""
        full_cmd = "python3 ipu_sparse_ops/tools/sparse_transformer_ffn_block.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_ffn_block_fp16(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = "--embedding-dtype=tf.float16"
        full_cmd = "python3 ipu_sparse_ops/tools/sparse_transformer_ffn_block.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_self_attention_fp32(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = ""
        full_cmd = "python3 ipu_sparse_ops/tools/sparse_transformer_self_attention.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_self_attention_fp16(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = "--embedding-dtype=tf.float16"
        full_cmd = "python3 ipu_sparse_ops/tools/sparse_transformer_self_attention.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_encoder_layer_fp32(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = ""
        full_cmd = "python3 ipu_sparse_ops/tools/sparse_transformer_encoder_layer.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_encoder_layer_fp16(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = "--embedding-dtype=tf.float16"
        full_cmd = "python3 ipu_sparse_ops/tools/sparse_transformer_encoder_layer.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All results match."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_end_to_end_fp32(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = "--embedding-dtype=tf.float32"
        full_cmd = "python3 mnist_rigl/sparse_transformer_rigl_mnist.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All asserts pass."], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_transformer_end_to_end_fp16(self):
        env = os.environ.copy()
        env["PYTHONPATH"] += ":./"
        cmd_args = "--embedding-dtype=tf.float16 --optimizer-arg=epsilon=0.001"
        full_cmd = "python3 mnist_rigl/sparse_transformer_rigl_mnist.py " + cmd_args
        self.run_command(full_cmd, build_dir, ["All asserts pass."], env=env)
