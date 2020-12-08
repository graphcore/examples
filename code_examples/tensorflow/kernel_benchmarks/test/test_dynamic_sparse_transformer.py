# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import os

import pytest
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


@pytest.mark.usefixtures("ipu_sparse_ops")
class TestTensorFlowSparseTransformerBenchmarks(SubProcessChecker):
    """High-level integration tests for TensorFlow dynamic sparse transformer layer benchmarks"""

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_feed_forward_fp32(self):
        # GPT2 medium training with seq 256
        options = " --mode train --sparsity=0.9 --source-sequence-length=256 --hidden-length=1024 --ff-length=4096"
        self.run_command("python3 dynamic_sparse_transformer_ffn_block.py" + options, working_path,
                         [r"(\w+.\w+) tokens/sec, problem size (\w+.\w+) GFLOPS, (\w+.\w+) TFLOPS/sec"])

        # GPT2 large with training seq 128
        options = " --mode train --sparsity=0.9 --source-sequence-length=128 --hidden-length=1280 --ff-length=5120"
        self.run_command("python3 dynamic_sparse_transformer_ffn_block.py" + options, working_path,
                         [r"(\w+.\w+) tokens/sec, problem size (\w+.\w+) GFLOPS, (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_feed_forward_fp16(self):
        # GPT2 medium training with seq 256
        options = " --mode train --sparsity=0.9 --source-sequence-length=256 --hidden-length=1024 --ff-length=4096 --dtype=float16"
        self.run_command("python3 dynamic_sparse_transformer_ffn_block.py" + options, working_path,
                         [r"(\w+.\w+) tokens/sec, problem size (\w+.\w+) GFLOPS, (\w+.\w+) TFLOPS/sec"])

        # GPT2 large training with seq 128
        options = " --mode train --sparsity=0.9 --source-sequence-length=128 --hidden-length=1280 --ff-length=5120 --dtype=float16"
        self.run_command("python3 dynamic_sparse_transformer_ffn_block.py" + options, working_path,
                         [r"(\w+.\w+) tokens/sec, problem size (\w+.\w+) GFLOPS, (\w+.\w+) TFLOPS/sec"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_attention_fp32(self):
        # GPT2 medium training with seq 256
        options = " --mode train --sparsity=0.9 --source-sequence-length=256 --hidden-length=1024"
        options += "  --attention-heads=16 --qkv-length=64"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_self_attention.py" + options, working_path, [regex])

        # GPT2 large training with seq 128
        options = " --mode train --sparsity=0.9 --source-sequence-length=128 --hidden-length=1280"
        options += "  --attention-heads=16 --qkv-length=80"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_self_attention.py" + options, working_path, [regex])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_attention_fp16(self):
        # GPT2 medium training with seq 256
        options = " --mode train --sparsity=0.9 --source-sequence-length=256 --hidden-length=1024 --dtype=float16"
        options += "  --attention-heads=16 --qkv-length=64"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_self_attention.py" + options, working_path, [regex])

        # GPT2 large training with seq 128
        options = " --mode train --sparsity=0.9 --source-sequence-length=128 --hidden-length=1280 --dtype=float16"
        options += "  --attention-heads=16 --qkv-length=80"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_self_attention.py" + options, working_path, [regex])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_static_sparse_autoregressive_self_attention_fp16(self):
        # GPT2 medium training with seq 256
        options = " --mode train --sparsity=0.9 --source-sequence-length=256 --hidden-length=1024 --dtype=float16"
        options += "  --attention-heads=16 --qkv-length=64 --use-static-sparse-autoregressive-attention"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_self_attention.py" + options, working_path, [regex])


    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_encoder_layer_fp32(self):
        # GPT2 medium training with seq 128
        options = " --mode train --sparsity=0.9 --source-sequence-length=128 --hidden-length=1024"
        options += "  --attention-heads=16 --qkv-length=64 --ff-length=4096"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        env = os.environ.copy()
        env['POPLAR_ENGINE_OPTIONS'] = '{"opt.internalExchangeOptimisationTarget": "balanced"}'
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)

        # GPT2 medium inference with seq 128, batch size 2
        options = " --mode infer --sparsity=0.9 --source-sequence-length=128 --hidden-length=1024"
        options += " --batch-size=2 --attention-heads=16 --qkv-length=64 --ff-length=4096"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_encoder_layer_fp16(self):
        sparse_matmul_options = {"metaInfoBucketOversizeProportion": 0.2, "availableMemoryProportion": 0.9}
        sparse_matmul_options = str(sparse_matmul_options).replace("'", '"').replace(" ", "")
        # GPT2 medium training with seq 128
        options = " --mode train --sparsity=0.9 --source-sequence-length=128 --hidden-length=1024"
        options += "  --attention-heads=16 --qkv-length=64 --ff-length=4096 --dtype=float16"
        options += f" --sparse-matmul-options={sparse_matmul_options}"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        env = os.environ.copy()
        env['POPLAR_ENGINE_OPTIONS'] = '{"opt.internalExchangeOptimisationTarget": "balanced"}'
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)

        # GPT2 medium inference with seq 128, batch size 2
        options = " --mode infer --sparsity=0.9 --source-sequence-length=128 --hidden-length=1024"
        options += " --batch-size=2  --attention-heads=16 --qkv-length=64 --ff-length=4096 --dtype=float16"
        options += f" --sparse-matmul-options={sparse_matmul_options}"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_encoder_layer_fp32_blocksize_8(self):
        # GPT2 medium training with seq 128
        options = " --mode train --block-size=8 --sparsity=0.9 --source-sequence-length=128"
        options += " --hidden-length=1024 --attention-heads=16 --qkv-length=64 --ff-length=4096"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        env = os.environ.copy()
        env['POPLAR_ENGINE_OPTIONS'] = '{"opt.internalExchangeOptimisationTarget": "balanced"}'
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)

        # GPT2 medium inference with seq 128, batch size 2
        options = " --mode infer --block-size=8 --sparsity=0.9 --source-sequence-length=128 --batch-size=2"
        options += " --hidden-length=1024 --attention-heads=16 --qkv-length=64 --ff-length=4096"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)


    @pytest.mark.category1
    @pytest.mark.ipus(1)
    def test_encoder_layer_fp16_blocksize_16(self):
        # Take advantage of FP16 partials
        sparse_matmul_options = {"metaInfoBucketOversizeProportion": 0.2, "availableMemoryProportion": 0.9, "partialsType": "half"}
        sparse_matmul_options = str(sparse_matmul_options).replace("'", '"').replace(" ", "")
        # GPT2 medium training with seq 128
        options = " --mode train --block-size=16 --sparsity=0.9 --source-sequence-length=128 --dtype=float16"
        options += " --hidden-length=1024 --attention-heads=16 --qkv-length=64 --ff-length=4096"
        options += f" --sparse-matmul-options={sparse_matmul_options}"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        env = os.environ.copy()
        env['POPLAR_ENGINE_OPTIONS'] = '{"opt.internalExchangeOptimisationTarget": "balanced"}'
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)

        # GPT2 medium inference with seq 128, batch size 2
        options = " --mode infer --block-size=16 --sparsity=0.9 --batch-size=2 --source-sequence-length=128 --dtype=float16"
        options += " --hidden-length=1024 --attention-heads=16 --qkv-length=64 --ff-length=4096"
        options += f" --sparse-matmul-options={sparse_matmul_options}"
        regex = r"(\w+.\w+) tokens/sec, "
        regex += r"problem size (\w+.\w+) GFLOPS \(of which (\w+.\w+) sparse and (\w+.\w+) dense\). "
        regex += r"overall: (\w+.\w+) TFLOPS/sec"
        self.run_command("python3 dynamic_sparse_transformer_encoder_layer.py" + options, working_path, [regex], env=env)
