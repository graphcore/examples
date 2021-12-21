# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import os
import math
import torch
from tempfile import TemporaryDirectory
import subprocess
import re

from rnnt_reference import config

import popart
import transducer_blocks


def assert_lists_equal(alist, blist):
    assert(all([a == b for a, b in zip(alist, blist)]))


def setup_generated_data_pipeline(conf, transducer_config, mel_bands=80,
                                  max_spec_len_before_stacking=1980,
                                  max_token_sequence_len=125,
                                  num_symbols=1024):
    """ returns a data loader providing random generated data """
    class GeneratedDataLoader:
        def __init__(self, conf, num_steps=5):

            self.num_steps = num_steps
            self._data_iterator = self.get_data_iterator(conf, num_steps)

        def get_data_iterator(self, conf, num_steps):

            for _ in range(conf.num_epochs * num_steps):
                generated_audio_data = torch.randn(conf.samples_per_step_per_instance,
                                                   conf.mel_bands,
                                                   conf.max_spec_len_before_stacking)
                generated_audio_lens_data = torch.randint(conf.max_token_sequence_len + 1,
                                                          conf.max_spec_len_before_stacking,
                                                          [conf.samples_per_step_per_instance],
                                                          dtype=torch.int32)
                generated_txt_data = torch.randint(0, conf.num_symbols,
                                                   [conf.samples_per_step_per_instance, conf.max_token_sequence_len],
                                                   dtype=torch.int32)
                generated_txt_lens_data = torch.randint(conf.max_token_sequence_len // 4,
                                                        conf.max_token_sequence_len,
                                                        [conf.samples_per_step_per_instance],
                                                        dtype=torch.int32)
                yield generated_audio_data, generated_audio_lens_data, generated_txt_data.numpy(), generated_txt_lens_data

        def data_iterator(self):
            return self._data_iterator

        def __len__(self):
            return self.num_steps

    train_dataset_kw, train_features_kw, train_splicing_kw, train_specaugm_kw = config.input(transducer_config, 'train')
    conf.train_splicing_kw = train_splicing_kw
    conf.train_specaugm_kw = train_specaugm_kw

    assert (conf.samples_per_step % conf.num_instances == 0)
    conf.samples_per_step_per_instance = conf.samples_per_step // conf.num_instances

    conf.mel_bands = mel_bands
    conf.max_spec_len_before_stacking = max_spec_len_before_stacking
    conf.max_spec_len_after_stacking = round(max_spec_len_before_stacking /
                                             train_splicing_kw["frame_subsampling"])
    conf.max_token_sequence_len = max_token_sequence_len
    conf.num_symbols = num_symbols

    generated_data_loader = GeneratedDataLoader(conf, num_steps=5)

    return generated_data_loader


@pytest.mark.category1
@pytest.mark.parametrize("num_in_channels, kernel_size, sequence_length",
                         [(256, 16, 64), (512, 32, 64), (256, 35, 64), (512, 42, 64),
                          (256, 16, 75), (512, 32, 83), (256, 35, 91), (512, 42, 14)])
def test_convolution_subsampler_build(num_in_channels, kernel_size, sequence_length):
    """ testing build of convolution subsampler """
    builder = popart.Builder()

    batch_size = 4
    subsampling_factor = 4
    num_out_channels = 3 * num_in_channels
    conv_subsampler = transducer_blocks.ConvolutionSubSampler(builder, num_in_channels, num_out_channels,
                                                              kernel_size, subsampling_factor,
                                                              np.float16, "ConvolutionSubSamplerTest")

    test_input = builder.addInputTensor(popart.TensorInfo("FLOAT16", [batch_size, num_in_channels, sequence_length]))

    output = conv_subsampler(test_input)

    assert_lists_equal(builder.getTensorShape(output), [batch_size, num_out_channels, math.ceil(sequence_length / subsampling_factor)])

    assert (conv_subsampler.param_count == num_out_channels * num_in_channels * kernel_size + num_out_channels)


@pytest.mark.category1
@pytest.mark.parametrize("num_heads, num_features", [(2, 16), (2, 64), (2, 256), (2, 512),
                                                     (8, 16), (8, 64), (8, 256), (8, 512)])
def test_multihead_attention_block_build(num_heads, num_features):
    """ testing build of multi-headed attention block """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    mha = transducer_blocks.MultiHeadedAttention(builder, num_heads, num_features, np.float16, "MultiHeadedAttentionTest")

    queries = builder.addInputTensor(popart.TensorInfo("FLOAT16", [batch_size, num_features, sequence_length]))
    keys = builder.addInputTensor(popart.TensorInfo("FLOAT16", [batch_size, num_features, sequence_length]))
    values = builder.addInputTensor(popart.TensorInfo("FLOAT16", [batch_size, num_features, sequence_length]))

    context_vecs = mha(queries, keys, values)

    assert_lists_equal(builder.getTensorShape(context_vecs), [batch_size, num_features, sequence_length])

    assert (mha.param_count == 4 * num_features * num_features)


@pytest.mark.category1
@pytest.mark.parametrize("num_heads, num_features, kernel_size", [(2, 16, 33), (2, 64, 33), (2, 256, 33), (2, 512, 33),
                                                                  (8, 16, 33), (8, 64, 33), (8, 256, 33), (8, 512, 33)])
def test_transformer_block_build(num_heads, num_features, kernel_size):
    """ testing build of transformer block """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    transformer_block = transducer_blocks.TransformerBlock(builder, num_heads, num_features,
                                                           np.float16, "TransformerBlockTest")

    test_input = builder.addInputTensor(popart.TensorInfo("FLOAT16", [batch_size, num_features, sequence_length]))

    output = transformer_block(test_input)

    assert_lists_equal(builder.getTensorShape(output), [batch_size, num_features, sequence_length])

    assert (transformer_block.param_count == (transformer_block.mhsa.param_count +
                                              transformer_block.linear_1.param_count +
                                              transformer_block.linear_2.param_count))


@pytest.mark.category3
@pytest.mark.ipus(2)
@pytest.mark.ipu_version("ipu2")
def test_transformer_transducer_train():
    """ testing train script for transformer-transducer """

    with TemporaryDirectory() as tmp_dir:
        cmd = ["python3", "transducer_train.py"]
        args = "--model-conf-file configs/transducer-mini.yaml " \
               "--model-dir {} --data-dir {} --max-duration 16.8 --batch-size 4 " \
               "--optimizer LAMB --enable-half-partials --enable-lstm-half-partials " \
               "--replication-factor 2 --batches-per-step 1 --gradient-accumulation-factor 32 " \
               "--loss-scaling 512.0 --base-lr 0.004 --joint-net-split-size 15 " \
               "--enable-stochastic-rounding --generated-data --num-epochs 1".format(tmp_dir, tmp_dir)

        args = args.split(" ")
        cmd.extend(args)

        try:
            output = subprocess.check_output(
                cmd, cwd=os.path.dirname(__file__), stderr=subprocess.PIPE
            ).decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(f"TEST FAILED")
            print(f"stdout={e.stdout.decode('utf-8', errors='ignore')}")
            print(f"stderr={e.stderr.decode('utf-8', errors='ignore')}")
            raise

        strings_to_match = ["Training graph preparation complete", "Throughput"]
        regexes = [re.compile(s) for s in strings_to_match]
        for i, r in enumerate(regexes):
            match = r.search(output)
            assert match, "Output of command: '{}' contained no match for: {} " \
                          "\nOutput was:\n{}".format(cmd, strings_to_match[i], output)
