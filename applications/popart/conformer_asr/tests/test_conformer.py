# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import os

import popart
import conformer_blocks


def assert_lists_equal(alist, blist):
    assert(all([a == b for a, b in zip(alist, blist)]))


@pytest.mark.category1
@pytest.mark.parametrize("num_channels", [64, 256, 512])
def test_feedforward_module_build(num_channels):
    """ testing build of feed-forward module """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    ffn = conformer_blocks.FeedForwardModule(builder, num_channels)

    test_input = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_channels, sequence_length]))

    output = ffn(test_input)

    assert_lists_equal(builder.getTensorShape(output), [batch_size, num_channels, sequence_length])


@pytest.mark.category1
@pytest.mark.parametrize("num_channels, kernel_size", [(256, 17), (512, 17), (256, 33), (512, 33)])
def test_convolution_module_build(num_channels, kernel_size):
    """ testing build of convolution module """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    conv_module = conformer_blocks.ConvolutionModule(builder, num_channels, kernel_size)

    test_input = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_channels, sequence_length]))

    output = conv_module(test_input)

    assert_lists_equal(builder.getTensorShape(output), [batch_size, num_channels, sequence_length])


@pytest.mark.category1
@pytest.mark.parametrize("num_heads, num_features", [(2, 16), (2, 64), (2, 256), (2, 512),
                                                     (8, 16), (8, 64), (8, 256), (8, 512)])
def test_multihead_attention_block_build(num_heads, num_features):
    """ testing build of multi-headed attention block """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    mha = conformer_blocks.MultiHeadedAttention(builder, num_heads, num_features, sequence_length)

    queries = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_features, sequence_length]))
    keys = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_features, sequence_length]))
    values = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_features, sequence_length]))

    context_vecs = mha(queries, keys, values)

    assert_lists_equal(builder.getTensorShape(context_vecs), [batch_size, num_features, sequence_length])


@pytest.mark.category1
@pytest.mark.parametrize("num_heads, num_features", [(2, 16), (2, 64), (2, 256), (2, 512),
                                                     (8, 16), (8, 64), (8, 256), (8, 512)])
def test_multihead_self_attention_module_build(num_heads, num_features):
    """ testing build of multi-headed self-attention module """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    mhsa = conformer_blocks.MultiHeadedSelfAttentionModule(builder, num_heads, num_features, sequence_length)

    test_input = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_features, sequence_length]))

    output = mhsa(test_input)

    assert_lists_equal(builder.getTensorShape(output), [batch_size, num_features, sequence_length])


@pytest.mark.category1
@pytest.mark.parametrize("num_heads, num_features, kernel_size", [(2, 16, 33), (2, 64, 33), (2, 256, 33), (2, 512, 33),
                                                                  (8, 16, 33), (8, 64, 33), (8, 256, 33), (8, 512, 33)])
def test_conformer_block_build(num_heads, num_features, kernel_size):
    """ testing build of conformer block """
    builder = popart.Builder()

    batch_size = 4
    sequence_length = 100
    conformer_block = conformer_blocks.ConformerBlock(builder, num_heads, num_features, sequence_length, kernel_size)

    test_input = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, num_features, sequence_length]))

    output = conformer_block(test_input)

    assert_lists_equal(builder.getTensorShape(output), [batch_size, num_features, sequence_length])
