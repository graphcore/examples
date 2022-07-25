# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu


def setup_random_seed():
    seed = 1989
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ipu.utils.reset_ipu_seed(seed)


@pytest.mark.usefixtures("length_regulator_op")
@pytest.mark.ipus(1)
def test_lr():
    from fastspeech2 import LengthRegulator, FastSpeech2Config
    from utils import create_ipu_config
    from tests.test_utils import check_tensor, check_tensor_relative

    setup_random_seed()
    config = FastSpeech2Config()
    batch_size = 1
    seq_len = config.max_seq_length
    wave_len = config.max_wave_length
    hidden_size = config.encoder_self_attention_params.hidden_size

    # create input data
    encoder_hidden_state = np.random.random((batch_size, hidden_size, seq_len))
    duration_gt = np.random.randint(0, 7, size=(batch_size, seq_len))
    while duration_gt.sum() > wave_len:
        duration_gt = np.random.randint(0, 7, size=(batch_size, seq_len))

    # expand each hidden state according to duration_gt
    out_gt = [
        np.array(
            [np.repeat(encoder_hidden_state[i, :, :], duration_gt[i], axis=-1)])
        for i in range(batch_size)
    ]

    # pad to wave_len
    out_gt = np.concatenate([
        np.pad(out_gt[i], [(0, 0), (0, 0),
               (0, wave_len - out_gt[i].shape[-1])])
        for i in range(len(out_gt))
    ],
        axis=0)

    encoder_hidden_state = tf.convert_to_tensor(
        encoder_hidden_state, tf.float32)
    duration_gt = tf.convert_to_tensor(duration_gt, tf.float32)

    cfg = create_ipu_config(
        available_memory_proportion=0.4, num_required_ipus=1)
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        lr = LengthRegulator(config)
        out_pd, mask = lr([encoder_hidden_state, duration_gt])
        check_tensor_relative(out_pd.numpy(), out_gt, margin=1e-8)
