# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import numpy as np
import time
import torch
import pytest

from common.data import features
import logging_util

logger = logging_util.get_basic_logger('test')

CPP = "cpp"
CPP_TEST = "cpp_test"
CPP_ASYNC = "cpp_async"


def add_conf_args():
    """ define the argument parser object """
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch Size")
    parser.add_argument('--steps', type=int, default=5,
                        help="Number of steps")
    parser.add_argument('--task',
                        default = "cpp_async",
                        choices = [CPP, CPP_TEST, CPP_ASYNC],
                        help = "Task")
    return parser


def get_train_feat_proc(conf):
    train_feat_proc = torch.nn.Sequential(
        conf.train_specaugm_kw and features.SpecAugment(optim_level=0, **conf.train_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=0, **conf.train_splicing_kw),
        features.FillPadding(optim_level=0, max_seq_len=conf.max_spec_len_after_stacking),
    )
    return train_feat_proc


def train_feat_proc_torch(conf, audio, audio_lens, txt, txt_lens):
    train_feat_proc = get_train_feat_proc(conf)
    audio_processed = train_feat_proc([audio, audio_lens])

    feats = audio_processed[0]

    start_time = time.time()

    feats = feats.numpy().astype(conf.precision)

    end_time = time.time()
    logger.debug("torch numpy()+fp16 cast time cpp: \t{}".format(end_time - start_time))

    feat_lens = audio_processed[1]
    feat_lens = feat_lens.numpy().astype('int32')

    txt = txt.astype('int32')

    txt_lens = txt_lens.numpy().astype('int32')
    return feats, feat_lens, txt, txt_lens


def data_processor_test(conf):
    if conf.task == CPP_TEST:
        from feat_proc_cpp_test import FeatProc as FeatProcCppTest
    elif conf.task == CPP:
        from feat_proc_cpp import FeatProc as FeatProcCpp
    else:  # CPP_ASYNC in conf.tasks
        from feat_proc_cpp_async import FeatProcAsync as FeatProcCppAsync

    conf.precision = np.float16
    conf.input_seq_len = 32
    conf.target_seq_len = 16
    conf.joint_n_hid = 64
    conf.num_symbols = 64
    conf.batches_per_step = 10
    conf.enc_stack_time_factor = 2
    conf.gradient_accumulation_factor = 10

    np.random.seed(0)

    mel_bands = 80
    max_spec_len_before_stacking = 1965
    conf.max_token_sequence_len = 125
    frame_subsampling = 3
    conf.max_spec_len_after_stacking = round(max_spec_len_before_stacking / frame_subsampling)

    conf.train_specaugm_kw = {
        'freq_masks': 2,
        'min_freq': 0,
        'max_freq': 20,
        'time_masks': 10,
        'min_time': 0,
        'max_time': 0.03,
        'noise_magnitude': 0
    }

    conf.train_splicing_kw = {
        'frame_stacking': 3,
        'frame_subsampling': 3
    }

    torch.manual_seed(0)

    conf.samples_per_step = conf.batch_size * conf.batches_per_step * conf.gradient_accumulation_factor
    synthetic_audio_data = torch.randn(conf.samples_per_step,
                                       mel_bands,
                                       max_spec_len_before_stacking)
    synthetic_audio_lens_data = torch.randint(conf.max_token_sequence_len + 1, max_spec_len_before_stacking,
                                              [conf.samples_per_step],
                                              dtype=torch.int32)
    synthetic_txt_data = torch.randint(0, conf.num_symbols,
                                       [conf.samples_per_step,
                                        conf.max_token_sequence_len],
                                       dtype=torch.int32)
    synthetic_txt_lens_data = torch.randint(conf.max_token_sequence_len // 4, conf.max_token_sequence_len,
                                            [conf.samples_per_step],
                                            dtype=torch.int32)

    torch.manual_seed(0)
    if conf.task == CPP_TEST:
        feat_proc_cpp_test = FeatProcCppTest(conf)
        feat_proc_cpp_test.setRandomSeed(0)
    elif conf.task == CPP:
        feat_proc_cpp = FeatProcCpp(conf)
        feat_proc_cpp.setRandomSeed(0)
    else:  # CPP_ASYNC
        feat_proc_cpp_async = FeatProcCppAsync(conf)
        feat_proc_cpp_async.setRandomSeed(0)

    for step in range(conf.steps):
        results = []
        step_start_time = time.time()
        result = train_feat_proc_torch(conf, synthetic_audio_data, synthetic_audio_lens_data,
                                       synthetic_txt_data.numpy(), synthetic_txt_lens_data)
        assert(result and len(result) == 4)
        feat_proc_time = time.time() - step_start_time

        logger.debug("Feature acquisition time torch: \t{}".format(feat_proc_time))

        results.append(result)

        step_start_time = time.time()
        if conf.task == CPP_TEST:
            result = feat_proc_cpp_test(synthetic_audio_data, synthetic_audio_lens_data,
                                        synthetic_txt_data.numpy(), synthetic_txt_lens_data)
        elif conf.task == CPP:
            result = feat_proc_cpp(synthetic_audio_data, synthetic_audio_lens_data,
                                   synthetic_txt_data.numpy(), synthetic_txt_lens_data)
        else:  # CPP_ASYNC
            feat_proc_cpp_async.submit(synthetic_audio_data, synthetic_audio_lens_data,
                                       synthetic_txt_data.numpy(), synthetic_txt_lens_data)
            result = feat_proc_cpp_async.get()

        assert(result and len(result) == 4)
        feat_proc_time = time.time() - step_start_time

        logger.debug("Feature acquisition time cpp: \t{}".format(feat_proc_time))

        results.append(result)
        res_gold = results[0]
        res_cur = results[-1]
        diff = [res_gold[i] - res_cur[i] for i in range(4)]
        logger.info("diff = {}, {}, {}, {}".format(np.sum(diff[0]), np.sum(diff[1]),
                                                   np.sum(diff[2]), np.sum(diff[3])))
        np.testing.assert_allclose(res_gold[0], res_cur[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_equal(res_gold[1], res_cur[1])
        np.testing.assert_equal(res_gold[2], res_cur[2])
        np.testing.assert_equal(res_gold[3], res_cur[3])

    if conf.task == CPP_ASYNC:
        feat_proc_cpp_async.stop()

"""
pytest entry
"""


@pytest.mark.category1
def test_data_processor():
    class Conf(object):
        pass
    conf = Conf()
    conf.batch_size = 4
    conf.steps = 5
    conf.task = CPP_ASYNC

    data_processor_test(conf)


"""
command-line entry
"""
if __name__ == '__main__':
    parser = add_conf_args()
    conf = parser.parse_args()

    data_processor_test(conf)
