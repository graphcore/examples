# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import time
import logging_util
from custom_op_utils import load_custom_lib

logger = logging_util.get_basic_logger('feat_proc_cpp')


class FeatProc:
    """
    This class is similar to FeatProc in feat_proc_cpp.py,
    but it breaks feature processing into steps to ,make it easier to test in Python
    """
    def __init__(self, conf):
        libc = load_custom_lib("feat_proc")

        c_contig_flag = "C_CONTIGUOUS"

        self.fun_stackSubsampleSize = libc.stackSubsampleSize
        self.fun_stackSubsampleSize.restype = ctypes.c_uint32
        self.fun_stackSubsampleSize.argtypes = [ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                                ctypes.c_uint32,  # stacking
                                                ctypes.c_uint32]  # maxSeqLen

        self.fun_stackSubsample = libc.stackSubsample
        self.fun_stackSubsample.restype = None
        self.fun_stackSubsample.argtypes = [ndpointer(ctypes.c_float, flags=c_contig_flag),  # featsIn
                                            ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                            ndpointer(flags=c_contig_flag),  # featsOut
                                            ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsOutShape
                                            ndpointer(ctypes.c_int32, flags=c_contig_flag),  # featLens
                                            ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featLensShape
                                            ctypes.c_uint32,  # stacking
                                            ctypes.c_uint32,  # subsampling
                                            ctypes.c_uint32]  # maxSeqLen

        self.fun_specAugment = libc.specAugment
        self.fun_specAugment.restype = None
        self.fun_specAugment.argtypes = [ndpointer(ctypes.c_float, flags=c_contig_flag),  # feats
                                         ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                         ndpointer(ctypes.c_int32, flags=c_contig_flag),  # featLens
                                         ctypes.c_uint32,  # freqMasks
                                         ctypes.c_uint32,  # minFreq
                                         ctypes.c_uint32,  # maxFreq
                                         ctypes.c_float,  # timeMasks
                                         ctypes.c_float,  # minTime
                                         ctypes.c_float]  # maxTime

        self.fun_setRandomSeed = libc.setRandomSeed
        self.fun_setRandomSeed.restype = None
        self.fun_setRandomSeed.argtypes = [ctypes.c_uint64]  # seed

        self.stacking = conf.train_splicing_kw["frame_stacking"]
        self.subsampling = conf.train_splicing_kw["frame_subsampling"]

        self.freqMasks = conf.train_specaugm_kw["freq_masks"]
        self.minFreq = conf.train_specaugm_kw["min_freq"]
        self.maxFreq = conf.train_specaugm_kw["max_freq"]

        self.timeMasks = float(conf.train_specaugm_kw["time_masks"])
        self.minTime = float(conf.train_specaugm_kw["min_time"])
        self.maxTime = float(conf.train_specaugm_kw["max_time"])

        self.maxSeqLen = conf.max_spec_len_after_stacking


    def __call__(self, audio, audio_lens, txt, txt_lens):

        start_time = time.time()

        audio = audio.numpy()
        # Unconditional conversion numpy.astype() is too slow
        # We rely on a fact that audio is already in float32 format
        assert(audio.dtype == np.float32)
        audio_lens = audio_lens.numpy().astype(np.int32)
        txt = txt.astype(np.int32)
        txt_lens = txt_lens.numpy().astype(np.int32)

        end_time = time.time()
        logger.debug("Preprocessing time cpp: \t{}".format(end_time - start_time))
        start_time = end_time

        if self.freqMasks:
            feats, feat_lens = self.spec_augment(audio, audio_lens)
        else:
            feats, feat_lens = (audio, audio_lens, audio)

        end_time = time.time()
        logger.debug("Spec augment time cpp: \t{}".format(end_time - start_time))
        start_time = end_time

        feats, feat_lens = self.stack_subsample_frames(feats, feat_lens)

        end_time = time.time()
        logger.debug("Subsample time cpp: \t{}".format(end_time - start_time))

        return feats, feat_lens, txt, txt_lens


    def spec_augment(self, x, x_lens):
        x_shape = np.array(x.shape, dtype = np.uint32)

        self.fun_specAugment(x, x_shape, x_lens, self.freqMasks, self.minFreq, self.maxFreq, self.timeMasks, self.minTime, self.maxTime)

        return x, x_lens


    def stack_subsample_frames(self, x, x_lens):
        """ Stacks frames together across feature dim, and then subsamples

        input is batch_size, feature_dim, num_frames
        output is batch_size, feature_dim * stacking, num_frames / subsampling

        """
        x_shape = np.array(x.shape, dtype = np.uint32)
        x_lens_shape = np.array(x_lens.shape, dtype = np.uint32)

        x_out_size = self.fun_stackSubsampleSize(x_shape, self.stacking, self.maxSeqLen)

        x_out = np.zeros((x_out_size,), dtype = np.float16)
        x_out_shape = np.zeros((3,), dtype = np.uint32)
        self.fun_stackSubsample(x, x_shape, x_out, x_out_shape, x_lens, x_lens_shape, self.stacking, self.subsampling, self.maxSeqLen)
        x = x_out.reshape(x_out_shape)

        return x, x_lens


    def setRandomSeed(self, seed):
        self.fun_setRandomSeed(seed)
