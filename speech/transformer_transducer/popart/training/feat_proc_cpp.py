# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import time
import logging_util
from custom_op_utils import load_custom_lib

logger = logging_util.get_basic_logger('feat_proc_cpp')


class FeatProc:
    """ Wrapper class for C++ feature processing functions """
    def __init__(self, conf):
        libc = load_custom_lib("feat_proc")

        c_contig_flag = "C_CONTIGUOUS"

        self.fun_featsOutSize = libc.featsOutSize
        self.fun_featsOutSize.restype = ctypes.c_uint32
        self.fun_featsOutSize.argtypes = [ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                          ctypes.c_uint32,  # stacking
                                          ctypes.c_uint32]  # maxSeqLen

        self.fun_featProcess = libc.featProcess
        self.fun_featProcess.restype = None
        self.fun_featProcess.argtypes = [ndpointer(ctypes.c_float, flags=c_contig_flag),  # featsIn
                                         ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                         ndpointer(flags=c_contig_flag),  # featsOut
                                         ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsOutShape
                                         ndpointer(ctypes.c_int32, flags=c_contig_flag),  # featLens
                                         ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featLensShape
                                         ctypes.c_bool,  # doSpecaugm
                                         ctypes.c_uint32,  # freqMasks
                                         ctypes.c_uint32,  # minFreq
                                         ctypes.c_uint32,  # maxFreq
                                         ctypes.c_float,  # timeMasks
                                         ctypes.c_float,  # minTime
                                         ctypes.c_float,  # maxTime
                                         ctypes.c_uint32,  # stacking
                                         ctypes.c_uint32,  # subsampling
                                         ctypes.c_uint32]  # maxSeqLen

        self.fun_setRandomSeed = libc.setRandomSeed
        self.fun_setRandomSeed.restype = None
        self.fun_setRandomSeed.argtypes = [ctypes.c_uint64]  # seed

        self.stacking = conf.train_splicing_kw["frame_stacking"]
        self.subsampling = conf.train_splicing_kw["frame_subsampling"]

        self.doSpecaugm = True if conf.train_specaugm_kw else False

        self.freqMasks = conf.train_specaugm_kw["freq_masks"]
        self.minFreq = conf.train_specaugm_kw["min_freq"]
        self.maxFreq = conf.train_specaugm_kw["max_freq"]

        self.timeMasks = float(conf.train_specaugm_kw["time_masks"])
        self.minTime = float(conf.train_specaugm_kw["min_time"])
        self.maxTime = float(conf.train_specaugm_kw["max_time"])

        self.maxSeqLen = conf.max_spec_len_after_stacking


    def __call__(self, feats, feat_lens, txt, txt_lens):

        start_time = time.time()

        feats = feats.numpy()
        # Unconditional conversion numpy.astype() is too slow
        # We rely on a fact that feats is already in float32 format
        if feats.dtype != np.dtype("float32"):
            raise Exception("Unexpected feats type {}. It should be 'float32'".format(feats.dtype))

        feat_lens = feat_lens.numpy().astype(np.int32)
        txt = txt.astype(np.int32)
        txt_lens = txt_lens.numpy().astype(np.int32)

        end_time = time.time()
        logger.debug("Preprocessing time cpp: \t{}".format(end_time - start_time))
        start_time = end_time

        feats_shape = np.array(feats.shape, dtype = np.uint32)
        feat_lens_shape = np.array(feat_lens.shape, dtype = np.uint32)

        feats_out_size = self.fun_featsOutSize(feats_shape, self.stacking, self.maxSeqLen)

        feats_out = np.zeros((feats_out_size,), dtype = np.float16)
        feats_out_shape = np.zeros((3,), dtype = np.uint32)
        self.fun_featProcess(feats, feats_shape, feats_out, feats_out_shape, feat_lens, feat_lens_shape,
                             self.doSpecaugm, self.freqMasks, self.minFreq, self.maxFreq, self.timeMasks, self.minTime, self.maxTime,
                             self.stacking, self.subsampling, self.maxSeqLen)
        feats = feats_out.reshape(feats_out_shape)

        end_time = time.time()
        logger.debug("Processing time cpp: \t{}".format(end_time - start_time))

        return feats, feat_lens, txt, txt_lens


    def setRandomSeed(self, seed):
        self.fun_setRandomSeed(seed)
