# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import time
from collections import deque
import logging_util
from custom_op_utils import load_custom_lib

logger = logging_util.get_basic_logger("feat_proc_cpp_async")


class FeatProcAsync:
    """ Wrapper class for calling C++ feature processing functions asynchronously """
    def __init__(self, conf):
        libc = load_custom_lib("feat_proc")

        c_contig_flag = "C_CONTIGUOUS"

        self.fun_featsOutSize = libc.featsOutSize
        self.fun_featsOutSize.restype = ctypes.c_uint32
        self.fun_featsOutSize.argtypes = [ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                          ctypes.c_uint32,  # stacking
                                          ctypes.c_uint32]  # maxSeqLen

        self.fun_init = libc.featProcInit
        self.fun_init.restype = None
        self.fun_init.argtypes = [ctypes.c_bool,  # doSpecaugm
                                  ctypes.c_uint32,  # freqMasks
                                  ctypes.c_uint32,  # minFreq
                                  ctypes.c_uint32,  # maxFreq
                                  ctypes.c_float,  # timeMasks
                                  ctypes.c_float,  # minTime
                                  ctypes.c_float,  # maxTime
                                  ctypes.c_uint32,  # stacking
                                  ctypes.c_uint32,  # subsampling
                                  ctypes.c_uint32]  # maxSeqLen

        self.fun_submit = libc.featProcSubmit
        self.fun_submit.restype = None
        self.fun_submit.argtypes = [ctypes.c_int64,  # tag
                                    ndpointer(ctypes.c_float, flags=c_contig_flag),  # featsIn
                                    ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsInShape
                                    ndpointer(flags=c_contig_flag),  # featsOut
                                    ndpointer(ctypes.c_uint32, flags=c_contig_flag),  # featsOutShape
                                    ndpointer(ctypes.c_int32, flags=c_contig_flag),  # featLens
                                    ndpointer(ctypes.c_uint32, flags=c_contig_flag)]  # featLensShape

        self.fun_get = libc.featProcGet
        self.fun_get.restype = ctypes.c_int64
        self.fun_get.argtypes = []

        self.fun_stop = libc.featProcStop
        self.fun_stop.restype = None
        self.fun_stop.argtypes = []

        self.fun_current_queue_len = libc.featProcCurrentQueueLen
        self.fun_current_queue_len.restype = ctypes.c_uint32
        self.fun_current_queue_len.argtypes = []

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

        self.tag = np.int64(0)
        self.args_queue = deque()

        self.fun_init(self.doSpecaugm, self.freqMasks, self.minFreq, self.maxFreq, self.timeMasks, self.minTime, self.maxTime,
                      self.stacking, self.subsampling, self.maxSeqLen)


    def submit(self, feats, feat_lens, txt, txt_lens):
        start_time = time.time()

        feats = feats.numpy()
        # Unconditional conversion numpy.astype() is too slow
        # We rely on a fact that feats is already in float32 format
        if feats.dtype != np.dtype("float32"):
            raise Exception("Unexpected features type {}. It should be 'float32'".format(feats.dtype))

        feat_lens = feat_lens.numpy().astype(np.int32)
        txt = txt.astype(np.int32)
        txt_lens = txt_lens.numpy().astype(np.int32)

        feats_shape = np.array(feats.shape, dtype = np.uint32)
        feat_lens_shape = np.array(feat_lens.shape, dtype = np.uint32)

        feats_out_size = self.fun_featsOutSize(feats_shape, self.stacking, self.maxSeqLen)

        feats_out = np.zeros((feats_out_size,), dtype = np.float16)
        feats_out_shape = np.zeros((3,), dtype = np.uint32)
        self.fun_submit(self.tag, feats, feats_shape, feats_out, feats_out_shape, feat_lens, feat_lens_shape)

        self.args_queue.append((self.tag, feats, feats_shape, feats_out, feats_out_shape, feat_lens, feat_lens_shape, txt, txt_lens))
        self.tag = self.tag + 1

        end_time = time.time()
        logger.debug("Feature suibmission time cpp: \t{}".format(end_time - start_time))


    def get(self):
        tag_out = self.fun_get()
        if len(self.args_queue) == 0:
            raise Exception("Feature processing queue is empty!")
        (tag, feats, feats_shape, feats_out, feats_out_shape, feat_lens, feat_lens_shape, txt, txt_lens) = self.args_queue[0]
        if (tag_out != tag):
            raise Exception("Unexpected value of the feature processing tag!")

        feats = feats_out.reshape(feats_out_shape)

        self.args_queue.popleft()

        return feats, feat_lens, txt, txt_lens


    def current_queue_len(self):
        return self.fun_current_queue_len()


    def stop(self):
        self.fun_stop()


    def setRandomSeed(self, seed):
        self.fun_setRandomSeed(seed)


class AsyncDataProcessor(FeatProcAsync):
    """ Class that provides interface for asynchronous feature processing during training """
    def __init__(self, conf):
        super(AsyncDataProcessor, self).__init__(conf)

    def submit_data(self):
        try:
            data = next(self.data_iterator)
            assert(len(data) == 4)
            audio, audio_lens, txt, txt_lens = data

            self.submit(audio, audio_lens, txt, txt_lens)
        except StopIteration:
            # All data for epoch has been submitted to processing
            # restart with new epoch next
            pass

    def set_iterator(self, data_iterator):
        self.data_iterator = data_iterator
