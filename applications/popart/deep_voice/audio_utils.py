# Copyright 2020 Graphcore Ltd.
from scipy import signal
import numpy as np

REF_DB = 20.0
MIN_DB = -100.0


def preemphasis(x, coef=0.97):
    """ Pre-emphasis - apply filter to boost frequency range susceptible to noise """
    b = np.array([1., -coef], x.dtype)
    a = np.array([1.], x.dtype)
    return signal.lfilter(b, a, x)


def inv_preemphasis(x, coef=0.97):
    """ Inverse of pre-emphasis """
    b = np.array([1.], x.dtype)
    a = np.array([1., -coef], x.dtype)
    return signal.lfilter(b, a, x)
