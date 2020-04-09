# Copyright 2019 Graphcore Ltd.

'''Unit tests for spectral normalization initialisation'''

from spectral_norm_init import spectral_normalization, spectral_normalization_QKV

from scipy.stats import truncnorm
import numpy as np
import pytest


@pytest.fixture
def init_data():

    np.random.seed(1879)
    shape = (3, 3)
    mean = 0
    std_dev = 0.02
    dtype = np.float16

    data = truncnorm.rvs(-2, 2, loc=mean,
                         scale=std_dev, size=np.prod(shape))

    return data.reshape(shape).astype(dtype)


def test_spectral_norm(init_data):

    data_normalized = spectral_normalization(init_data)

    output = np.array([[0.3635, 0.783, -0.1306],
                       [0.6973, -0.142,   0.322],
                       [-0.261, -0.1261, -0.6157]], dtype=np.float16)

    assert not np.allclose(init_data, output)
    assert np.allclose(data_normalized, output)


def test_spectral_norm_QKV(init_data):

    data_normalized_QKV = spectral_normalization_QKV(init_data)

    output = np.array([[0.4387,  0.9717, -0.1848],
                       [0.842, -0.1763,   0.4553],
                       [-0.315, -0.1565, -0.871]], dtype=np.float16)

    assert not np.allclose(init_data, output)
    assert np.allclose(data_normalized_QKV, output)
