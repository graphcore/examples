# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import numpy as np
from PIL import Image

from utils.preprocessing import Cutout, HSV


class TestPreprocessing:
    """
    Test the correctness of each pre-processing class
    """

    @pytest.mark.category1
    def test_cutout(self):
        test_img = np.random.randint(0, 256, [64, 64, 3])
        label = [[23.0,  0.77,  0.49,  0.34,  0.46]]

        # setting obsercured_pct = 0, forcing label removal
        cutout = Cutout(patch_scales=[1, 1, 0.5, 0.25], obscured_pct=0.0)
        augmented_img, augmented_label = cutout((np.copy(test_img), np.copy(label)))

        # check whether the unequal area is masked with pixel value between 64 and 191
        cutout_idx = augmented_img != test_img
        assert np.isin(augmented_img[cutout_idx], np.arange(64, 191)).all()
        # assert empty label
        assert np.array_equal(augmented_label, np.empty((0, 5)))

    @pytest.mark.category1
    def test_hsv(self):
        img = np.ones([64, 64, 3]) * 255
        label = [[23.0,  0.77,  0.49,  0.34,  0.46]]

        hsv = HSV(h_gain=.5, s_gain=.5, v_gain=.5)
        augmented_img, augmented_label = hsv((Image.fromarray(np.uint8(img)).convert('RGB'), np.copy(label)))
        augmented_img = np.ascontiguousarray(augmented_img)

        assert np.array_equal(augmented_label, label)
        # hsv applies the same mapping for every pixel
        assert (augmented_img == augmented_img[0][0]).all()
