# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import torch
import numpy as np
from PIL import Image

from utils.preprocessing import Cutout, HSV, VerticalFlip, HorizontalFlip, ToTensor, Pad, ResizeImage


@pytest.fixture(name="data_manager", scope="class")
def data_manager_fixture():
    """
    Manages the data creation for the tests. Since we require only
    a few images and labels, instead of creating them inside each
    function, we abstract that part into this fixture, and pass it
    to the functions instead which allows us to access the data.
    """

    class DataManager:
        def __init__(self):
            self.img1 = np.random.randint(0, 256, [64, 64, 3])
            self.img2 = np.ones([64, 64, 3]) * 255
            self.img3 = np.random.randint(0, 256, [64, 32, 3])
            # We use the same labels for both the images
            self.labels = np.array([[23.0, 0.77, 0.49, 0.34, 0.46]])

    return DataManager()


class TestPreprocessing:
    """
    Test the correctness of each pre-processing class
    """

    def test_cutout(self, data_manager):
        # setting obsercured_pct = 0, forcing label removal
        cutout = Cutout(patch_scales=[1, 1, 0.5, 0.25], obscured_pct=0.0)
        augmented_img, augmented_label = cutout((np.copy(data_manager.img1), np.copy(data_manager.labels)))
        # check whether the unequal area is masked with pixel value between 64 and 191
        cutout_idx = augmented_img != data_manager.img1
        assert np.isin(augmented_img[cutout_idx], np.arange(64, 191)).all()
        # assert empty label
        assert np.array_equal(augmented_label, np.empty((0, 5)))

    def test_hsv(self, data_manager):
        hsv = HSV(h_gain=0.5, s_gain=0.5, v_gain=0.5)
        augmented_img, augmented_label = hsv(
            (Image.fromarray(np.uint8(data_manager.img2)).convert("RGB"), np.copy(data_manager.labels))
        )
        augmented_img = np.ascontiguousarray(augmented_img)

        assert np.array_equal(augmented_label, data_manager.labels)
        # hsv applies the same mapping for every pixel
        assert (augmented_img == augmented_img[0][0]).all()

    def test_horizontal_flip(self, data_manager):
        horizontal_flip = HorizontalFlip()
        augmented_img, augmented_label = horizontal_flip(
            (Image.fromarray(np.uint8(data_manager.img1)).convert("RGB"), np.copy(data_manager.labels))
        )
        augmented_img = np.ascontiguousarray(augmented_img)

        assert np.array_equal(augmented_img, np.fliplr(data_manager.img1))
        assert np.array_equal(augmented_label[..., [0, 2, 3, 4]], data_manager.labels[..., [0, 2, 3, 4]])
        assert np.array_equal(augmented_label[..., 1], 1.0 - data_manager.labels[..., 1])

    def test_vertical_flip(self, data_manager):
        vertical_flip = VerticalFlip()
        augmented_img, augmented_label = vertical_flip(
            (Image.fromarray(np.uint8(data_manager.img1)).convert("RGB"), np.copy(data_manager.labels))
        )
        augmented_img = np.ascontiguousarray(augmented_img)

        assert np.array_equal(augmented_img, np.flipud(data_manager.img1))
        assert np.array_equal(augmented_label[..., [0, 1, 3, 4]], data_manager.labels[..., [0, 1, 3, 4]])
        assert np.array_equal(augmented_label[..., 2], 1.0 - data_manager.labels[..., 2])

    def test_to_tensor(self, data_manager):
        to_tensor = ToTensor(max_bbox_per_scale=90, image_type="half")
        augmented_img, augmented_label = to_tensor((np.copy(data_manager.img1), np.copy(data_manager.labels)))

        assert type(augmented_img) == torch.Tensor
        assert type(augmented_label) == torch.Tensor

    def test_pad(self, data_manager):
        image_size = 128
        pad = Pad(image_size)
        augmented_img, augmented_label = pad(
            (Image.fromarray(np.uint8(data_manager.img1)).convert("RGB"), np.copy(data_manager.labels))
        )
        augmented_img = np.ascontiguousarray(augmented_img)

        pad_w = int(np.round(abs(image_size - data_manager.img1.shape[1]) / 2))
        pad_h = int(np.round(abs(image_size - data_manager.img1.shape[0]) / 2))

        padded_img = np.pad(
            data_manager.img1, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), "constant", constant_values=114
        )

        assert np.array_equal(augmented_img, padded_img)
        assert np.array_equal(augmented_label[:, 0], data_manager.labels[:, 0])
        assert np.array_equal(
            augmented_label[:, 1], (data_manager.labels[:, 1] * data_manager.img1.shape[1] + pad_w) / image_size
        )
        assert np.array_equal(
            augmented_label[:, 2], (data_manager.labels[:, 2] * data_manager.img1.shape[0] + pad_h) / image_size
        )
        assert np.array_equal(
            augmented_label[:, [3, 4]],
            data_manager.labels[:, [3, 4]]
            * np.array([[data_manager.img1.shape[1], data_manager.img1.shape[0]]])
            / image_size,
        )

    def test_resize_image(self, data_manager):
        image_size = 128
        resize_image = ResizeImage(image_size)
        augmented_img, augmented_label = resize_image(
            (Image.fromarray(np.uint8(data_manager.img3)).convert("RGB"), np.copy(data_manager.labels))
        )
        augmented_img = np.ascontiguousarray(augmented_img)

        assert image_size in augmented_img.shape
        assert np.array_equal(augmented_label, data_manager.labels)
