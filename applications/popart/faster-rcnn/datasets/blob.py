# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# This file has been modified by Graphcore Ltd.
"""Blob helper functions."""

import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob_eval(im, pixel_means, pixel_std, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im /= pixel_std
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_max)
    # Prevent the biggest axis from being more than MAX_SIZE
    im = cv2.resize(im,
                    None,
                    None,
                    fx=im_scale,
                    fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    pad_im = np.zeros((target_size, target_size, 3), dtype=np.float32)
    h, w, _ = im.shape
    pad_im[:h, :w, :] = im

    return pad_im, im_scale


def prep_im_for_blob(im, pixel_means, pixel_std, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im /= pixel_std
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    im = cv2.resize(im,
                    None,
                    None,
                    fx=im_scale,
                    fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
