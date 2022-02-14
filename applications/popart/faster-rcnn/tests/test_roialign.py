# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../IPU'))
from torchvision.ops import RoIAlign as torch_roialign
import torch
import numpy as np
from ipu_tensor import gcop
from gc_session import Session, add_aux_vars_to_tensor

# make test data for ROI-Align
IMAGE_SIZE = 800
BOX_SIZE_RANGE = [5, 500]
BOX_NUM = 256
DTYPE = np.float32
CHANNELS = 1024
STRIDE = 16
ALIGNED_SIZE = 14
FEATURE_SIZE = int(IMAGE_SIZE/STRIDE)


# define helper function
def my_test_func(feat, box, pt_on_output, output_shape, spatial_scale, aligned=True, sample_rate=1):
    # feat: featmap 2d array,shape is (h,w)
    # box: (x1,y1,x2,y2)
    # pt_on_output: (x,y)
    # output shape: (h,w)
    # spatial_scale, aligned, sample_rate same as detectron2 roi_align
    # only implement aligned=True and smaple_rate=1
    assert aligned and sample_rate == 1
    mapped_box = np.asarray(box)*spatial_scale
    start_x_on_feat, start_y_on_feat = mapped_box[0:2]-0.5
    box_shape = np.array([mapped_box[3]-mapped_box[1],
                         mapped_box[2]-mapped_box[0]])
    grid_h, grid_w = np.array(box_shape)/np.asarray(output_shape)
    grid_idx, grid_idy = pt_on_output
    assert grid_idx < output_shape[1] and grid_idx >= 0
    assert grid_idy < output_shape[0] and grid_idy >= 0
    target_x = start_x_on_feat+grid_w*(0.5+grid_idx)
    target_y = start_y_on_feat+grid_h*(0.5+grid_idy)
    target_x = max(target_x, 0)
    target_y = max(target_y, 0)
    xl, xh, yl, yh = int(target_x), int(
        target_x+1), int(target_y), int(target_y+1)
    top_left = feat[yl, xl]
    top_right = feat[yl, xh]
    bot_left = feat[yh, xl]
    bot_right = feat[yh, xh]
    w_top_left = (xh-target_x)*(yh-target_y)
    w_top_right = (target_x-xl)*(yh-target_y)
    w_bot_right = (target_x-xl)*(target_y-yl)
    w_bot_left = (xh-target_x)*(target_y-yl)
    result = top_left*w_top_left+top_right*w_top_right + \
        bot_left*w_bot_left+bot_right*w_bot_right
    return result


def test_roialign():
    box_centers = np.random.rand(1, BOX_NUM, 2) * np.asarray(IMAGE_SIZE)
    box_whs = np.random.rand(
        1, BOX_NUM, 2) * (BOX_SIZE_RANGE[1] - BOX_SIZE_RANGE[0]) + BOX_SIZE_RANGE[0]
    start_xys = box_centers - box_whs / 2
    end_xys = box_centers + box_whs / 2
    boxes = np.concatenate([start_xys, end_xys], axis=2).astype(DTYPE)
    features = np.random.rand(
        1, CHANNELS, FEATURE_SIZE, FEATURE_SIZE).astype(DTYPE)
    features = features/(features**2).sum()

    # define helper function
    def clip_boxes(boxes, im_info):
        with gcop.variable_scope("clip_boxes"):
            x1 = gcop.clip_by_value(boxes[:, :, 0],
                                    clip_value_min=0,
                                    clip_value_max=im_info[1] -
                                    1).unsqueeze(-1)
            y1 = gcop.clip_by_value(boxes[:, :, 1],
                                    clip_value_min=0,
                                    clip_value_max=im_info[0] -
                                    1).unsqueeze(-1)
            x2 = gcop.clip_by_value(boxes[:, :, 2],
                                    clip_value_min=0,
                                    clip_value_max=im_info[1] -
                                    1).unsqueeze(-1)
            y2 = gcop.clip_by_value(boxes[:, :, 3],
                                    clip_value_min=0,
                                    clip_value_max=im_info[0] -
                                    1).unsqueeze(-1)

            boxes = gcop.concat([x1, y1, x2, y2], 2)
        return boxes

    # build net
    with gcop.device("0"):
        input_boxes = gcop.placeholder(shape=[1, BOX_NUM, 4], dtype=DTYPE)
        input_features = gcop.placeholder(
            shape=[1, CHANNELS, FEATURE_SIZE, FEATURE_SIZE], dtype=DTYPE)
        clipped_boxes = clip_boxes(input_boxes, [IMAGE_SIZE, IMAGE_SIZE])
        areas = (clipped_boxes[:, :, 2] - clipped_boxes[:, :, 0]) * \
            (clipped_boxes[:, :, 3] - clipped_boxes[:, :, 1])
        pooled_feat = gcop.cOps.roi_align(input_features,
                                          clipped_boxes,
                                          spatial_scale=1/STRIDE,
                                          num_rois=BOX_NUM,
                                          aligned_height=ALIGNED_SIZE,
                                          aligned_width=ALIGNED_SIZE)
        loss = gcop.reduce_mean(pooled_feat)
        # result = builder.aiOnnx.mul([result,builder.aiOnnx.constant(np.array(0.0,dtype=np.float32))])
        loss = gcop.abs(loss)
        add_aux_vars_to_tensor(input_features, ['Gradient___'])

    # make session
    gcop.safe_mode_on()
    optimizer = gcop.bF.SGD()
    sess = Session([pooled_feat, clipped_boxes, input_features,
                   areas], optimizer=optimizer, loss=loss)

    # run session
    feed_dict = {input_boxes: boxes, input_features: features}
    pooled_feat, clipped_boxes, input_features, areas = sess.run(feed_dict)

    # run reference
    torch_feat = torch.from_numpy(features)
    torch_feat.requires_grad = True
    torch_feat.retain_grad()
    torch_rois = torch.from_numpy(clipped_boxes.data)
    alignNet = torch_roialign(output_size=(
        ALIGNED_SIZE, ALIGNED_SIZE), spatial_scale=1/STRIDE, sampling_ratio=1, aligned=True)
    alignNet.train()
    torch_result = alignNet(torch_feat, [ele for ele in torch_rois])
    loss = torch.nn.L1Loss()
    loss(torch_result.mean(), torch.tensor(0)).backward()

    # compare result
    # annotated codes blew shows detailed calculation process of ROI-Align
    # local_feat = features[0,0]
    # local_box = clipped_boxes.data[0,0]
    # pt_on_output = [0,0]
    # output_shape = (7,7)
    # local_spatial_scale = 1/16
    # local_result = my_test_func(local_feat,local_box,pt_on_output,output_shape,local_spatial_scale)
    np.testing.assert_allclose(
        torch_result.detach().numpy(), pooled_feat.data[0], rtol=1e-2, atol=1e-5)
    np.testing.assert_allclose(
        torch_feat.grad.detach().numpy(), input_features.grad, rtol=1e-3, atol=1e-6)
