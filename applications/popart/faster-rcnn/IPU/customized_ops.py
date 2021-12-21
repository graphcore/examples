# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import ctypes
import os
import numpy as np
import basic_func as bF


LIB_LOADED = False


def load_lib():
    global LIB_LOADED
    if not LIB_LOADED:
        basedir = os.path.abspath(os.path.dirname(__file__))
        ctypes.cdll.LoadLibrary(
            os.path.join(basedir, "custom_ops/nms/build/libnms_ops.so"))
        ctypes.cdll.LoadLibrary(
            os.path.join(basedir, "custom_ops/roi_align/build/roi_align.so"))
        LIB_LOADED = True


def get_valid_area_mask(boxes):
    # input boxes: 1,n,4
    # output: mask: n,1
    ws = boxes[:, :, 2] - boxes[:, :, 0]
    hs = boxes[:, :, 3] - boxes[:, :, 1]
    areas = ws * hs
    valid_flags = bF.greater(
        areas, bF.constant(np.asarray(0.0,
                                      dtype=bF.mappin_gc2npy[areas.dtype])))
    valid_mask = bF.cast(valid_flags, target_type=boxes.dtype)
    return bF.transpose(valid_mask, [1, 0])


def nms(input_scores,
        input_boxes,
        threshold=0.7,
        numDetections=300,
        score_threshold=None,
        debugContext=''):
    load_lib()
    input_scores = input_scores.cast('FLOAT')
    input_boxes = input_boxes.cast('FLOAT')
    valid_area_mask = bF.transpose(get_valid_area_mask(input_boxes),
                                   [1, 0])  # 1,n
    input_scores = input_scores + 1e-6  # if score==0, proposals will be ignored
    local_input_scores = bF.identity(input_scores * valid_area_mask,
                                     debugContext=debugContext).detach()
    local_input_boxes = bF.identity(input_boxes,
                                    debugContext=debugContext).detach()

    if local_input_scores.shape.ndims == 1:
        local_input_scores = local_input_scores.unsqueeze(0)
    if local_input_boxes.shape.ndims == 2:
        local_input_boxes = local_input_boxes.unsqueeze(0)
    assert local_input_boxes.pureShape[0] == 1, 'only implemented batch=1'
    if score_threshold is not None:
        assert isinstance(score_threshold, float)
        local_mask = bF.greater(
            local_input_scores,
            bF.to_tensor(score_threshold, dtype=local_input_scores.dtype))
        local_mask = bF.cast(local_mask, target_type=local_input_scores.dtype)
        local_input_scores = local_input_scores * local_mask
    with bF.name_scope("nms"):
        out = bF.get_builder().customOp(opName="nms",
                                        opVersion=1,
                                        domain="ai.graphcore",
                                        inputs=[
                                            local_input_scores.getIpuIndex(),
                                            local_input_boxes.getIpuIndex()
                                        ],
                                        attributes={
                                            "threshold": threshold,
                                            "numDetections": numDetections
                                        },
                                        numOutputs=3,
                                        name="nmsCustomOp")
        #
        _, output_boxes, output_keep = out[0], bF.TTensor(out[1]), bF.TTensor(
            out[2])
        targetType = input_scores.dtype
        roiKeeps_flag = bF.cast(bF.greater(
            output_keep, bF.constant(np.asarray(-1, dtype=np.int32))),
            target_type='INT32')
        num_valids = bF.reduceSum(roiKeeps_flag, axes=[1])
        roiKeeps_flag = bF.cast(roiKeeps_flag, target_type=targetType)
        roiKeeps_flag = bF.unsqueeze(roiKeeps_flag, [-1])
        output_boxes = bF.mul([output_boxes, roiKeeps_flag])
    return output_boxes, output_keep, num_valids


def roi_align(bottom_data,
              bottom_rois,
              spatial_scale=1 / 16.0,
              num_rois=300,
              aligned_height=7,
              aligned_width=7,
              fp16_on=None):
    """roi_align implements."""

    load_lib()
    assert isinstance(aligned_height, int) and isinstance(aligned_width, int), 'they should be int or IndexError: map::at will raised'
    cast_flag, bottom_data, fp16_on = bF.deduce_half(bottom_data, fp16_on)
    if fp16_on:
        bottom_rois = bottom_rois.cast('FLOAT16')
    else:
        bottom_rois = bottom_rois.cast('FLOAT')

    if fp16_on:
        raise NotImplementedError('maybe not implemented')

    # same as detectron2 roi_align version2(aligned=True and sampling_ratio=1)
    batch_size, channels, height, width = bottom_data.pureShape
    with bF.name_scope("roiAlign"):
        out = bF.get_builder().customOp(
            opName="roiAlign",
            opVersion=1,
            domain="ai.graphcore",
            inputs=[bottom_data.getIpuIndex(),
                    bottom_rois.getIpuIndex()],
            attributes={
                "spatial_scale": spatial_scale,
                "batch_size": batch_size,
                "num_rois": num_rois,
                "height": height,
                "width": width,
                "channels": channels,
                "aligned_height": aligned_height,
                "aligned_width": aligned_width
            },
            numOutputs=1)
    result = bF.TTensor(out[0])

    if cast_flag:
        result = result.cast(cast_flag)

    return result
