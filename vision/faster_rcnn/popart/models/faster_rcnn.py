# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Build the faster R CNN model.
"""
import numpy as np
import json
from layer.rpn import RPN
from layer.resnet import Resnet
from layer.tpu_proposal_target_layer import ProposalTargetLayer
from config import cfg
from utils import logger
import os
from IPU.ipu_tensor import gcop
import math
from layer.base import BaseModel, smooth_l1_loss, get_valid_area_mask, roi_align
from .stage_configs import get_stage_configs


if logger.GLOBAL_LOGGER is not None:
    print = logger.GLOBAL_LOGGER.log_str


class MovingVar:
    def __init__(self, name, rate=0.99, val=None, athrd=float('inf')):
        self.name = name
        self.var = val
        self.rate = rate
        self.athrd = athrd

    def update(self, var):
        if var > self.athrd:
            return
        if self.var is None:
            self.var = var
        else:
            self.var = self.var * self.rate + var * (1 - self.rate)


class FasterRcnn(BaseModel):
    """The Faster-rcnn model on IPU with Popart.
    Paper reference:
        https://arxiv.org/abs/1506.01497
    Official code:
        https://github.com/rbgirshick/py-faster-rcnn
    """

    def __init__(
        self,
        classes=[1] * 21,
        fp16_on=False,
        input_im_shape=[1, 3, 480, 320],
        input_box_num=None,
        training=False,
    ):
        super().__init__(fp16_on=fp16_on, training=training)
        self.input_im_shape = input_im_shape
        self.input_box_num = input_box_num

        fix_Resnet_BN = cfg.TRAIN.RESNET.FIXED_BN if training else True
        self.backbone = Resnet(
            classes=classes,
            fp16_on=fp16_on,
            training=training,
            fix_bn=fix_Resnet_BN,
            network_type=cfg.TRAIN.RESNET.NETWORK_TYPE,
            fix_blocks=cfg.TRAIN.RESNET.FIXED_BLOCKS)

        self.rpn = RPN(
            classes=classes,
            fp16_on=fp16_on,
            training=training,
            rpn_channel=cfg.MODEL.RPN_CHANNEL,
            input_size=[self.input_im_shape[3], self.input_im_shape[2]])

        gcop.set_weight_fp16(cfg.WEIGHT_FP16)

        if self.training:
            self.roialign_fp16 = cfg.TRAIN.ROI_ALIGN_FP16
        else:
            self.roialign_fp16 = cfg.TEST.ROI_ALIGN_FP16

        if self.training:
            self.rcnn_proposal_target = ProposalTargetLayer(
                fp16_on=False,
                positive_fraction=cfg.TRAIN.RPN_OUT_FG_FRACTION,
                roi_thrd=cfg.TRAIN.ROI_THRD,
                batch_size_per_im=cfg.TRAIN.RPN_BATCHSIZE,
                num_classes=len(classes),
            )

            self.input_im = self.add_input(self.input_im_shape,
                                           dtype=gcop.float32)
            self.gt_boxes = self.add_input([1, self.input_box_num, 5],
                                           dtype=gcop.float32)
            anchor_nums = len(cfg.ANCHOR_SCALES)*len(cfg.ANCHOR_RATIOS)
            rpn_feature_stride = cfg.FEAT_STRIDE
            rpn_feature_h = math.ceil(self.input_im_shape[2] /
                                      rpn_feature_stride)
            rpn_feature_w = math.ceil(self.input_im_shape[3] /
                                      rpn_feature_stride)
            num_labels_rpn = rpn_feature_h * rpn_feature_w * \
                anchor_nums  # num of points x num of anchors
            rpn_label = self.add_input(
                [1, num_labels_rpn],
                dtype=np.uint32)
            rpn_keep = self.add_input(
                [1, cfg.TRAIN.RPN_BATCHSIZE],
                dtype=np.uint32)  # unify first dimension for GA
            rpn_bbox_targets = self.add_input(
                [1, rpn_feature_h, rpn_feature_w, 4*anchor_nums], dtype=gcop.float32)
            rpn_bbox_inside_weights = self.add_input(
                [1, rpn_feature_h, rpn_feature_w, 4*anchor_nums], dtype=gcop.float32)
            rpn_bbox_outside_weights = self.add_input(
                [1, rpn_feature_h, rpn_feature_w, 4*anchor_nums], dtype=gcop.float32)

            self.rpn_data = [
                rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights]
        else:
            self.input_im = self.add_input(self.input_im_shape,
                                           dtype=gcop.float32)

        ipu_sets = cfg.TRAIN.SET_PIPELINE_MANUALLY if training else cfg.TEST.SET_PIPELINE_MANUALLY
        self.stage_configs = get_stage_configs(
            ipu_set=ipu_sets, train=training, fp16=fp16_on)
        print('set ipu configs:', str(self.stage_configs))

    def bulid_graph(self, ):
        if self.training:
            self.__build_graph_train_resnet__()
        else:
            self.__build_graph_inference__()

    # build graph
    def __build_graph_inference__(self):
        # process:
        # 1. backbone(x) ---> Feature Map
        # 2. rpn ---> Rois
        # 3. roi_align ---> pooled_feat
        # 4. head2tail
        # 5. bboxoffset and claasification

        with gcop.device(self.stage_configs[0]):
            x = self.input_im
            if cfg.FLOAT16_ON and cfg.FP16_EARLY:
                x = x.cast(gcop.float16)
            x = self.backbone.init_block(x)
            x = self.backbone.layerX(x, 0)
            if cfg.FLOAT16_ON and not cfg.FP16_EARLY:
                x = x.cast(gcop.float16)
            x = self.backbone.layerX(x, 1)
            base_feat = self.backbone.layerX(x, 2)

            x = rpn_out = self.rpn.forward(base_feat,
                                           im_info=np.asarray([
                                               self.input_im_shape[2],
                                               self.input_im_shape[3]
                                           ]).astype(np.int32),
                                           stage_configs=self.stage_configs[1])

        with gcop.device(self.stage_configs[2]):
            valid_area_mask = get_valid_area_mask(x[0])
            self.add_output('valid_area_mask', valid_area_mask)

            pooled_feat = roi_align(
                base_feat,
                x[0],
                1 / cfg.FEAT_STRIDE,
                cfg.TEST.RPN_POST_NMS_TOP_N,
                fp16_on=self.roialign_fp16,
                aligned_height=cfg.MODEL.ALIGNED_HEIGHT,
                aligned_width=cfg.MODEL.ALIGNED_WIDTH)

            pooled_feat = pooled_feat.squeeze(0)

        pooled_feat = self.backbone.head_to_tail(
            pooled_feat, self.stage_configs[3])

        with gcop.device(self.stage_configs[4]):
            B, C, H, W = pooled_feat.shape.as_list()
            pooled_feat = pooled_feat.cast(gcop.float32)
            pooled_feat = gcop.reduce_mean(pooled_feat,
                                           [2, 3])
            cls_score, bbox_pred = self.backbone.cls_reg_head(
                pooled_feat)
            cls_prob = gcop.nn.softmax(cls_score, axis=1)

            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                bbox_pred = bbox_pred.reshape([bbox_pred.shape[0], -1, 4])
                bbox_pred = bbox_pred * gcop.constant(
                    np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=self.dtype))
                bbox_pred = bbox_pred + gcop.constant(
                    np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=self.dtype))
                bbox_pred = bbox_pred.reshape([bbox_pred.shape[0], -1])

        self.add_output('cls_prob', cls_prob)
        self.add_output('bbox_pred', bbox_pred)
        self.add_output('fixed_length_roi', rpn_out[0])
        self.add_output('roi_keeps', rpn_out[1])

    def __build_graph_train_resnet__(self):
        x = self.input_im
        gt_boxes = self.gt_boxes

        with gcop.device(self.stage_configs[0]):
            if cfg.FLOAT16_ON and cfg.FP16_EARLY:
                x = x.cast(gcop.float16)
            x = self.backbone.init_block(x)
            x = self.backbone.layerX(x, 0)
            if cfg.FLOAT16_ON and not cfg.FP16_EARLY:
                x = x.cast(gcop.float16)
            x = self.backbone.layerX(x, 1)
        with gcop.device(self.stage_configs[1]):
            base_feat = self.backbone.layerX(x, 2)
            x = None
        with gcop.device(self.stage_configs[2]):
            fixed_length_roi, roi_keeps, self.rpn_loss_cls, self.rpn_loss_bbox = self.rpn.forward(
                base_feat,
                rpn_data=self.rpn_data,
                im_info=np.asarray(
                    [self.input_im_shape[2],
                     self.input_im_shape[3]]).astype(np.int32),
                stage_configs=self.stage_configs[3])
        with gcop.device(self.stage_configs[4]):
            roi_data = self.rcnn_proposal_target(fixed_length_roi, roi_keeps,
                                                 gt_boxes)
            rois, rois_label, rois_target, _, labels_mask, encoded_rois_target, bbox_inside_weights = gcop.bF.recomputation_checkpoint(roi_data)
            rois_label = rois_label.reshape([-1]).detach()

            local_num_rois = cfg.TRAIN.RPN_BATCHSIZE
            valid_area_mask = get_valid_area_mask(rois)
            rois = rois.detach()
        with gcop.device(self.stage_configs[5]):
            _pooled_feat = roi_align(
                base_feat,
                rois,
                num_rois=local_num_rois,
                fp16_on=self.roialign_fp16,
                aligned_height=cfg.MODEL.ALIGNED_HEIGHT,
                aligned_width=cfg.MODEL.ALIGNED_WIDTH)

            pooled_feat = _pooled_feat.squeeze([0])

        pooled_feat = self.backbone.head_to_tail(
            pooled_feat, self.stage_configs[6])

        with gcop.device(self.stage_configs[7]):
            B, C, H, W = pooled_feat.shape.as_list()
            pooled_feat = pooled_feat.cast(gcop.float32)
            pooled_feat = gcop.reduce_mean(pooled_feat, [2, 3])
            cls_score, bbox_pred = self.backbone.cls_reg_head(pooled_feat)
            cls_prob = gcop.nn.softmax(cls_score, axis=1)
            cls_score_masked = cls_score
            rois_label = rois_label.cast(gcop.int32)

            bbox_pred, encoded_rois_target, bbox_inside_weights, cls_score_masked = [
                ele.cast(gcop.float32) for ele in [
                    bbox_pred, encoded_rois_target, bbox_inside_weights,
                    cls_score_masked
                ]
            ]

            self.rcnn_loss_cls = gcop.nn.sparse_softmax_cross_entropy_with_logits(
                labels=rois_label,
                logits=cls_score_masked,
                name="RCNN_loss_cls")
            encoded_rois_target = encoded_rois_target.detach()
            bbox_inside_weights = bbox_inside_weights.detach()
            self.rcnn_loss_bbox = smooth_l1_loss(bbox_pred.unsqueeze(0),
                                                 encoded_rois_target,
                                                 bbox_inside_weights,
                                                 None,
                                                 reduceDim=[2])

            valid_area_rois = gcop.reduce_sum(valid_area_mask)

            rpn_loss_cls_weight = gcop.constant(
                np.asarray(cfg.TRAIN.RPN_LOSS_CLS_WEIGHT, dtype=np.float32))
            rpn_loss_bbox_weight = gcop.constant(
                np.asarray(cfg.TRAIN.RPN_LOSS_BBOX_WEIGHT, dtype=np.float32))
            rcnn_loss_cls_weight = gcop.constant(
                np.asarray(cfg.TRAIN.RCNN_LOSS_CLS_WEIGHT, dtype=np.float32))
            rcnn_loss_bbox_weight = gcop.constant(
                np.asarray(cfg.TRAIN.RCNN_LOSS_BBOX_WEIGHT, dtype=np.float32))

            self.loss = rpn_loss_cls_weight * self.rpn_loss_cls + rpn_loss_bbox_weight * self.rpn_loss_bbox + \
                rcnn_loss_cls_weight * self.rcnn_loss_cls + \
                rcnn_loss_bbox_weight * self.rcnn_loss_bbox

            if cfg.TRAIN.SKIP_LARGE_LOSS > 0:
                skip_large_loss = gcop.constant(
                    np.asarray(cfg.TRAIN.SKIP_LARGE_LOSS, dtype=np.float32))
                _skip = gcop.math.less_equal(self.loss, skip_large_loss)
                _skip = _skip.cast(gcop.float32).detach()
                self.loss = self.loss * _skip

            self.loss = self.loss * cfg.TRAIN.LOSS_FACTOR

        self.loss_names = [
            'loss', 'rpn_loss_cls', 'rpn_loss_bbox', 'rcnn_loss_cls',
            'rcnn_loss_bbox'
        ]
        self.loss_tensors = [
            self.loss, self.rpn_loss_cls, self.rpn_loss_bbox,
            self.rcnn_loss_cls, self.rcnn_loss_bbox
        ]
        self.moving_loss = {ele: MovingVar(ele) for ele in self.loss_names}

        for loss_tensor, loss_name in zip(self.loss_tensors, self.loss_names):
            self.add_output(loss_name, loss_tensor)

        self.add_output('valid_area_rois', valid_area_rois)

    def get_loss_info(self, outputs_dic):
        for i, loss_name in enumerate(self.loss_names):
            local_loss = outputs_dic[loss_name].data.mean()
            local_moving_loss = self.moving_loss[loss_name]
            local_moving_loss.update(local_loss)

        info = ' >>> total loss: %.6f(%.6f)\n >>> rpn_loss_cls: %.6f(%.6f)\n '\
               '>>> rpn_loss_box: %.6f(%.6f)\n >>> loss_cls: %.6f(%.6f)\n >>> '\
               'loss_box: %.6f(%.6f)' % (
                outputs_dic['loss'].data.mean(), self.moving_loss['loss'].var,
                outputs_dic['rpn_loss_cls'].data.mean(),
                self.moving_loss['rpn_loss_cls'].var,
                outputs_dic['rpn_loss_bbox'].data.mean(),
                self.moving_loss['rpn_loss_bbox'].var,
                outputs_dic['rcnn_loss_cls'].data.mean(),
                self.moving_loss['rcnn_loss_cls'].var,
                outputs_dic['rcnn_loss_bbox'].data.mean(),
                self.moving_loss['rcnn_loss_bbox'].var)
        return info

    def snap(self, output_dir, sess, iters, name=''):
        state_json = os.path.join(output_dir, 'state.json')
        model_name = 'iter' + str(iters) if name == '' else name
        sess.save_model(os.path.join(output_dir, model_name + '.onnx'))
        state_dict = {}
        state_dict['iters'] = iters
        state_dict['loss'] = float(self.moving_loss['loss'].var)
        state_dict['rpn_loss_cls'] = float(
            self.moving_loss['rpn_loss_cls'].var)
        state_dict['rpn_loss_bbox'] = float(
            self.moving_loss['rpn_loss_bbox'].var)
        state_dict['rcnn_loss_cls'] = float(
            self.moving_loss['rcnn_loss_cls'].var)
        state_dict['rcnn_loss_bbox'] = float(
            self.moving_loss['rcnn_loss_bbox'].var)
        with open(state_json, 'w') as f:
            json.dump(state_dict, f)

    def load_from_snap(
        self,
        output_dir,
    ):
        state_json = os.path.join(output_dir, 'state.json')
        with open(state_json, 'r') as f:
            state_dict = json.load(f)
        resume_model_path = os.path.join(
            output_dir, 'iter{}.onnx'.format(state_dict['iters']))
        print('resume:', 'load model {}'.format(resume_model_path))
        gcop.load_model(resume_model_path)
        self.moving_loss['loss'].var = state_dict['loss']
        self.moving_loss['rpn_loss_cls'].var = state_dict['rpn_loss_cls']
        self.moving_loss['rpn_loss_bbox'].var = state_dict['rpn_loss_bbox']
        self.moving_loss['rcnn_loss_cls'].var = state_dict['rcnn_loss_cls']
        self.moving_loss['rcnn_loss_bbox'].var = state_dict['rcnn_loss_bbox']
        return state_dict['iters']
