# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
from torchvision.ops import nms
import cv2
import argparse
import os
import numpy as np
import time
import tqdm
import json
import torch
import sys
sys.path.append('./IPU')
from models.get_model import make_model
from utils.utils import load_onnx
from config import cfg
from datasets.factory import get_imdb
from ipu_tensor import gcop
from gc_session import Session
from utils.utils import load_from_pth_with_mappin
from utils import logger
from arg_parser import parse_args


def clip_boxes_npy(boxes, im_shape):
    """
  Clip boxes to image boundaries.
  boxes must be tensor or Variable, im_shape can be anything but Variable
  """

    boxes = boxes.reshape(boxes.shape[0], -1, 4)
    boxes = np.stack([
        boxes[:, :, 0].clip(0, im_shape[1] - 1), boxes[:, :, 1].clip(
            0, im_shape[0] - 1), boxes[:, :, 2].clip(0, im_shape[1] - 1),
        boxes[:, :, 3].clip(0, im_shape[0] - 1)
    ], 2).reshape(boxes.shape[0], -1)

    return boxes


def bbox_transform_inv_npy(boxes, deltas):
    # boxes: (n,4)
    # deltas: (n,4)
    # Input should be both tensor or both Variable and on the same device
    if len(boxes) == 0:
        return deltas * 0

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.concatenate([
        _[:, :, np.newaxis] for _ in [
            pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x +
            0.5 * pred_w, pred_ctr_y + 0.5 * pred_h
        ]
    ], 2).reshape(len(boxes), -1)

    return pred_boxes

# change the cfg inplace by yaml config and cmd args
parse_args(train=False)

# init outputs dir
output_dir = cfg.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# init log
log_prefix = cfg.EVAL_MODEL_NAME if cfg.EVAL_MODEL_NAME != '' else ''
log_prefix = '_inference' + log_prefix
logger.init_log(output_dir,
                log_name=cfg.task_name,
                post_fix=log_prefix,
                resume=False,
                tb_on=False,
                wandb_on=False)
logger.log_str('output dir:', output_dir)

if cfg.TEST.DATASET == 'voc':
    imdb = get_imdb('voc_2007_test')
elif cfg.TEST.DATASET == 'coco':
    imdb = get_imdb('coco_2017_val')
else:
    raise ValueError("Unknown dataset!")

imdb.competition_mode(False)
val_size = imdb.num_images
logger.log_str('{:d} roidb entries'.format(val_size))

IM_WIDTH, IM_HEIGHT = cfg.TEST.SCALES
INPUT_SHAPE = [1, 3, IM_HEIGHT, IM_WIDTH]
THRESH = cfg.TEST.SCORE_THRESH_TEST
MAX_PER_IMAGE = 100

total_iters = val_size

last_state_json = os.path.join(output_dir, 'state.json')
if cfg.TEST.MODEL == '':
    with open(last_state_json, 'r') as f:
        last_state = json.load(f)
    iters = last_state['iters']
    pretrained_weights_path = os.path.join(output_dir,
                                           'iter{}.onnx'.format(iters))
else:
    pretrained_weights_path = cfg.TEST.MODEL

if cfg.EVAL_MODEL_NAME != '':
    pretrained_weights_path = os.path.join(output_dir,
                                           '{}.onnx'.format(cfg.EVAL_MODEL_NAME))

# load resnet50 weights
init_weights_path = cfg.INIT_WEIGHTS_PATH
initializer = {}
if init_weights_path is not None and init_weights_path != '':
    mappin_path = cfg.WEIGHTS_MAPPIN_PATH
    initializer = load_from_pth_with_mappin(init_weights_path, mappin_path)

# load faster-rcnn trained weights
if pretrained_weights_path is not None and pretrained_weights_path != '':
    logger.log_str('load weights from :', pretrained_weights_path)
    if pretrained_weights_path.endswith('.pth'):
        pretrained_weights = load_from_pth_with_mappin(pretrained_weights_path,
                                                       mappin_path)
    elif pretrained_weights_path.endswith('.onnx'):
        pretrained_weights = load_onnx(pretrained_weights_path)
    else:
        raise RuntimeError('wrong file format')
    # merge them
    pretrained_weights = {
        **initializer,
        **pretrained_weights
    }  # overwrite some weights in initializer by weights in pretrained_weights
    gcop.enable_global_initializer(pretrained_weights)

# set IPU
gcop.safe_mode_on()
cfg.SESSION.COMMON.enableEngineCaching = False
gcop.set_options(cfg.SESSION, training=False)
gcop.set_memory_proportion(cfg.TRAIN.AVAILABLE_MEMORY_PROPORTION)

# build net
net = make_model(
    cfg.MODEL_NAME,
    input_im_shape=INPUT_SHAPE,
    fp16_on=cfg.FLOAT16_ON,
    classes=[1] * cfg.NUM_CLASSES,
    training=False,
)

net.bulid_graph()
currentT = time.time()
sess = Session(net.outputs)
logger.log_str('model build time:', (time.time() - currentT) / 60,
               ' miniutes')

# gather results
all_boxes = [[[] for _ in range(imdb.num_images)]
             for _ in range(imdb.num_classes)]

inference_start_time = time.time()
for im_id in tqdm.tqdm(list(range(total_iters))):
    im = cv2.imread(imdb.image_path_at(im_id))
    normalized_im = im - cfg.TEST.PIXEL_MEAN
    normalized_im = normalized_im / cfg.TEST.PIXEL_STD
    h, w, _ = im.shape
    if cfg.TEST.KEEP_RATIO:  # IM_WIDTH,IM_HEIGHT
        x_scale = min(IM_HEIGHT / h, IM_WIDTH / w)
        y_scale = x_scale
    else:
        x_scale = IM_WIDTH / w
        y_scale = IM_HEIGHT / h
    normalized_im = cv2.resize(normalized_im,
                               None,
                               None,
                               fx=x_scale,
                               fy=y_scale,
                               interpolation=cv2.INTER_LINEAR).astype(
                                   np.float32)
    im_data = np.zeros([IM_HEIGHT, IM_WIDTH, 3], np.float32)
    im_data[:normalized_im.shape[0], :normalized_im.
            shape[1], :] = normalized_im
    im_data = np.transpose(
        im_data[np.newaxis, :, :, :],
        [0, 3, 1, 2]).astype(np.float32)
    im_data = np.ascontiguousarray(im_data)

    feed_dict = {net.inputs[k]: n for k, n in zip(net.inputs, [im_data])}
    results = sess.run(feed_dict)

    scores = results['cls_prob'].data.astype(np.float32)  # 300,21
    bbox_deltas = results['bbox_pred'].data.astype(np.float32)  # 300,84
    rois = results['fixed_length_roi'].data.astype(np.float32)  # 1,300,4
    rois_keep = results['roi_keeps'].data.astype(np.float32)  # 1,300
    # collect valid results
    valid_area_mask = results['valid_area_mask'].data  # 256,1
    valid_area_indices = np.where(valid_area_mask[:, 0] > 0)[0]
    scores = scores[valid_area_indices, :]
    bbox_deltas = bbox_deltas[valid_area_indices, :]
    rois = rois[:, valid_area_indices, :]

    boxes = rois[0] / np.array([x_scale, y_scale, x_scale, y_scale],
                               dtype=np.float32)
    pred_boxes = bbox_transform_inv_npy(boxes, bbox_deltas)
    pred_boxes = clip_boxes_npy(pred_boxes, [IM_HEIGHT, IM_WIDTH])

    for j in range(1, imdb.num_classes):
        inds = np.where(scores[:, j] > THRESH)[0]
        cls_scores = scores[inds, j]
        cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
                   cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
        cls_dets = cls_dets[keep, :]
        all_boxes[j][im_id] = cls_dets

    # Limit to MAX_PER_IMAGE detections *over all classes*
    if MAX_PER_IMAGE > 0:
        image_scores = np.hstack(
            [all_boxes[j][im_id][:, -1] for j in range(1, imdb.num_classes)])
        if len(image_scores) > MAX_PER_IMAGE:
            image_thresh = np.sort(image_scores)[-MAX_PER_IMAGE]
            for j in range(1, imdb.num_classes):
                keep = np.where(all_boxes[j][im_id][:, -1] >= image_thresh)[0]
                all_boxes[j][im_id] = all_boxes[j][im_id][keep, :]

inference_time = time.time() - inference_start_time
logger.log_str('inference time:', inference_time,
               ' seconds')
logger.log_str('inference throughput:', val_size / inference_time)

eval_output_dir = os.path.join(output_dir, 'eval')
if not os.path.exists(eval_output_dir):
    os.mkdir(eval_output_dir)
np.save(os.path.join(eval_output_dir, 'all_boxes.npy'), all_boxes)
mAP = imdb.evaluate_detections(all_boxes, eval_output_dir, logger=logger)
logger.log_str('inference mAP:', mAP)
logger.log_str('end!!!')
