# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import random
import cv2
import shutil
import argparse
import torch
import time
import numpy as np
import os
import sys
sys.path.append('./IPU')
from arg_parser import parse_args
from utils.utils import load_from_pth_with_mappin, load_onnx, checkNaN_np
from config import cfg
from yaml_parser import save_yaml
from utils import logger
from datasets.data_loader import get_data_loader
from ipu_tensor import gcop
from gc_session import Session
from models.get_model import make_model


# change the cfg inplace by yaml config and cmd args
parse_args(train=True)

threads = 0
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
cv2.setNumThreads(threads)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(cfg.TRAIN.SEED)

# init results folder
output_dir = cfg.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
save_yaml(os.path.join(output_dir, 'config.yaml'))
shutil.copy(cfg.YAML, os.path.join(output_dir, 'neat_config.yaml'))

# set IPU
gcop.safe_mode_on()
gcop.set_options(cfg.SESSION)
gcop.set_seed(cfg.TRAIN.SEED)
gcop.set_memory_proportion(cfg.TRAIN.AVAILABLE_MEMORY_PROPORTION)
if cfg.MODEL.LOAD_STRICT:
    gcop.set_load_strict()

# init log
logger.init_log(output_dir,
                log_name=cfg.task_name,
                resume=cfg.TRAIN.RESUME,
                tb_on=cfg.TRAIN.TB_ON,
                wandb_on=cfg.TRAIN.WANDB_ON)
logger.log_str('output dir:', output_dir)

# set data
train_dataloader = get_data_loader(cfg)
iters_per_epoch = len(train_dataloader)
train_dataloader_iter = iter(train_dataloader)
train_size = iters_per_epoch * train_dataloader.batch_size
IM_WIDTH, IM_HEIGHT = cfg.INPUT_SIZE
input_im_shape = [1, 3, IM_HEIGHT, IM_WIDTH]

# load initializers
init_weights_path = cfg.INIT_WEIGHTS_PATH
mappin_path = cfg.WEIGHTS_MAPPIN_PAtH
initializer = load_from_pth_with_mappin(init_weights_path, mappin_path)
weights_path = cfg.TRAIN.PRETRAINED_WEIGHTS
if weights_path is not None:
    logger.log_str('loading weights:', weights_path)
    if weights_path.endswith('.pth'):
        append_initializer = load_from_pth_with_mappin(weights_path,
                                                       mappin_path)
    elif weights_path.endswith('.onnx'):
        append_initializer = load_onnx(weights_path)
    else:
        raise RuntimeError('wrong format: {}'.format(weights_path))
    initializer = {**initializer, **append_initializer}
    gcop.enable_global_initializer(initializer)

# make model
net = make_model(
    cfg.MODEL_NAME,
    input_im_shape=input_im_shape,
    input_box_num=cfg.TRAIN.NUM_GT_BOXES,
    fp16_on=cfg.FLOAT16_ON,
    classes=[1] * cfg.NUM_CLASSES,
    training=True,
)

net.bulid_graph()

specific_dic = {}
if not cfg.TRAIN.BIAS_DECAY:
    trainable_variables = gcop.trainable_variables()
    trainbale_bias = list(
        filter(lambda tensor: 'bias' in tensor.name, trainable_variables))
    for bias in trainbale_bias:
        specific_dic[bias.name] = {"weightDecay": (0.0, True)}

# optimizer
start_lr = cfg.TRAIN.LEARNING_RATE * \
    cfg.TRAIN.WARMUP_FACTOR if cfg.TRAIN.WARMUP_ITERS > 0 else cfg.TRAIN.LEARNING_RATE
end_lr = cfg.TRAIN.LEARNING_RATE
current_lr = start_lr
next_momentum = cfg.TRAIN.MOMENTUM
optimizer = gcop.bF.SGD(learning_rate=current_lr,
                        momentum=next_momentum,
                        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                        clip_norm=cfg.TRAIN.CLIP_NORM,
                        lossScaling=cfg.TRAIN.LOSS_SCALING,
                        specific_dic=specific_dic)

# check if resume
start_iters = -1
state_json = os.path.join(output_dir, 'state.json')
if cfg.TRAIN.RESUME and os.path.exists(state_json):
    start_iters = net.load_from_snap(output_dir)

currentT = time.time()
sess = Session(net.outputs, optimizer=optimizer, loss=net.loss)
logger.log_str('model build time:', (time.time() - currentT) / 60,
               ' miniutes')

local_iters = 0

logger.log_str('task name: ', cfg.task_name)
stepsize = cfg.TRAIN.STEPSIZE
max_iters = cfg.TRAIN.MAX_ITERS
save_iters = cfg.TRAIN.SAVE_ITERS
start_to_find_smallest_loss_iters = cfg.TRAIN.START_TO_FIND_SMALLEST_LOSS_ITERS if cfg.TRAIN.START_TO_FIND_SMALLEST_LOSS_ITERS > 0 else float(
    'inf')
logger.log_str('stepsize:{}, max iters:{}'.format(stepsize, max_iters))
smallest_loss = float('inf')

if start_iters == -1:  # no past training is resumed
    sess.save_model(os.path.join(output_dir, 'init_weights.onnx'))

# init data collector
indices_collector = []  # collect image indices
currentT = time.time()
while 1:

    if local_iters <= cfg.TRAIN.WARMUP_ITERS and cfg.TRAIN.WARMUP_ITERS != 0:
        current_lr = (start_lr * (cfg.TRAIN.WARMUP_ITERS - local_iters) +
                      end_lr * local_iters) / cfg.TRAIN.WARMUP_ITERS
        optimizer.adj_lr(current_lr, sess.session, specific_dic=specific_dic)

    if local_iters in stepsize:
        if isinstance(cfg.TRAIN.GAMMA, list):
            current_lr = cfg.TRAIN.GAMMA.pop(0)
        else:
            current_lr *= cfg.TRAIN.GAMMA
        optimizer.adj_lr(current_lr, sess.session, specific_dic=specific_dic)

    if local_iters <= start_iters:
        local_iters += 1
        continue

    blobs = next(train_dataloader_iter)

    im_data = blobs['img'].numpy().astype(np.float32)
    raw_boxes = blobs['gt_bboxes'].numpy().astype(np.float32)
    raw_labels = blobs['gt_labels'].numpy().astype(np.float32)[
        :, :, np.newaxis]
    assert raw_labels.max() < cfg.NUM_CLASSES
    gt_boxes = np.concatenate([raw_boxes, raw_labels],
                              axis=2)
    # TODO -1 will be converted to 4294967295, but fortunately we only sample non-negative labels for training
    rpn_label = blobs['rpn_label'].numpy().astype(np.uint32)
    rpn_keep = blobs['rpn_keep'].numpy().astype(np.uint32)
    rpn_bbox_targets = blobs['rpn_bbox_targets'].numpy().astype(np.float32)
    rpn_bbox_inside_weights = blobs['rpn_bbox_inside_weights'].numpy().astype(
        np.float32)
    rpn_bbox_outside_weights = blobs['rpn_bbox_outside_weights'].numpy().astype(
        np.float32)

    local_inputs = [
        im_data, gt_boxes,
        rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights,
        rpn_bbox_outside_weights
    ]

    feed_dict = {net.inputs[k]: n for k, n in zip(net.inputs, local_inputs)}

    start_time = time.time()
    for i in range(100):
        results_dic = sess.run(feed_dict=feed_dict)
    time_used = time.time() - start_time
    tput = 100 * train_dataloader.batch_size / time_used
    logger.log_str('Faster-RCNN training Tput: {}'.format(tput))
