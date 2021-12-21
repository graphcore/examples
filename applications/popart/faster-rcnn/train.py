# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import sys
sys.path.append('./IPU')
import os
import numpy as np
import time
import torch
import argparse
from utils.utils import load_from_pth_with_mappin, load_onnx, checkNaN_np
from config import cfg
from yaml_parser import change_cfg_by_yaml_file, save_yaml
from utils import logger
from datasets.data_loader import get_data_loader
from ipu_tensor import gcop
from gc_session import Session
from models.get_model import make_model
import random
import cv2


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('yaml', type=str, help='path of yaml')
    args = parser.parse_args()
    return args


# change the trash cfg inplace by yaml config
args = parse_args()
yaml_file_path = args.yaml
change_cfg_by_yaml_file(yaml_file_path)

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
output_dir = os.path.join('outputs', cfg.task_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
save_yaml(os.path.join(output_dir, 'config.yaml'))

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

# set data
train_dataloader = get_data_loader(cfg)
iters_per_epoch = len(train_dataloader)
train_dataloader_iter = iter(train_dataloader)
train_size = iters_per_epoch * cfg.TRAIN.BATCH_SIZE
logger.log_str('{:d} roidb entries'.format(train_size))
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
if cfg.TRAIN.RECORD_WEIGHTS_GRAD:
    net.record_all_weights_grad()

specific_dic = {}
if not cfg.TRAIN.BIAS_DECAY:
    trainable_variables = gcop.trainable_variables()
    trainbale_bias = list(
        filter(lambda tensor: 'bias' in tensor.name, trainable_variables))
    for bias in trainbale_bias:
        specific_dic[bias.name] = {"weightDecay": (0.0, True)}

# optimizer
start_lr = cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.WARMUP_FACTOR if cfg.TRAIN.WARMUP_ITERS > 0 else cfg.TRAIN.LEARNING_RATE
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
logger.log_str('start training...')
while 1:

    if local_iters <= cfg.TRAIN.WARMUP_ITERS:
        current_lr = (start_lr * (cfg.TRAIN.WARMUP_ITERS - local_iters) + end_lr * local_iters) / cfg.TRAIN.WARMUP_ITERS
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
    rpn_label = blobs['rpn_label'].numpy().astype(np.uint32)  # TODO -1 will be converted to 4294967295, but fortunately we only sample non-negative labels for training
    rpn_keep = blobs['rpn_keep'].numpy().astype(np.uint32)
    rpn_bbox_targets = blobs['rpn_bbox_targets'].numpy().astype(np.float32)
    rpn_bbox_inside_weights = blobs['rpn_bbox_inside_weights'].numpy().astype(np.float32)
    rpn_bbox_outside_weights = blobs['rpn_bbox_outside_weights'].numpy().astype(np.float32)

    local_inputs = [
        im_data, gt_boxes,
        rpn_label, rpn_keep, rpn_bbox_targets, rpn_bbox_inside_weights,
        rpn_bbox_outside_weights
    ]

    feed_dict = {net.inputs[k]: n for k, n in zip(net.inputs, local_inputs)}
    results_dic = sess.run(feed_dict=feed_dict)

    if checkNaN_np(results_dic['loss'].data):
        raise Exception('NaN')

    db_inds = blobs['db_inds']

    # collect image ids
    indices_collector.append([int(ele) for ele in db_inds])

    if local_iters % cfg.TRAIN.LOG_INTERVAL == 0 or local_iters in list(
            range(40)):
        log_str = "iter: {}, lr: {}, db_ind: {}\n".format(
            local_iters, current_lr, db_inds) + net.get_loss_info(results_dic)
        valid_area_rois = results_dic['valid_area_rois'].data
        clipped_valid_area_boxes = results_dic[
            'clipped_valid_area_boxes'].data
        valid_area_output_boxes = results_dic['valid_area_output_boxes'].data
        append_str = ' >>> valid_area_rois: {}/256, clipped_valid_area_boxes: {}/9216, valid_area_output_boxes: {}/2000'.format(
            valid_area_rois, clipped_valid_area_boxes,
            valid_area_output_boxes)
        log_str = log_str + '\n' + append_str

        # log
        summary_names = net.loss_names
        [
            logger.log_data(name,
                            float(results_dic[name].data),
                            step=local_iters) for name in summary_names
        ]
        [
            logger.log_data('moving_' + name,
                            float(net.moving_loss[name].var),
                            step=local_iters) for name in summary_names
        ]
        logger.log_str(log_str)

        # record weights grads
        if cfg.TRAIN.RECORD_WEIGHTS_GRAD:
            for _key in results_dic:
                if hasattr(results_dic[_key], 'grad'):
                    _mean = np.abs(results_dic[_key].grad).mean()
                    _std = np.std(results_dic[_key].grad)
                    logger.log_data(_key+'.grad.mean', _mean, step=local_iters)
                    logger.log_data(_key+'.grad.std', _std, step=local_iters)

    local_iters += 1

    if local_iters > max_iters:
        break

    if local_iters > start_to_find_smallest_loss_iters:
        # save smallest loss model in the final epoch
        net.snap(output_dir, sess, iters=local_iters - 1, name='best')

    if local_iters in save_iters:
        # save model and states
        net.snap(output_dir, sess, iters=local_iters - 1)

logger.log_str('total training time: {} hours'.format(
    (time.time() - currentT) / 60 / 60))

net.snap(output_dir, sess, iters=local_iters - 1)

# do something after training
image_indices_path = os.path.join(output_dir, 'image_indices.npy')
np.save(image_indices_path, np.asarray(indices_collector))
