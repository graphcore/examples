# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Written by Hu Di
import random
import cv2
import shutil
import argparse
import torch
import time
import numpy as np
import popart
import os
import sys
sys.path.append('./IPU')
from models.get_model import make_model
from gc_session import Session
from ipu_tensor import gcop
from datasets.data_loader import get_data_loader
from utils import logger
from arg_parser import parse_args
from yaml_parser import save_yaml
from config import cfg
from utils.utils import load_from_pth_with_mappin, load_onnx, checkNaN_np

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
logger.init_log(log_dir=output_dir,
                log_name=cfg.task_name,
                resume=cfg.TRAIN.RESUME,
                tb_on=cfg.TRAIN.TB_ON,
                wandb_on=cfg.TRAIN.WANDB_ON,
                wandb_project_name=cfg.TRAIN.WANDB_PROJECT_NAME,
                wandb_run_name=cfg.TRAIN.WANDB_RUN_NAME
                )
logger.log_str('output dir:', output_dir)

# log sdk version
logger.log_str('sdk version: ', str(popart.__version__))

# set data
train_dataloader = get_data_loader(cfg)
iters_per_epoch = len(train_dataloader)
train_dataloader_iter = iter(train_dataloader)
train_size = iters_per_epoch * train_dataloader.batch_size
logger.log_str('{:d} roidb entries'.format(train_size))
IM_WIDTH, IM_HEIGHT = cfg.INPUT_SIZE
input_im_shape = [1, 3, IM_HEIGHT, IM_WIDTH]

# load initializers if INIT_WEIGHTS_PATH is set
init_weights_path = cfg.INIT_WEIGHTS_PATH
initializer = {}
if init_weights_path is not None and init_weights_path != '':
    mappin_path = cfg.WEIGHTS_MAPPIN_PATH
    initializer = load_from_pth_with_mappin(init_weights_path, mappin_path)

weights_path = cfg.TRAIN.PRETRAINED_WEIGHTS
if weights_path is not None and weights_path != '':
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
# record all tensors information
all_tensors = gcop.bF.get_all_tensors_info()
tensors_info_log_file = os.path.join(output_dir, 'tensors_info.txt')
with open(tensors_info_log_file, 'w') as f:
    for t in all_tensors:
        f.write(str(t) + '\n')
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
logger.log_str('start training...')
while 1:
    iter_start_time = time.time()
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
    results_dic = sess.run(feed_dict=feed_dict)

    iter_time_used = time.time() - iter_start_time
    tput = train_dataloader.batch_size / iter_time_used

    if checkNaN_np(results_dic['loss'].data):
        raise Exception('NaN')

    db_inds = blobs['db_inds']

    # collect image ids
    indices_collector.append([int(ele) for ele in db_inds])

    if local_iters % cfg.TRAIN.LOG_INTERVAL == 0 or local_iters in list(
            range(40)):
        log_str = "iter: {}, lr: {}, throughput: {} samples/sec, db_ind: {}\n".format(
            local_iters, current_lr, tput, db_inds)
        log_str = log_str + net.get_loss_info(results_dic)

        # log
        summary_names = net.loss_names
        [
            logger.log_data(name,
                            float(results_dic[name].data.mean()),
                            step=local_iters) for name in summary_names
        ]
        [
            logger.log_data('moving_' + name,
                            float(net.moving_loss[name].var),
                            step=local_iters) for name in summary_names
        ]

        logger.log_data('throughput', tput, step=local_iters)
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
