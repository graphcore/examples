# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# --------------------------------------------------------
# Swin Transformer
# This file has been modified by Graphcore Ltd.
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# The LICENSE referenced above is reproduced below:
# MIT License
#
#     Copyright (c) Microsoft Corporation.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
# Written by Ze Liu
# --------------------------------------------------------
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 1
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = [384, 384]
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model device
_C.MODEL.DEVICE = 'gpu'
# Model alignment
_C.MODEL.ALIGNMENT = False
# Model half
_C.MODEL.HALF = False

# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 12
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
# Normalization layers in SwinTransformerBlock before MLP, default: 'ln',
# choice: ['ln', 'bn']
_C.MODEL.SWIN.NORM_BEFORE_MLP = 'ln'

# MoBY parameters
_C.MODEL.MOBY = CN()
_C.MODEL.MOBY.ENCODER = 'swin'
_C.MODEL.MOBY.ONLINE_DROP_PATH_RATE = 0.1
_C.MODEL.MOBY.TARGET_DROP_PATH_RATE = 0.0
_C.MODEL.MOBY.CONTRAST_MOMENTUM = 0.99
_C.MODEL.MOBY.CONTRAST_TEMPERATURE = 0.2
_C.MODEL.MOBY.CONTRAST_NUM_NEGATIVE = 4096
_C.MODEL.MOBY.PROJ_NUM_LAYERS = 2
_C.MODEL.MOBY.PRED_NUM_LAYERS = 2
_C.MODEL.MOBY.EMA_PATH = ''
_C.MODEL.MOBY.QUEUE_DIM = 256

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-6
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# loss scalling
_C.TRAIN.LOSS_SCALING = 128

# IPU related settings
_C.IPU = CN()
_C.IPU.NUM_REPLICAS = 1
_C.IPU.GRADIENT_ACCUMULATION_STEPS = 8
_C.IPU.DEVICE_ITERATIONS = 1
_C.IPU.IPUS = 1
_C.IPU.LAYERS_PER_IPU = [3, 4, 3, 2]


# -----------------------------------------------------------------------------
# Linear eval settings
# -----------------------------------------------------------------------------
_C.LINEAR_EVAL = CN()
_C.LINEAR_EVAL.PRETRAINED = ''

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# Self-Supervised Learning Augmentation
_C.AUG.SSL_AUG = False
# SSL-Aug type
_C.AUG.SSL_AUG_TYPE = 'byol'
# SSL-Aug crop
_C.AUG.SSL_AUG_CROP = 0.08
# Self-Supervised Learning Linear Evaluation Augmentation
_C.AUG.SSL_LINEAR_AUG = False

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.PRECISION = ['half', 'float']
_C.PRETRAINED = None


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()

    try:
        if args.opts:
            config.merge_from_list(args.opts)
    except Exception as e:
        print(e)

    try:
        # merge from specific arguments
        if args.batch_size:
            config.DATA.BATCH_SIZE = args.batch_size
    except Exception as e:
        print(e)

    try:
        if args.resume:
            config.MODEL.RESUME = args.resume
    except Exception as e:
        print(e)

    try:
        if args.output:
            config.OUTPUT = args.output
    except Exception as e:
        print(e)

    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.output:
        config.OUTPUT = args.output
    if args.pretrained_model:
        config.PRETRAINED = args.pretrained_model

    try:
        if args.ema_so:
            config.MODEL.MOBY.EMA_PATH = args.ema_so
        if args.alignment:
            config.MODEL.ALIGNMENT = args.alignment
        if args.device:
            config.MODEL.DEVICE = args.device
        if args.half:
            config.MODEL.HALF = args.half
    except Exception as e:
        print(e)
    config.MODEL.MOBY.CONTRAST_NUM_NEGATIVE = (
        config.MODEL.MOBY.CONTRAST_NUM_NEGATIVE // config.DATA.BATCH_SIZE) * config.DATA.BATCH_SIZE

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    update_config(config, args)

    return config
