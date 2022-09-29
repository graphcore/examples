# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from easydict import EasyDict as edict
import numpy as np
import os.path as osp
import torch

_C = edict()

_C.NUM_CLASSES = 21
_C.FLOAT16_ON = False
_C.FP16_EARLY = False
_C.WEIGHT_FP16 = None
_C.WEIGHTS_MAPPIN_PATH = 'keys_mappin.txt'
_C.INIT_WEIGHTS_PATH = ''

_C.TRAIN = edict()
_C.TRAIN.TB_ON = True
_C.TRAIN.SEED = 666
_C.TRAIN.WARMUP_ITERS = -1
_C.TRAIN.WARMUP_FACTOR = 0.001
_C.TRAIN.MAX_ITERS = 90000
_C.STEPSIZE = [55000, ]
_C.SAVE_ITERS = [40000, 60000, 80000, ]
_C.START_TO_FIND_SMALLEST_LOSS_ITERS = -1
_C.TRAIN.SET_PIPELINE_MANUALLY = None
_C.TRAIN.WANDB_ON = False
_C.TRAIN.WANDB_PROJECT_NAME = 'faster-rcnn'
_C.TRAIN.WANDB_RUN_NAME = None
_C.TRAIN.SKIP_LARGE_LOSS = -1.0
_C.TRAIN.FC_FP16 = None
_C.TRAIN.RECORD_WEIGHTS_GRAD = False
_C.TRAIN.ROI_ALIGN_FP16 = False
_C.TRAIN.LOSS_FACTOR = 1.0
_C.TRAIN.LOSS_SCALING = 1
_C.TRAIN.BG_THRESH_LO = 0.0
_C.TRAIN.BBOX_INSIDE_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
_C.TRAIN.AVAILABLE_MEMORY_PROPORTION = 0.3
_C.TRAIN.BIAS_DECAY = True
_C.TRAIN.PRESET_INDICES = ''
_C.TRAIN.NUM_GT_BOXES = 20
_C.TRAIN.CLIP_NORM = None
_C.TRAIN.RESUME = False
_C.TRAIN.LOG_INTERVAL = 20
_C.TRAIN.RPN_LOSS_CLS_WEIGHT = 1.0
_C.TRAIN.RPN_LOSS_BBOX_WEIGHT = 1.0
_C.TRAIN.RCNN_LOSS_CLS_WEIGHT = 1.0
_C.TRAIN.RCNN_LOSS_BBOX_WEIGHT = 1.0
_C.TRAIN.PRETRAINED_WEIGHTS = None

_C.MODEL = edict()
_C.MODEL.LOAD_STRICT = True
_C.MODEL.INIT_BLOCK_CONV_STRIDE = 2
_C.MODEL.INIT_BLOCK_POOL_STRIDE = 2
_C.MODEL.LAYERS_STRIDE = [1, 2, 2, 2]
_C.MODEL.RPN_CHANNEL = 512
_C.MODEL.ALIGNED_HEIGHT = 7
_C.MODEL.ALIGNED_WIDTH = 7
_C.MODEL.LAYER4S = False
_C.MODEL.LAYER1_FP16_ON = None
_C.MODEL.LAYER2_FP16_ON = None
_C.MODEL.LAYER3_FP16_ON = None
_C.MODEL.RPN_CONV_FP16_ON = False

_C.aiOnnxOpsetVersion = 11

_C.INPUT_SIZE = [512, 512]
# Feature stride for RPN
# rpn feat stride = (init_block'conv stride) x (init_block'maxpool stride) x (layer0 stride) x (layer1 stride x layer2 stride)
_C.FEAT_STRIDE = _C.MODEL.INIT_BLOCK_CONV_STRIDE * _C.MODEL.INIT_BLOCK_POOL_STRIDE * \
    _C.MODEL.LAYERS_STRIDE[0] * \
    _C.MODEL.LAYERS_STRIDE[1] * _C.MODEL.LAYERS_STRIDE[2]
_C.ANCHOR_SCALES = [8, 16, 32]
_C.ANCHOR_RATIOS = [0.5, 1., 2.]
_C.ANCHOR_VERSION = 'v1'

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
_C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
_C.PIXEL_STD = np.array([[[1.0, 1.0, 1.0]]])

# Root directory of project
_C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '.'))

# Data directory
_C.DATA_DIR = osp.abspath(osp.join(_C.ROOT_DIR, 'data'))

_C.TEST = edict()
_C.TEST.MODEL = ''
_C.TEST.FC_FP16 = None
_C.TEST.ROI_ALIGN_FP16 = False
_C.TEST.KEEP_RATIO = False
_C.TEST.SCORE_THRESH_TEST = 0.0
_C.TEST.SET_PIPELINE_MANUALLY = None


_C.TEST.PIXEL_MEAN = [123.675, 116.28, 103.53]
_C.TEST.PIXEL_STD = [1.0, 1.0, 1.0]

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.TEST.NMS = 0.3

_C.TEST.RPN_PRE_NMS_TOP_N = 6000
_C.TEST.RPN_POST_NMS_TOP_N = 300
_C.TEST.RPN_NMS_THRESH = 0.7

_C.TEST.BATCH_SIZE_PER_REPLICA = 1
_C.TEST.DATASET = "voc"  # voc or coco

_C.MODEL.RCNN = edict()
_C.MODEL.RCNN.EXPAND_PREDICTED_BOXES = True
_C.MODEL.RPN_CONV_FP16_ON = None
_C.MODEL.RCNN.CONV_WEIGHTS_FP16_OFF_INDICES = []

_C.TRAIN.RESNET = edict()
_C.TRAIN.RESNET.NETWORK_TYPE = '50'
_C.TRAIN.ROI_SAMPLER = edict()
_C.TRAIN.ROI_SAMPLER.one_hot_opti = False
_C.TRAIN.RESNET.FIXED_BLOCKS = 0
_C.TRAIN.RESNET.FIXED_BN = False
_C.TRAIN.ADD_GT_BOX_IN_SAMPLER = True
_C.TRAIN.BATCH_SIZE_PER_REPLICA = 1
# IOU >= thresh: positive example
_C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
_C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
_C.TRAIN.ROI_THRD = 0.5
# If an anchor statisfied by positive and negative conditions set to negative
_C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
_C.TRAIN.RPN_FG_FRACTION = 0.5
# Max number of foreground examples of out rois of rpn for rcnn training
_C.TRAIN.RPN_OUT_FG_FRACTION = 0.5
# Total number of examples
_C.TRAIN.RPN_BATCHSIZE = 256
# Deprecated (outside weights)
_C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
_C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False

_C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

_C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
_C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
_C.TRAIN.LEARNING_RATE = 0.001
# Momentum
_C.TRAIN.MOMENTUM = 0.9
# Weight decay, for regularization
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.PROPOSAL_METHOD = "gt"
_C.TRAIN.USE_FLIPPED = True

_C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
_C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
_C.TRAIN.USE_ALL_GT = True

# Maximal number of gt rois in an image during Training
_C.MAX_NUM_GT_BOXES = 20

_C.SESSION = edict()
_C.SESSION.TRAIN = edict()
_C.SESSION.EVAL = edict()
_C.SESSION.COMMON = edict()
# Inferrence session config begin
_C.SESSION.COMMON.enableStochasticRounding = False
_C.SESSION.COMMON.device_iterations = 1
_C.SESSION.COMMON.enableReplicatedGraphs = True
_C.SESSION.COMMON.replicatedGraphCount = 1
_C.SESSION.COMMON.enableEngineCaching = True
_C.SESSION.COMMON.cachePath = 'engine_cache/'

_C.SESSION.TRAIN.CFGS = edict()
_C.SESSION.EVAL.CFGS = edict()
_C.SESSION.COMMON.CFGS = edict()
_C.SESSION.COMMON.CFGS.VirtualGraphMode = 'Manual'
# RecomputationType: NoRecompute, Standard, NormOnly, Pipeline
_C.SESSION.TRAIN.CFGS.RecomputationType = 'NoRecompute'
_C.SESSION.TRAIN.CFGS.accumulationAndReplicationReductionType = 'Mean'
_C.SESSION.TRAIN.CFGS.replication_factor = 1

voc_classes = [
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# nanodata config
_C.NANO_DATA_CFG = edict()
_C.NANO_DATA_CFG.DATA = edict()
_C.NANO_DATA_CFG.DATA.NUM_WORKERS = 8
_C.NANO_DATA_CFG.DATA.SHUFFLE = False
_C.NANO_DATA_CFG.DATA.TRAIN = edict()
_C.NANO_DATA_CFG.DATA.TRAIN.include_difficult = False
_C.NANO_DATA_CFG.DATA.TRAIN.name = "XMLDatasetForRcnn"
_C.NANO_DATA_CFG.DATA.TRAIN.class_names = voc_classes
_C.NANO_DATA_CFG.DATA.TRAIN.img_path = ""
_C.NANO_DATA_CFG.DATA.TRAIN.ann_path = ""
_C.NANO_DATA_CFG.DATA.TRAIN.keep_ratio = True
_C.NANO_DATA_CFG.DATA.TRAIN.area_filter_thrd = 0.5
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline = edict()
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.perspective = 0.0
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.scale = [1.0, 1.0]
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.stretch = [[1, 1], [1, 1]]
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.rotation = 0
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.shear = 0
# _C.NANO_DATA_CFG.DATA.TRAIN.pipeline.translate = 0.2
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.flip = 0.5
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.brightness = 1.0
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.contrast = [1.0, 1.0]
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.saturation = [1.0, 1.0]
_C.NANO_DATA_CFG.DATA.TRAIN.pipeline.normalize = [[103.53, 116.28, 123.675],
                                                  [1.0, 1.0, 1.0]]

cfg = _C


def get_initializer_from_pth(pth_model_path):
    initializer = {}
    #
    weights = torch.load(pth_model_path, map_location=torch.device('cpu'))
    for key, value in weights.items():
        key_popart = key.replace('.', '/')
        initializer[key_popart] = value.detach().numpy().copy()

    return initializer
