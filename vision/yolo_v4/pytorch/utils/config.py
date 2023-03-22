# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from os import terminal_size
from yacs.config import CfgNode as CN
from ruamel import yaml
from utils.parse_args import opt_to_cfg_map

config = CN()


config.system = CN()
# Number of IPUs to use in the experiment
config.system.num_ipus = 1
# Number of workers for the dataloader
config.system.num_workers = 20


config.model = CN()
# Channels of the input images to the model
config.model.input_channels = 3
# Activation used in the model
config.model.activation = "relu"
# Normalization function used in the model
config.model.normalization = "group"

# Anchor boxes
config.model.anchors = CN()
config.model.anchors.p3width = [13, 31, 24, 61]
config.model.anchors.p3height = [17, 25, 51, 45]
config.model.anchors.p4width = [48, 119, 97, 217]
config.model.anchors.p4height = [102, 96, 189, 184]
config.model.anchors.p5width = [171, 324, 616, 800]
config.model.anchors.p5height = [384, 451, 618, 800]

# Number of classes of the model
config.model.n_classes = 80
# Class name path
config.model.class_name_path = "configs/class_name.yaml"
# Image size of the input images to the model
config.model.image_size = 896
# Strides of the prediction layers of the model
config.model.strides = [8, 16, 32]
# Float precision used in the model
config.model.precision = "half"
# The number of samples calculated in one full forward/backward pass
config.model.micro_batch_size = 1
# Mode to run the model
config.model.mode = "test"
# Run model on cpu or ipu
config.model.ipu = True
# Compute optimal anchors
config.model.auto_anchors = False
# Anchors threshold to compare against when chooseing the best anchors
config.model.anchor_threshold = 4.0
# Send the data using uint instead of floats to the IPU
config.model.uint_io = True
# Pipeline splits
config.model.pipeline_splits = []
# Recomputation checkpoints
config.model.recomputation_ckpts = []
# Use sharded execution
config.model.sharded = False


config.ipuopts = CN()
config.ipuopts.device_iterations = 1
config.ipuopts.gradient_accumulation = 1
config.ipuopts.available_memory_proportion = None


config.inference = CN()
# Whether to perform NMS or not
config.inference.nms = True
# Minimum threshold for objectness probability
config.inference.obj_threshold = 0.4
# Minimum threshold for class prediction probability
config.inference.class_conf_threshold = 0.6
# Minimum threshold for IoU used in NMS
config.inference.iou_threshold = 0.65
# Maximum number of detections after NMS
config.inference.nms_max_detections = 300
# Maximum number of detections filtered before NMS
config.inference.pre_nms_topk_k = 2000
# Plot output and save to file
config.inference.plot_output = False
# Plot every n image
config.inference.plot_step = 250
# Directory for storing the plot output
config.inference.plot_dir = "plots"
# Minimum confidence threshold for plotting a bounding box in the final plot
config.inference.plot_threshold = 0.3


config.training = CN()
# Initial learning rate
config.training.initial_lr = 0.01
# Momentum used in SGD or Adam
config.training.momentum = 0.937
# Optimizer weight decay
config.training.weight_decay = 0.0005
# SGD loss scaling factor
config.training.loss_scaling = 4096.0
# Enable automatic loss scaling
config.training.auto_loss_scaling = False
# Stochastic Rounding
config.training.stochastic_rounding = False
# Number of training epochs
config.training.epochs = 300
# Logging interval to wandb
config.training.logging_interval = 200
# Executable cache path, storing compiled model
config.training.exec_cache_path = "./exec_cache"
# Checkpoint interval to save the internal state of the model
config.training.checkpoint_interval = 10
# Scaling factors for each loss component
config.training.box_gain = 0.05
config.training.class_gain = 0.5
config.training.object_gain = 1.0
config.training.fl_gamma = 0.0
config.training.ciou_ratio = 1.0
# Weight averaging decay factor
config.training.weight_avg_decay = 0.9999


config.dataset = CN()
# Name of the dataset
config.dataset.name = "coco"
# Maximum number of bounding boxes per image in the dataset
config.dataset.max_bbox_per_scale = 90


# The labels are masked using 5 different masks in the preprocessing
mask_n = 5
# Maximum number of labels per detector size after preprocessing the labels
config.model.max_nlabels_p3 = mask_n * len(config.model.anchors.p3width) * config.dataset.max_bbox_per_scale
config.model.max_nlabels_p4 = mask_n * len(config.model.anchors.p3width) * config.dataset.max_bbox_per_scale
config.model.max_nlabels_p5 = mask_n * len(config.model.anchors.p3width) * config.dataset.max_bbox_per_scale


config.dataset.train = CN()
config.dataset.test = CN()
# Cache the images on ram instead of reading them from disk
config.dataset.train.cache_data = False
config.dataset.test.cache_data = False
# Path to the annotations of the coco dataset
config.dataset.train.file = "train2017.txt"
config.dataset.test.file = "val2017.txt"
config.dataset.test.annotation = "instances_val2017.json"
# Path to cache the data on disk (Labels, shapes and names of image files)
config.dataset.train.cache_path = "./utils/data/train"
config.dataset.test.cache_path = "./utils/data/test"
# Use data augmentation
config.dataset.train.data_aug = False
config.dataset.test.data_aug = False
# Data augmentation modes mosaic and color (TODO)
config.dataset.mosaic = False
config.dataset.color = False
# Data aug (cutout) - proportion of object obscured in order to be removed from the label
config.dataset.train.cutout_obscured_pct = 0.6
# Data aug (cutout) - minimum cutout area scale required for label to be removed
config.dataset.train.cutout_scaled_treshold = 0.3
# Data aug (hsv) - hue, sat and val gain
config.dataset.train.hsv_h_gain = 0.5
config.dataset.train.hsv_s_gain = 0.5
config.dataset.train.hsv_v_gain = 0.5
# Data Aug - random perspective config
config.dataset.train.degrees = 0.0
config.dataset.train.translate = 0.5
config.dataset.train.scale = 0.5
config.dataset.train.shear = 0.0
config.dataset.train.perspective = 0.0
# Data Aug - flip
config.dataset.train.flipud = 0.0
config.dataset.train.fliplr = 0.5

config.eval = CN()
# Compute eval metrics
config.eval.metrics = True
# Display eval metrics per class
config.eval.verbose = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for YoloV4."""
    return config.clone()


def override_cfg(opt, cfg):
    override_list = []
    for k, v in vars(opt).items():
        if k in opt_to_cfg_map and v is not None:
            override_list += [opt_to_cfg_map[k], v]
    cfg.merge_from_list(override_list)
    return cfg


def convert_to_dict(cfg_node, key_list=[]):
    def flist(x):
        retval = yaml.comments.CommentedSeq(x)
        retval.fa.set_flow_style()
        return retval

    if not isinstance(cfg_node, CN):
        if isinstance(cfg_node, list):
            cfg_node = flist(cfg_node)
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def save_cfg(path, cfg):
    cfg_dict = convert_to_dict(cfg)
    with open(path, "w") as f:
        ryaml = yaml.YAML()
        ryaml.dump(cfg_dict, f)
