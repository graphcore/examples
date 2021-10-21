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
config.model.class_name_path = "./configs/class_name.yaml"
# Image size of the input images to the model
config.model.image_size = 416
# Strides of the prediction layers of the model
config.model.strides = [8, 16, 32]
# Float precision used in the model
config.model.half = True
# The number of samples calculated in one full forward/backward pass
config.model.micro_batch_size = 1
# Mode to run the model
config.model.mode = "test"
# Run model on cpu or ipu
config.model.ipu = True
# If run sweep to calculate the best anchors TODO
config.model.auto_anchors = False
# Anchors threshold to compare against when chooseing the best anchors
config.model.anchor_threshold = 4.0
# Send the data using uint instead of floats to the IPU
config.model.uint_io = True


config.ipuopts = CN()
config.ipuopts.batches_per_step = 1


config.inference = CN()
# Minimum threshold for objectness probability
config.inference.obj_threshold = 0.4
# Minimum threshold for class prediction probability
config.inference.class_conf_threshold = 0.6
# Minimum threshold for IoU used in NMS
config.inference.iou_threshold = 0.5
# Plot output and save to file
config.inference.plot_output = False
# Plot every n image
config.inference.plot_step = 250
# Directory for storing the plot output
config.inference.plot_dir = "plots"


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
