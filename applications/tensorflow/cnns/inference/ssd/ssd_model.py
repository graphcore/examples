# Copyright 2020 Graphcore Ltd.
"""
SSD implementation based on Vgg-16 entry network as originally published by Liu et al.

This is an implementation of the Single Shot MultiBox Detector (SSD) using a dual-device, single-graph
framework as deployed for inference. The convolutional component of the model is entirely deployed
on the IPU, while the decoding component lives entirely on host.

The current code is heavily derived from the code base presented by Pierluigi Ferrari
in his Github repository:

https://github.com/pierluigiferrari/ssd_keras

The code can be run with randomly generated weights for purely synthetic benchmarking purposes,
or trained weights can be loaded to run actual detections. Further details can be found in the README
file included in this directory.

"""


import logging
import os
import time
from typing import Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Layer imports
import tf_layers as layers
from PIL.JpegImagePlugin import JpegImageFile
from gcprofile import \
    save_tf_report  # FIXME This depends on the user having installed the Toolshed .whl file (which requires checking out the repo, firing the make script to get the whl)
# Custom Keras Imports
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
# General Keras Imports
from tensorflow.keras.preprocessing import image
from tensorflow.python import ipu
# IPU TF Imports
from tensorflow.python.ipu import utils
# Weight loading dictionary
from trained_weights.LoadWeights import hdf5_key_list

# Image typeset
ImageData = Union[np.ndarray, Tuple[int, int], JpegImageFile]

# Use random weights
RANDOM_WEIGHTS = True

# For loading the hdf5 file
if not RANDOM_WEIGHTS:
    trained_weight_path = os.getcwd()+'/trained_weights/VGG_VOC0712_SSD_300x300_iter_120000.h5'

# Reporting flag
REPORT = False

# Save output flag
SAVE_IMAGE = False

# Number of IPUs
NUM_IPUS = 1
IPU_MODEL = False

# Precision
DTYPE = np.float16

if IPU_MODEL:
    os.environ['TF_POPLAR_FLAGS'] = "--use_ipu_model"

# Input Dimensions
BATCH_SIZE = 1
WIDTH = 300
HEIGHT = 300
CHANNELS = 3

# Number of steps
N_ITERATIONS = 5
BATCHES_PER_STEP = 1000

# Setup number of classes, aspect ratios, and the number of boxes per layer
N_CLASSES = 21  # Includes +1 for the background class
ASPECT_RATIOS_PER_LAYER = [[1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                           [1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5]]
SCALES = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
STEPS = [8, 16, 32, 64, 100, 300]
OFFSETS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
VARIANCES = [0.1, 0.1, 0.2, 0.2]
N_BOXES = []
for ar in ASPECT_RATIOS_PER_LAYER:
    if 1 in ar:
        N_BOXES.append(len(ar) + 1)
    else:
        N_BOXES.append(len(ar))
print("Boxes per-layer are: ", N_BOXES)

# Post detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Image rendition items
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']

# Test image
image_path = './example_images/fish-bike.jpg'


def prepare_image(path_to_image: str):
    """
    Load an image, subtract a pre-defined mean, and then do a channel swap. (RGB --> BGR).

    :param path_to_image: string/directory where the sample image is located
    :return:
        processed_image: (np.ndarray) post processed image that can be fed to SSD model
        original_dimensions: (tuple) size of original image
        loaded_image:(JpegImageFile) that is used for final rendition of bounding box
    """
    loaded_image = image.load_img(path_to_image)
    original_dimensions = loaded_image.size
    # Resize image to appropriate processing dimensions
    resized_image = image.load_img(path_to_image, target_size=(300, 300))
    base_image = image.img_to_array(resized_image)
    '''
        Pre-processing in line with what network is expecting
    '''
    # Subtract the mean
    subtract_mean_image = base_image - np.array([123, 117, 104])
    # Swap channels
    channel_swap = subtract_mean_image[..., [2, 1, 0]]
    # Add the batch dimension and convert to single precision
    processed_image = np.expand_dims(channel_swap, axis=0).astype('float16')
    return processed_image, original_dimensions, loaded_image


def process_detections(raw_predictions: np.ndarray) -> list:
    """
    Filter the produced detections for those that are above a given confidence threshold or for non-background
    labels. (Both are available here for debugging purposes.)

    :param raw_predictions: (np.ndarray) detections produced from detection decoder
    :return:
        no_background: (list) filtered detections based on threshold and classification

    """
    # Filter out low confidence identifications
    filtered_predictions = [raw_predictions[k][raw_predictions[k, :, 1] > CONFIDENCE_THRESHOLD]
                            for k in range(raw_predictions.shape[0])]

    return filtered_predictions


def draw_detections(raw_image: JpegImageFile, original_width: int, original_height: int, detections: list) -> None:
    """
    Draws the bounding box around produced detections.

    :param raw_image: (JpegImageFile) original loaded image
    :param original_width: (int) width of original image
    :param original_height: (int) height of original image
    :param detections: (list) filtered detections
    :return: None
    """
    plt.figure(figsize=(20, 12))
    plt.imshow(raw_image)
    current_axis = plt.gca()

    for detection_box in detections[0]:
        x_min = detection_box[2] * original_width / 300
        y_min = detection_box[3] * original_height / 300
        x_max = detection_box[4] * original_width / 300
        y_max = detection_box[5] * original_height / 300
        color = colors[int(detection_box[0])]
        label = '{}: {:.2f}'.format(classes[int(detection_box[0])], detection_box[1])
        current_axis.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                             color=color, fill=False, linewidth=2))
        current_axis.text(x_min, y_min, label, size='x-large', color='white',
                          bbox={'facecolor': color, 'alpha': 1.0})
    if SAVE_IMAGE:
        plt.savefig('detection_output.png')


# Load an image and prepare it for network processing
np_image, original_image_dims, original_image = prepare_image(image_path)

with tf.device('cpu'):
    """
        Create placeholder for detection
    """
    # Proposed detections
    input_detection = tf.placeholder(np.float16, [1, 8732, 33], name="input_detection")

# Prepare dataset for ipu_infeeds
dataset = tf.data.Dataset \
    .range(1) \
    .map(lambda k: {"features": np_image}) \
    .repeat() \
    .prefetch(BATCHES_PER_STEP)

# Setup infeed queue
if BATCHES_PER_STEP > 1:
    with tf.device('cpu'):
        infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset, feed_name="inference_infeed")
else:
    raise NotImplementedError(
        "batches per step == 1 not implemented yet.")

# Setup outfeed
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed")


def ssd_model():
    """
    Build SSD model.
    """
    def body(features):
        with tf.variable_scope("SSDGraph"):
            with ipu.scopes.ipu_scope('/device:IPU:0'):
                # conv 1 block
                conv1_1 = layers.conv(features, ksize=3, stride=1,
                                      filters_out=64, name="conv1_1")
                conv1_1 = layers.relu(conv1_1)
                conv1_2 = layers.conv(conv1_1, ksize=3, stride=1, filters_out=64, name="conv1_2")
                conv1_2 = layers.relu(conv1_2)
                pool1 = layers.maxpool(conv1_2, size=2, stride=2)
                # conv 2 block
                conv2_1 = layers.conv(pool1, ksize=3, stride=1, filters_out=128, name="conv2_1")
                conv2_1 = layers.relu(conv2_1)
                conv2_2 = layers.conv(conv2_1, ksize=3, stride=1, filters_out=128, name="conv2_2")
                conv2_2 = layers.relu(conv2_2)
                pool2 = layers.maxpool(conv2_2, size=2, stride=2)
                # conv 3 block
                conv3_1 = layers.conv(pool2, ksize=3, stride=1, filters_out=256, name="conv3_1")
                conv3_1 = layers.relu(conv3_1)
                conv3_2 = layers.conv(conv3_1, ksize=3, stride=1, filters_out=256, name="conv3_2")
                conv3_2 = layers.relu(conv3_2)
                conv3_3 = layers.conv(conv3_2, ksize=3, stride=1, filters_out=256, name="conv3_3")
                conv3_3 = layers.relu(conv3_3)
                pool3 = layers.maxpool(conv3_3, size=2, stride=2)
                # conv 4 block
                conv4_1 = layers.conv(pool3, ksize=3, stride=1, filters_out=512, name="conv4_1")
                conv4_1 = layers.relu(conv4_1)
                conv4_2 = layers.conv(conv4_1, ksize=3, stride=1, filters_out=512, name="conv4_2")
                conv4_2 = layers.relu(conv4_2)
                conv4_3 = layers.conv(conv4_2, ksize=3, stride=1, filters_out=512, name="conv4_3")
                conv4_3 = layers.relu(conv4_3)  # feature map to be used for object detection/classification
                pool4 = layers.maxpool(conv4_3, size=2, stride=2)
                # conv 5 block
                conv5_1 = layers.conv(pool4, ksize=3, stride=1, filters_out=512, name="conv5_1")
                conv5_1 = layers.relu(conv5_1)
                conv5_2 = layers.conv(conv5_1, ksize=3, stride=1, filters_out=512, name="conv5_2")
                conv5_2 = layers.relu(conv5_2)
                conv5_3 = layers.conv(conv5_2, ksize=3, stride=1, filters_out=512, name="conv5_3")
                conv5_3 = layers.relu(conv5_3)
                pool5 = layers.maxpool(conv5_3, size=3, stride=1)
                # END VGG

                # Extra feature layers
                # fc6
                fc6 = layers.conv(pool5, ksize=3, dilation_rate=(6, 6), stride=1, filters_out=1024, name="fc6")
                fc6 = layers.relu(fc6)
                # fc7
                fc7 = layers.conv(fc6, ksize=1, stride=1, filters_out=1024, name="fc7")
                fc7 = layers.relu(fc7)  # feature map to be used for object detection/classification
                # conv 6 block
                conv6_1 = layers.conv(fc7, ksize=1, stride=1, filters_out=256, name="conv6_1")
                conv6_1 = layers.relu(conv6_1)
                conv6_1 = tf.pad(conv6_1, paddings=([[0, 0], [1, 1], [1, 1], [0, 0]]), name='conv6_padding')
                conv6_2 = layers.conv(conv6_1, ksize=3, stride=2, filters_out=512, padding='VALID', name="conv6_2")
                conv6_2 = layers.relu(conv6_2)  # feature map to be used for object detection/classification
                # conv 7 block
                conv7_1 = layers.conv(conv6_2, ksize=1, stride=1, filters_out=128, name="conv7_1")
                conv7_1 = layers.relu(conv7_1)
                conv7_1 = tf.pad(conv7_1, paddings=([[0, 0], [1, 1], [1, 1], [0, 0]]), name='conv7_padding')
                conv7_2 = layers.conv(conv7_1, ksize=3, stride=2, filters_out=256, padding='VALID', name="conv7_2")
                conv7_2 = layers.relu(conv7_2)  # feature map to be used for object detection/classification
                # conv 8 block
                conv8_1 = layers.conv(conv7_2, ksize=1, stride=1, filters_out=128, name="conv8_1")
                conv8_1 = layers.relu(conv8_1)
                conv8_2 = layers.conv(conv8_1, ksize=3, stride=1, filters_out=256, padding='VALID', name="conv8_2")
                conv8_2 = layers.relu(conv8_2)  # feature map to be used for object detection/classification
                # conv 9 block
                conv9_1 = layers.conv(conv8_2, ksize=1, stride=1, filters_out=128, name="conv9_1")
                conv9_1 = layers.relu(conv9_1)
                conv9_2 = layers.conv(conv9_1, ksize=3, stride=1, filters_out=256, padding='VALID', name="conv9_2")
                conv9_2 = layers.relu(conv9_2)  # feature map to be used for object detection/classification
                # Perform L2 normalization on conv4_3
                conv4_3_norm = tf.math.l2_normalize(conv4_3, axis=3)
                # Conv confidence predictors have output depth N_BOXES * N_CLASSES
                conv4_3_norm_mbox_conf = layers.conv(conv4_3_norm, ksize=3, stride=1, filters_out=N_BOXES[0]*N_CLASSES,
                                                     name='conv4_3_norm_mbox_conf')
                fc7_mbox_conf = layers.conv(fc7, ksize=3, stride=1, filters_out=N_BOXES[1]*N_CLASSES,
                                            name='fc7_mbox_conf')
                conv6_2_mbox_conf = layers.conv(conv6_2, ksize=3, stride=1, filters_out=N_BOXES[2]*N_CLASSES,
                                                name='conv6_2_mbox_conf')
                conv7_2_mbox_conf = layers.conv(conv7_2, ksize=3, stride=1, filters_out=N_BOXES[3]*N_CLASSES,
                                                name='conv7_2_mbox_conf')
                conv8_2_mbox_conf = layers.conv(conv8_2, ksize=3, stride=1, filters_out=N_BOXES[4]*N_CLASSES,
                                                name='conv8_2_mbox_conf')
                conv9_2_mbox_conf = layers.conv(conv9_2, ksize=3, stride=1, filters_out=N_BOXES[5]*N_CLASSES,
                                                name='conv9_2_mbox_conf')
                # Conv box location predictors have output depth N_BOXES * 4 (box coordinates)
                conv4_3_norm_mbox_loc = layers.conv(conv4_3_norm, ksize=3, stride=1, filters_out=N_BOXES[0]*4,
                                                    name='conv4_3_norm_mbox_loc')
                fc7_mbox_loc = layers.conv(fc7, ksize=3, stride=1, filters_out=N_BOXES[1]*4,
                                           name='fc7_mbox_loc')
                conv6_2_mbox_loc = layers.conv(conv6_2, ksize=3, stride=1, filters_out=N_BOXES[2]*4,
                                               name='conv6_2_mbox_loc')
                conv7_2_mbox_loc = layers.conv(conv7_2, ksize=3, stride=1, filters_out=N_BOXES[3]*4,
                                               name='conv7_2_mbox_loc')
                conv8_2_mbox_loc = layers.conv(conv8_2, ksize=3, stride=1, filters_out=N_BOXES[4]*4,
                                               name='conv8_2_mbox_loc')
                conv9_2_mbox_loc = layers.conv(conv9_2, ksize=3, stride=1, filters_out=N_BOXES[5]*4,
                                               name='conv9_2_mbox_loc')
                # Generate the anchor boxes
                conv4_3_norm_mbox_priorbox = AnchorBoxes(HEIGHT, WIDTH, this_scale=SCALES[0], next_scale=SCALES[1],
                                                         two_boxes_for_ar1=True, this_steps=STEPS[0],
                                                         this_offsets=OFFSETS[0], clip_boxes=False,
                                                         variances=VARIANCES, aspect_ratios=ASPECT_RATIOS_PER_LAYER[0],
                                                         normalize_coords=True,
                                                         name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
                fc7_mbox_priorbox = AnchorBoxes(HEIGHT, WIDTH, this_scale=SCALES[1], next_scale=SCALES[2],
                                                two_boxes_for_ar1=True, this_steps=STEPS[1],
                                                this_offsets=OFFSETS[1], clip_boxes=False,
                                                variances=VARIANCES, aspect_ratios=ASPECT_RATIOS_PER_LAYER[1],
                                                normalize_coords=True,
                                                name='fc7_mbox_priorbox')(fc7_mbox_loc)
                conv6_2_mbox_priorbox = AnchorBoxes(HEIGHT, WIDTH, this_scale=SCALES[2], next_scale=SCALES[3],
                                                    two_boxes_for_ar1=True, this_steps=STEPS[2],
                                                    this_offsets=OFFSETS[2], clip_boxes=False,
                                                    variances=VARIANCES, aspect_ratios=ASPECT_RATIOS_PER_LAYER[2],
                                                    normalize_coords=True,
                                                    name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
                conv7_2_mbox_priorbox = AnchorBoxes(HEIGHT, WIDTH, this_scale=SCALES[3], next_scale=SCALES[4],
                                                    two_boxes_for_ar1=True, this_steps=STEPS[3],
                                                    this_offsets=OFFSETS[3], clip_boxes=False,
                                                    variances=VARIANCES, aspect_ratios=ASPECT_RATIOS_PER_LAYER[3],
                                                    normalize_coords=True,
                                                    name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
                conv8_2_mbox_priorbox = AnchorBoxes(HEIGHT, WIDTH, this_scale=SCALES[4], next_scale=SCALES[5],
                                                    two_boxes_for_ar1=True, this_steps=STEPS[4],
                                                    this_offsets=OFFSETS[4], clip_boxes=False,
                                                    variances=VARIANCES, aspect_ratios=ASPECT_RATIOS_PER_LAYER[4],
                                                    normalize_coords=True,
                                                    name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
                conv9_2_mbox_priorbox = AnchorBoxes(HEIGHT, WIDTH, this_scale=SCALES[5], next_scale=SCALES[6],
                                                    two_boxes_for_ar1=True, this_steps=STEPS[5],
                                                    this_offsets=OFFSETS[5], clip_boxes=False,
                                                    variances=VARIANCES, aspect_ratios=ASPECT_RATIOS_PER_LAYER[5],
                                                    normalize_coords=True,
                                                    name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
                # Reshape class predictions
                conv4_3_norm_mbox_conf_reshape = tf.reshape(conv4_3_norm_mbox_conf, shape=(-1,
                                                            conv4_3_norm_mbox_conf.shape[1] *
                                                            conv4_3_norm_mbox_conf.shape[2]*N_BOXES[0], N_CLASSES),
                                                            name='conv4_3_norm_mbox_conf_reshape')
                fc7_mbox_conf_reshape = tf.reshape(fc7_mbox_conf, shape=(-1,
                                                   fc7_mbox_conf.shape[1]*fc7_mbox_conf.shape[2]*N_BOXES[1], N_CLASSES),
                                                   name='fc7_mbox_conf_reshape')
                conv6_2_mbox_conf_reshape = tf.reshape(conv6_2_mbox_conf, shape=(-1,
                                                       conv6_2_mbox_conf.shape[1]*conv6_2_mbox_conf.shape[2] *
                                                                                 N_BOXES[2], N_CLASSES),
                                                       name='conv6_2_mbox_conf_reshape')
                conv7_2_mbox_conf_reshape = tf.reshape(conv7_2_mbox_conf, shape=(-1, conv7_2_mbox_conf.shape[1] *
                                                                                 conv7_2_mbox_conf.shape[2] *
                                                                                 N_BOXES[3], N_CLASSES),
                                                       name='conv7_2_mbox_conf_reshape')
                conv8_2_mbox_conf_reshape = tf.reshape(conv8_2_mbox_conf, shape=(-1, conv8_2_mbox_conf.shape[1] *
                                                                                 conv8_2_mbox_conf.shape[2] *
                                                                                 N_BOXES[4], N_CLASSES),
                                                       name='conv8_2_mbox_conf_reshape')
                conv9_2_mbox_conf_reshape = tf.reshape(conv9_2_mbox_conf, shape=(-1, conv9_2_mbox_conf.shape[1] *
                                                                                 conv9_2_mbox_conf.shape[2] *
                                                                                 N_BOXES[5], N_CLASSES),
                                                       name='conv9_2_mbox_conf_reshape')
                # Reshape box location predictions
                conv4_3_norm_mbox_loc_reshape = tf.reshape(conv4_3_norm_mbox_loc,
                                                           shape=(-1, conv4_3_norm_mbox_loc.shape[1] *
                                                                  conv4_3_norm_mbox_loc.shape[2] *
                                                                  N_BOXES[0], 4),
                                                           name='conv4_3_norm_mbox_loc_reshape')
                fc7_mbox_loc_reshape = tf.reshape(fc7_mbox_loc, shape=(-1,
                                                                       fc7_mbox_loc.shape[1] * fc7_mbox_loc.shape[2] *
                                                                       N_BOXES[1], 4),
                                                  name='fc7_mbox_loc_reshape')
                conv6_2_mbox_loc_reshape = tf.reshape(conv6_2_mbox_loc, shape=(-1, conv6_2_mbox_loc.shape[1] *
                                                                               conv6_2_mbox_loc.shape[2]*N_BOXES[2], 4),
                                                      name='conv6_2_mbox_loc_reshape')
                conv7_2_mbox_loc_reshape = tf.reshape(conv7_2_mbox_loc, shape=(-1, conv7_2_mbox_loc.shape[1] *
                                                                               conv7_2_mbox_loc.shape[2]*N_BOXES[3], 4),
                                                      name='conv7_2_mbox_loc_reshape')
                conv8_2_mbox_loc_reshape = tf.reshape(conv8_2_mbox_loc, shape=(-1, conv8_2_mbox_loc.shape[1] *
                                                                               conv8_2_mbox_loc.shape[2]*N_BOXES[4], 4),
                                                      name='conv8_2_mbox_loc_reshape')
                conv9_2_mbox_loc_reshape = tf.reshape(conv9_2_mbox_loc, shape=(-1, conv9_2_mbox_loc.shape[1] *
                                                                               conv9_2_mbox_loc.shape[2]*N_BOXES[5], 4),
                                                      name='conv9_2_mbox_loc_reshape')
                # Reshape anchor box tensors
                conv4_3_norm_mbox_priorbox_reshape = tf.reshape(conv4_3_norm_mbox_priorbox,
                                                                shape=(-1, conv4_3_norm_mbox_priorbox.shape[1] *
                                                                       conv4_3_norm_mbox_priorbox.shape[2] * N_BOXES[0],
                                                                       8),
                                                                name='conv4_3_norm_mbox_priorbox_reshape')
                fc7_mbox_priorbox_reshape = tf.reshape(fc7_mbox_priorbox,
                                                       shape=(-1,
                                                              fc7_mbox_priorbox.shape[1]*fc7_mbox_priorbox.shape[2] *
                                                              N_BOXES[1], 8), name='fc7_mbox_priorbox_reshape')
                conv6_2_mbox_priorbox_reshape = tf.reshape(conv6_2_mbox_priorbox,
                                                           shape=(-1, conv6_2_mbox_priorbox.shape[1] *
                                                                  conv6_2_mbox_priorbox.shape[2] * N_BOXES[2], 8),
                                                           name='conv6_2_mbox_priorbox_reshape')
                conv7_2_mbox_priorbox_reshape = tf.reshape(conv7_2_mbox_priorbox,
                                                           shape=(-1, conv7_2_mbox_priorbox.shape[1] *
                                                                  conv7_2_mbox_priorbox.shape[2] * N_BOXES[3], 8),
                                                           name='conv7_2_mbox_priorbox_reshape')
                conv8_2_mbox_priorbox_reshape = tf.reshape(conv8_2_mbox_priorbox,
                                                           shape=(-1, conv8_2_mbox_priorbox.shape[1] *
                                                                  conv8_2_mbox_priorbox.shape[2] * N_BOXES[4], 8),
                                                           name='conv8_2_mbox_priorbox_reshape')
                conv9_2_mbox_priorbox_reshape = tf.reshape(conv9_2_mbox_priorbox,
                                                           shape=(-1, conv9_2_mbox_priorbox.shape[1] *
                                                                  conv9_2_mbox_priorbox.shape[2] * N_BOXES[5], 8),
                                                           name='conv9_2_mbox_priorbox_reshape')
                # Concatenate predictions from different layers
                mbox_conf = tf.concat([conv4_3_norm_mbox_conf_reshape,
                                       fc7_mbox_conf_reshape,
                                       conv6_2_mbox_conf_reshape,
                                       conv7_2_mbox_conf_reshape,
                                       conv8_2_mbox_conf_reshape,
                                       conv9_2_mbox_conf_reshape],
                                      axis=1, name='mbox_conf')
                mbox_loc = tf.concat([conv4_3_norm_mbox_loc_reshape,
                                      fc7_mbox_loc_reshape,
                                      conv6_2_mbox_loc_reshape,
                                      conv7_2_mbox_loc_reshape,
                                      conv8_2_mbox_loc_reshape,
                                      conv9_2_mbox_loc_reshape],
                                     axis=1, name='mbox_loc')
                mbox_priorbox = tf.concat([conv4_3_norm_mbox_priorbox_reshape,
                                          fc7_mbox_priorbox_reshape,
                                          conv6_2_mbox_priorbox_reshape,
                                          conv7_2_mbox_priorbox_reshape,
                                          conv8_2_mbox_priorbox_reshape,
                                          conv9_2_mbox_priorbox_reshape],
                                          axis=1, name='mbox_priorbox')

                # Softmax activation layer
                mbox_conf_softmax = tf.nn.softmax(mbox_conf,
                                                  name='mbox_conf_softmax')
                predictions = tf.concat([mbox_conf_softmax, mbox_loc, mbox_priorbox], axis=2, name='predictions')

                # Output
                outfeed = outfeed_queue.enqueue(predictions)
                return outfeed
    return ipu.loops.repeat(n=BATCHES_PER_STEP, body=body, infeed_queue=infeed_queue)


def decoder_component(raw_predictions: tf.Tensor) -> tf.Tensor:
    """
    Decode the output of the multi-box detection boxes

    :param raw_predictions: (tf.Tensor) convolution output from peek network
    :return:
        decoded_predictions: (tf.Tensor) decoded predictions
    """
    with tf.variable_scope("decoder"):
        with tf.device('cpu'):
            decoded_predictions = DecodeDetections(img_height=HEIGHT, img_width=WIDTH, iou_threshold=0.5,
                                                   name='decoded_predictions',
                                                   normalize_coords=True)(raw_predictions)
            return decoded_predictions


"""
    Graph compile calls
"""
# Compiles graph and targets IPU(s)
inference_output = ipu.ipu_compiler.compile(ssd_model, inputs=[])
# Compiles decoder on host (CPU)
decoder = decoder_component(input_detection)

# Assignment operator for trained weight file
param_setters = dict()
for var in tf.trainable_variables():
    placeholder = tf.placeholder(var.dtype, var.shape, var.name.split(':')[0]+'_setter')
    param_setters[var.name] = (tf.assign(var, placeholder), placeholder)

# Capture IPU event trace for reporting
if REPORT:
    with tf.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

# Setup IPU configuration and build session
cfg = ipu.utils.create_ipu_config(profiling=REPORT, use_poplar_text_report=False,
                                  profile_execution=REPORT)
cfg = ipu.utils.auto_select_ipus(cfg, num_ipus=NUM_IPUS)
cfg = utils.set_convolution_options(cfg, convolution_options={"availableMemoryProportion": "0.4"})
ipu.utils.configure_ipu_system(cfg)
ipu.utils.move_variable_initialization_to_cpu()
outfeed = outfeed_queue.dequeue()

# Calculate total flops for graph (experimental)
run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()
flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op', options=opts)
print("Total FLOPs reported by TF is: ", flops.total_float_ops)

# Initiate and run the session
with tf.Session() as sess:
    fps = []
    latency = []
    sess.run(infeed_queue.initializer)
    sess.run(tf.global_variables_initializer())

    if not RANDOM_WEIGHTS:
        # Load trained weights from HDF5
        with h5py.File(trained_weight_path, 'r') as f:
            for key in hdf5_key_list.keys():
                val = f[hdf5_key_list[key]].value
                sess.run(param_setters[key][0], {param_setters[key][1]: val})

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total number of trainable params: ', total_parameters)
    # Warm up
    print("Compiling and Warmup...")
    start = time.time()
    sess.run(inference_output)
    convolution_predictions = sess.run(outfeed)
    # convolution_predictions = sess.run(inference_output, feed_dict={input_image: np_image})
    raw_output = sess.run(decoder, feed_dict={input_detection: convolution_predictions[0]})
    filtered_output = process_detections(raw_output)
    draw_detections(original_image, original_image_dims[0], original_image_dims[1], filtered_output)
    print("Done running inference.")
    duration = time.time() - start
    print("Duration: {:.3f} seconds\n".format(duration))
    if REPORT:
        rep_out = sess.run(report)
        save_tf_report(rep_out)
        rep = utils.extract_all_strings_from_event_trace(rep_out)
        with open(str(WIDTH) + "x" + str(HEIGHT) + "_ipus" + str(NUM_IPUS) + "_ssd_report.txt", "w") as f:
            f.write(rep)
    # Performance runs
    print("Executing...")
    for iter_count in range(N_ITERATIONS):
        print("Running iteration: ", iter_count)
        # Run
        start = time.time()
        sess.run(inference_output)
        convolution_predictions = sess.run(outfeed)
        raw_output = sess.run(decoder, feed_dict={input_detection: convolution_predictions[0]})
        filtered_output = process_detections(raw_output)
        stop = time.time()
        fps.append((BATCHES_PER_STEP * BATCH_SIZE) / (stop - start))
        logging.info(
            "Iter {3}: {0} Throughput using real data = {1:.1f} imgs/sec at batch size = {2}".format("SSD",
                                                                                                     fps[-1],
                                                                                                     BATCH_SIZE,
                                                                                                     iter_count))
        latency.append(1000 * (stop - start) / BATCHES_PER_STEP)
        logging.info(
            "Iter {3}: {0} Latency using real data = {2:.2f} msecs at batch_size = {1}".format("SSD", BATCH_SIZE,
                                                                                               latency[-1],
                                                                                               iter_count))

    print("Average statistics over {0} iterations, excluding the 1st iteration.".format(N_ITERATIONS))
    print("-------------------------------------------------------------------------------------------")
    fps = fps[1:]
    latency = latency[1:]
    print("Throughput at bs={} of {}: min={}, max={}, mean={}, std={}.".format(BATCH_SIZE, "SSD", min(fps),
                                                                               max(fps), np.mean(fps),
                                                                               np.std(fps)))
    print(
        "Latency at bs={} of {}: min={}, max={}, mean={}, std={}.".format(BATCH_SIZE, "SSD", min(latency),
                                                                          max(latency),
                                                                          np.mean(latency),
                                                                          np.std(latency)))
    print("Mean TFLOPs/S is: ", (flops.total_float_ops*(10**-9)/np.mean(latency)))
    plt.plot(fps)
    plt.xlabel("Step index")
    plt.ylabel("Images per sec")
    logging.info(f"Saving frames per second vs step size in ssd_fps.png")
    plt.savefig(f"SSD_fps")
