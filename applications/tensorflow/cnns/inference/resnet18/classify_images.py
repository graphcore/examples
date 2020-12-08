#!/usr/bin/env python
# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import argparse
import os
import itertools
import collections
import time

import imagenet_categories
from resnet import ResNet
from tensorflow.python.ipu import utils


class ImageClassifier(object):

    def __init__(self, weights_path = './weights'):
        """Builds a TensorFlow image classifier model for Graphcore IPUs."""
        # Set compile and device options
        self.opts = utils.create_ipu_config(profiling=False, use_poplar_text_report=False)
        cfg = utils.auto_select_ipus(self.opts, [1])
        utils.configure_ipu_system(cfg)

        # Build a Graph that computes the predictions from the inference model.
        img_size = 224
        num_classes = 1000
        checkpoint_file = os.path.join(weights_path, '16bit-0')

        self.jpeg_input = tf.placeholder(tf.string)
        raw_img = tf.image.decode_jpeg(self.jpeg_input, channels=3)
        image_data = tf.cast(tf.reshape(raw_img, [1, img_size, img_size, 3]), tf.float16) / 255.0

        # Build model
        with tf.device('/device:IPU:0'):
            with tf.variable_scope('', use_resource=True):
                self.network = ResNet(image_data, num_classes)
                self.network.build_model()

        # For restoring the data
        saver = tf.train.Saver()
        self.session = tf.Session()
        # Restore weights
        saver.restore(self.session, checkpoint_file)

        # Try to prime with a dummy image to force the graph compilation. Note that this is
        # a temporary workaround so that the first real inference doesn't include
        # a long time building the graph.
        try:
            jpeg_file = tf.gfile.GFile('images/zebra.jpg', 'rb').read()
            self.session.run(self.network.probs, feed_dict={self.jpeg_input: jpeg_file})
        except tf.errors.NotFoundError:
            pass

    def classify_image(self, image_filename):
        """Classify a single image

        image_filename -- A JPEG image of the appropriate size

        """
        jpeg_file = tf.gfile.GFile(image_filename, 'rb').read()
        preds = self.session.run(self.network.probs, feed_dict={self.jpeg_input: jpeg_file})
        predictions = np.squeeze(preds)

        # Print top predictions
        top_k = predictions.argsort()[-5:][::-1]
        print("\nFilename : {0}".format(os.path.basename(image_filename)))
        for v in top_k:
            print("Class {0: >3}: {1} {2:1.3g}%".format(v, imagenet_categories.labels[v], 100 * predictions[v]))

    def classify_images(self, image_filenames, loop):
        """Classify multiple images

        image_filenames -- list of JPEG images
        loop -- if True endlessly loop over the images

        """
        if loop:
            image_filenames = itertools.cycle(image_filenames)
        timings = collections.deque(maxlen=250)  # keep the most recent timings
        for f in image_filenames:
            self.classify_image(f)
            timings.append(time.time())
            if len(timings) > 1:
                fps = (len(timings) - 1) / (timings[-1] - timings[0])
                print("\nAverage images per second: {0:.1f}".format(fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify images using ResNet-18')
    parser.add_argument('image', type=str, nargs='+',
                        help='image file name(s) or single directory')
    parser.add_argument('--loop', action="store_true",
                        help="Endlessly loop through all the images")
    args = parser.parse_args()

    # If a directory was given then get the files in it
    if len(args.image) == 1 and os.path.isdir(args.image[0]):
        image_filenames = [os.path.join(args.image[0], f) for f in
                           os.listdir(args.image[0]) if not f.startswith('.')]
    else:
        image_filenames = args.image
    # Filter out non-jpeg images
    image_filenames = [f for f in image_filenames if tf.gfile.Exists(f) and
                       f.lower().endswith(('.jpg', '.jpeg'))]

    if image_filenames:
        print("{0} image(s) found".format(len(image_filenames)))
        ic = ImageClassifier()
        ic.classify_images(image_filenames, args.loop)
    else:
        print("No image files found.")
