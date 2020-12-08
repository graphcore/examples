# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import logging
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from models import optimize_for_infer
from models.densenet_weights import get_densenet_weights
from models.official_keras.densenet_base import DenseNet
from tensorflow.python.ipu import utils
from tensorflow.python.keras.applications.densenet import preprocess_input, \
    decode_predictions
from tensorflow.python.keras.preprocessing import image

# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

CHECKPOINT_PATH = "../tensorflow/densenet_weights_fp16/densenet_model.ckpt"
IMAGE_PATH = "../../tensorflow/image_classification/densenet/images/gorilla.jpg"
IMAGE_DIR = "../../tensorflow/image_classification/densenet/images"


class TestDensenet(unittest.TestCase):
    """Test densenet model. """

    @classmethod
    def setUpClass(cls):
        # Set up input to the network
        img_width = img_height = 224
        img_channels = 3
        densenet_121_blocks = (6, 12, 24, 16)
        cls.batch_size = 1
        cls.num_classes = 1000
        # Set up image input placeholder
        cls.placeholder_input = tf.placeholder(dtype=tf.float16,
                                               shape=(cls.batch_size, img_height, img_width, img_channels),
                                               name="image_input")

        # Set compile and device options
        opts = utils.create_ipu_config(profiling=False, use_poplar_text_report=False)
        utils.auto_select_ipus(opts, [1])
        utils.configure_ipu_system(opts)

        # Construct Densenet model
        cls.densenet_model = DenseNet(blocks=densenet_121_blocks, num_classes=cls.num_classes,
                                      image_width=img_width, image_height=img_height, image_channels=img_channels)

        cls.densenet_model(cls.placeholder_input)

        # Restore weights
        checkpoint_file = CHECKPOINT_PATH

        if not Path(checkpoint_file + ".index").exists():
            print('Checkpoint file does not exist, attempting to download pre-trained weights')
            checkpoint_file = get_densenet_weights(Path(checkpoint_file))

        # Create test session
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_file)
            logging.info('Restored imagenet weights.')

            # Optimize inference graph
            logging.info('Starting graph optimization.')
            densenet_graph_def = tf.get_default_graph().as_graph_def()
            frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, densenet_graph_def,
                                                                                      output_node_names=["output-prob"])
            # Remove identity ops in initializers to allow fusing batch norm with conv in the next line
            frozen_graph_def = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph_def)
            optimized_graph_def = optimize_for_infer.fold_batch_norms(frozen_graph_def)

            logging.info('Completed graph optimization.')

        tf.reset_default_graph()
        with tf.device('/device:IPU:0'):
            with tf.variable_scope('', use_resource=True):
                cls.output = tf.import_graph_def(optimized_graph_def, input_map={}, name="optimized",
                                                 return_elements=["output-prob:0"])[0]

    def test_output_shape(self):
        assert self.output._shape_as_list() == [self.batch_size, self.num_classes]

    def test_image(self, img_path: str = IMAGE_PATH) -> None:
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        with tf.Session() as session:
            predictions = session.run(self.output, feed_dict={"optimized/image_input:0": img})

        _, pred_label, pred_prob = decode_predictions(predictions, top=1)[0][0]
        assert Path(IMAGE_PATH).stem.lower() == pred_label.lower()
        assert pred_prob > 0.9
