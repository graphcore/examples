# Copyright 2019 Graphcore Ltd.
# Script to run inference on a trained Densenet model.

import argparse
import collections
import glob
import inspect
import itertools
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import utils
from tensorflow.python.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image

# Add path with models folder to pythonpath
cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
sys.path.insert(1, os.path.join(cwd, '..', '..'))

from models.official_keras.densenet_base import DenseNet
from models import optimize_for_infer
from models.densenet_weights import get_densenet_weights


try:
    import PIL
except ImportError:
    PIL = None

if PIL is None:
    raise ImportError('`densenet_inference` requires pillow python package.')

# Densenet 121 flavor - as defined in https://arxiv.org/abs/1608.06993
DENSENET_121_BLOCKS = (6, 12, 24, 16)

# IMAGENET dataset
NUM_CLASSES = 1000
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
CHECKPOINT_DIR = "../../../models/tensorflow/densenet_weights_fp16/"
IMAGE_DIR = "./images"

# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
tf.get_logger().setLevel(logging.ERROR)


def construct_graph(batch_size: int = 1) -> tf.Tensor:
    """Construct densenet inference graph on the IPU.

    Args:
        batch_size: Batch size for inference

    Returns: Output probability

    """
    # Set up the graph
    densenet_model = DenseNet(blocks=DENSENET_121_BLOCKS,
                              num_classes=NUM_CLASSES,
                              image_width=IMG_WIDTH,
                              image_height=IMG_HEIGHT,
                              image_channels=IMG_CHANNELS)
    image_input = tf.placeholder(dtype=tf.float16,
                                 shape=(batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                                 name="image_input")
    densenet_model(image_input)

    # Restore weights
    checkpoint_dir = CHECKPOINT_DIR
    if tf.train.latest_checkpoint(checkpoint_dir) is None:
        logging.info('Checkpoint directory `%s` does not contain a checkpoint, '
                     'attempting to download pre-trained weights.',
                     Path(checkpoint_dir))
        get_densenet_weights(Path(checkpoint_dir))

    if tf.train.latest_checkpoint(checkpoint_dir) is None:
        raise ValueError("Weight download failed. Please re-try downloading the weights using the `densenet_weights.py`"
                         " script under models/tensorflow/")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        logging.info('Successfully restored imagenet weights.')

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
            return tf.import_graph_def(optimized_graph_def,
                                       input_map={},
                                       name="optimized",
                                       return_elements=["output-prob:0"])[0]


def generate_report(batch_size: int, report_dest: str = "./densenet_report.txt") -> None:
    """Generate report from running model on IPU

    Args:
        batch_size: Batch size for inference
        report_dest: Location to save generated text report

    """
    # Set compile and device options
    os.environ['TF_POPLAR_FORCE_IPU_MODEL'] = "1"
    opts = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
    utils.auto_select_ipus(opts, [1])
    utils.configure_ipu_system(opts)
    output_probs = construct_graph(batch_size)

    with tf.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    with tf.Session() as session:
        session.run(output_probs,
                    feed_dict={"optimized/image_input:0":
                               np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float16)},
                    options=run_options)
        out = session.run(report)

    # extract the report
    rep = utils.extract_all_strings_from_event_trace(out)
    logging.info("Writing densenet profiling report to %s" % report_dest)
    with open(report_dest, "w") as f:
        f.write(rep)


def classify_image(session: tf.Session, img_path: Path, output_prob: tf.Tensor) -> None:
    """Forward pass single batch through the network and print results.

    Args:
        session: Tensorflow session to run the inference
        img_path: Path to image file
        output_prob: Probability of output

    """
    # TODO(lakshmik): Image pre-fetching
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = session.run(output_prob, feed_dict={"optimized/image_input:0": img})

    print('Actual: ', Path(img_path).stem)
    # Decode the results into a list of tuples (class, description, probability)
    print('Predicted: ', decode_predictions(predictions, top=3)[0])


def run_inference(batch_size: int, image_dir: Path = Path(IMAGE_DIR), loop: bool = False) -> None:
    """Run inference on pre-trained Densenet model.

    Args:
        batch_size: Batch size for inference
        image_dir: Path to dir of images
        loop: Flag to iterate through the images endlessly

    Raises:
        ValueError if `image_dir` does not contain test images.

    """

    image_filenames = glob.glob(image_dir.as_posix() + "/*.jpg")
    if len(image_filenames) == 0:
        raise ValueError(('Image directory: %s does not have images,'
                          'please run `./get_images.sh` '
                          'to download sample imagenet images' % image_dir.as_posix()))

    opts = utils.create_ipu_config(profiling=False, use_poplar_text_report=False)
    utils.auto_select_ipus(opts, [1])
    utils.configure_ipu_system(opts)

    output_probs = construct_graph(batch_size)

    timings = collections.deque(maxlen=250)  # keep the most recent timings
    with tf.Session() as session:
        if loop:
            image_filenames = itertools.cycle(image_filenames)

        for img_file in image_filenames:
            classify_image(session, img_file, output_probs)
            timings.append(time.time())
            if len(timings) > 2:
                fps = (len(timings) - 1) / (timings[-1] - timings[1])
                print("Average images per second: {0:.1f}".format(fps))


def main(ipu_model: bool = True, image_dir: str = IMAGE_DIR, batch_size: int = 1,
         loop: bool = False) -> None:
    """Run inference on ipu_model or on device.

    Args:
        ipu_model: Run on IPU model and generate a report.
        image_dir: Path to directory of images.
        batch_size: Batch size for inference.
        loop: Flag to iterate through the images endlessly

    """

    if ipu_model:
        generate_report(batch_size)
    else:
        # TODO(lakshmik): Add batch size > 1 support.
        if batch_size > 1:
            raise NotImplementedError('Densenet inference does not fit on single IPU with bs > 1.')
        run_inference(batch_size, Path(image_dir), loop)


if __name__ == "__main__":
    """Run Densenet121 inference on either IPU model or on device."""
    parser = argparse.ArgumentParser(description="Run Densenet-121 inference.")
    parser.add_argument('image_dir', type=str,
                        help="Path to directory of images to run inference on.")
    parser.add_argument('--ipu-model', dest='ipu_model', action='store_true',
                        help="Run on IPU model and generate report.")
    parser.add_argument('--loop', dest='loop', action='store_true',
                        help="Run inference on device endlessly.", default=False)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                        help="Batch size for inference.")

    parser.set_defaults(ipu_model=False)
    args = parser.parse_args()
    main(args.ipu_model, args.image_dir, args.batch_size, args.loop)
