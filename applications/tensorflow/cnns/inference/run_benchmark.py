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

"""Run image classification inference on chosen model."""

import argparse
import glob
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type

import numpy as np
import tensorflow.compat.v1 as tf
import yaml

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import scopes
from tensorflow.python.ipu import utils as ipu_utils
from tensorflow.python.ipu import sharding
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from tensorflow.python.ipu.config import IPUConfig

from data import get_dataset
from inference_network_base import InferenceNetwork
from inference_networks import model_dict

# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

tf.logging.set_verbosity("INFO")


def run_inference(loop_op: tf.Operation, infeed_queue_initializer: tf.Operation, outfeed_op: tf.Operation,
                  batch_size: int, batches_per_step: int, network_name: str,
                  decode_predictions: Callable, ground_truth: Tuple[str], num_iterations: Optional[int] = 500,
                  num_ipus: Optional[int] = 1, mode: Optional[str] = "single_ipu",
                  data: Optional[str] = "real", available_memory_proportion: Optional[float] = 0.6) -> None:
    """Run inference on device and decode predictions.

    Args:
        loop_op: Inference op.
        infeed_queue_initializer: Initializer for the infeed queue.
        outfeed_op: Outfeed operator to extract results.
        batch_size: Batch size per forward pass.
        batches_per_step: Number of forward passes per step.
        network_name: Name of this network, to use in frames_per_second plot filename.
        decode_predictions: Function to decode predictions with.
        ground_truth: Ground-truth labels.
        num_iterations: Number of iterations to run the inference, if running in a loop.
        num_ipus: Number of ipus to run the inference on.
        mode: Mode of inference - {"single_ipu", "replicated"}
        data: Run on real data transferred from host or on random synthetic data generated on device.
        available_memory_proportion: Proportion of tile memory available as temporary memory for
        matmul and convolution execution

    """
    # Set compile and device options
    opts = IPUConfig()
    opts.matmuls.poplar_options = {'availableMemoryProportion': str(
        available_memory_proportion)}
    opts.convolutions.poplar_options = {'availableMemoryProportion': str(
        available_memory_proportion)}

    if mode == 'replicated':
        num_replicas = num_ipus
        os.environ["TF_POPLAR_FLAGS"] += " --force_replicated_mode"
    else:
        num_replicas = 1
    opts.auto_select_ipus = num_ipus
    opts.configure_ipu_system()
    with tf.Session() as session:
        session.run(infeed_queue_initializer)
        fps = []
        for iter_count in range(num_iterations):
            start = time.time()
            session.run(loop_op)
            predictions = session.run(outfeed_op)
            stop = time.time()
            fps.append(batch_size * batches_per_step * num_replicas / (stop - start))
            logging.info(
                "Iter {4}: {0} Throughput using {1} data = {2:.1f} imgs/sec at batch size = {3}".format(network_name,
                                                                                                        data,
                                                                                                        fps[-1],
                                                                                                        batch_size,
                                                                                                        iter_count))
            duration = stop - start
            report_string = "{:<7.3} sec/itr.".format(duration)
            report_string += "   {:5f} images/sec.".format(fps[-1])
            print(report_string)
            print("Total time: {}".format(duration))

            # Decode a random prediction per step to check functional correctness.
            if data == 'real':
                predictions = np.reshape(predictions, (-1, predictions.shape[-1]))
                index = np.random.randint(0, len(predictions))
                if network_name in ("inceptionv1", "efficientnet-s", "efficientnet-m", "efficientnet-l"):
                    # These models encode background in 0th index.
                    decoded_predictions = decode_predictions(predictions[index: index + 1, 1:], top=3)
                else:
                    decoded_predictions = decode_predictions(predictions[index: index + 1, :], top=3)
                labels_and_probs = [(label, prob) for _, label, prob in decoded_predictions[0]]
                print('Actual: ',
                      ground_truth[
                          (index + num_replicas * iter_count * batches_per_step * batch_size) % len(ground_truth)])
                print('Predicted: ', labels_and_probs)

    print("Average statistics excluding the 1st 20 iterations.")
    print("-------------------------------------------------------------------------------------------")
    fps = fps[20:]
    print("Throughput at bs={}, data_mode={}, data_type={}, mode={},"
          " num_ipus={}, of {}: min={}, max={}, mean={}, std={}.".format(batch_size,
                                                                         data,
                                                                         predictions.dtype,
                                                                         mode,
                                                                         num_ipus,
                                                                         network_name,
                                                                         min(fps),
                                                                         max(fps),
                                                                         np.mean(fps),
                                                                         np.std(fps)))


def construct_graph(network_class: Type[InferenceNetwork],
                    config: Path, checkpoint_dir: str,
                    batch_size: int,
                    batches_per_step: int,
                    image_filenames: Tuple[str],
                    loop: bool,
                    preprocess_fn: Callable,
                    num_ipus: int,
                    mode: str,
                    save_graph_pb: bool) -> Tuple[tf.Operation, tf.Operation, tf.Operation]:
    """Create inference graph on the device, set up in-feeds and out-feeds, connect dataset iterator to the graph.

    This function also exports the frozen graph into an event file, to be viewed in Tensorboard in `network_name_graph`
    directory.

    Args:
        network_class: Class corresponding to chosen model.
        config: Path to config file.
        checkpoint_dir: Checkpoint location.
        batch_size: Batch size per forward pass.
        batches_per_step: Number of forward passes per step.
        image_filenames: Collection of path to images.
        loop: Run inference in a loop.
        preprocess_fn: Pre-process function to apply to the image before feeding into the graph.
        num_ipus: Number of ipus.
        mode: Inference mode.
        save_graph_pb: If true, export frozen graph to event file to view in Tensorboard

    Returns: Compiled loop operator to run repeated inference over the dataset, infeed_queue intitializer, outfeed op.

    """
    # Model specific config
    with open(config.as_posix()) as file_stream:
        try:
            config_dict = yaml.safe_load(file_stream)
        except yaml.YAMLError as exc:
            tf.logging.error(exc)

    config_dict['network_name'] = config.stem
    if 'dtype' not in config_dict:
        config_dict["dtype"] = 'float16'

    # Create inference optimized frozen graph definition
    network = network_class(input_shape=config_dict["input_shape"],
                            num_outputs=1000, batch_size=batch_size,
                            data_type=config_dict['dtype'],
                            config=config_dict,
                            checkpoint_dir=checkpoint_dir)

    # Export frozen graph to event file to view in Tensorboard"
    if save_graph_pb:
        log_dir = Path(f"{config_dict['network_name']}_graph")
        graph_filename = f"{log_dir}/{config_dict['network_name']}_graph.pb"
        if not log_dir.exists():
            log_dir.mkdir()
        with tf.io.gfile.GFile(graph_filename, "wb") as f:
            f.write(network.optimized_graph.SerializeToString())
        logging.info("%d ops in the final graph." % len(network.optimized_graph.node))
        import_to_tensorboard(graph_filename, log_dir=log_dir.as_posix())

    # Reset graph before creating one on the IPU
    tf.reset_default_graph()

    # Create dataset
    dataset = get_dataset(image_filenames, batch_size, loop=loop, preprocess_fn=preprocess_fn,
                          img_width=config_dict["input_shape"][1],
                          img_height=config_dict["input_shape"][0], dtype=config_dict['dtype'])

    # Set up graph on device, connect infeed and outfeed to the graph.
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def comp_fn():
        def body(img):
            with scopes.ipu_scope('/device:IPU:0'):
                probs = tf.import_graph_def(network.optimized_graph,
                                            input_map={network.graph_input: img},
                                            name="optimized",
                                            return_elements=[network.graph_output])[0]
                outfeed_op = outfeed_queue.enqueue(probs)
                # Note that enqueue happens on the IPU.
                return outfeed_op

        return loops.repeat(batches_per_step,
                            body,
                            [],
                            infeed_queue)

    loop_op = ipu_compiler.compile(comp_fn, [])

    # The dequeue of the outfeed needs to happen on the CPU.
    with tf.device('cpu'):
        outfeed_dequeue = outfeed_queue.dequeue()

    ipu_utils.move_variable_initialization_to_cpu()
    return loop_op, infeed_queue.initializer, outfeed_dequeue


def main(model_arch: str, images: List, batch_size: int,
         batches_per_step: int, loop: bool, num_iterations: int, num_ipus: int, mode: str, data: str,
         available_memory_proportion: float, save_graph_pb: bool, use_ipu_model: bool) -> None:
    """Run inference on chosen model.

    Args:
        model_arch: Type of image classification model
        images: List of images to run inference on.
        batch_size: Batch size per forward pass.
        batches_per_step: Number of batches to run per step.
        loop: Run inference on device in a loop for num_iterations steps.
        num_iterations: Number of iterations, if running in a loop.
        num_ipus: Number of IPU to run inference on.
        mode: Mode of inference - {"single_ipu", "replicated"}
        data: Run on real data (transfer images host -> device) or using on-device synthetic data
        available_memory_proportion: Proportion of tile memory available as
         temporary memory for matmul and convolution execution
        save_graph_pb: If true, export frozen graph to event file to view in Tensorboard
        use_ipu_model: Run code with a CPU based IPU simulator.

    """

    if (available_memory_proportion <= 0.05) or (available_memory_proportion > 1):
        raise ValueError('Invalid "availableMemoryProportion" value: must be a float >=0.05'
                         ' and <=1 (default value is 0.6)')

    if "TF_POPLAR_FLAGS" in os.environ:
        os.environ["TF_POPLAR_FLAGS"] += " --log_cycle_count=0"
    else:
        os.environ["TF_POPLAR_FLAGS"] = "--log_cycle_count=0"

    if data == "synthetic":
        if "TF_POPLAR_FLAGS" in os.environ:
            os.environ["TF_POPLAR_FLAGS"] += " --use_synthetic_data --synthetic_data_initializer=random"
        else:
            os.environ["TF_POPLAR_FLAGS"] = "--use_synthetic_data --synthetic_data_initializer=random"
    else:
        os.environ["TF_POPLAR_FLAGS"] = ""

    if use_ipu_model:
        if "TF_POPLAR_FLAGS" in os.environ:
            os.environ["TF_POPLAR_FLAGS"] += " --use_ipu_model"
        else:
            os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    # Select model architecture
    model_cls = model_dict[model_arch]
    if model_arch == 'googlenet':
        model_arch = 'inceptionv1'
    config = Path(f'configs/{model_arch}.yml')

    # Create graph and data iterator
    loop_op, infeed_initializer, outfeed_op = construct_graph(model_cls, config,
                                                              f"./checkpoints/{model_arch}/",
                                                              batch_size, batches_per_step,
                                                              images, loop,
                                                              model_cls.preprocess_method(), num_ipus,
                                                              mode, save_graph_pb)
    # Run on model or device
    ground_truth = tuple([Path(filename).stem for filename in images])
    run_inference(loop_op, infeed_initializer, outfeed_op, batch_size, batches_per_step, config.stem,
                  model_cls.decode_method(), ground_truth, num_iterations, num_ipus, mode, data,
                  available_memory_proportion=available_memory_proportion)


if __name__ == "__main__":
    """Benchmark image classification inference."""
    parser = argparse.ArgumentParser(description="Run inference on image classification models.")
    parser.add_argument('model_arch', type=str.lower,
                        choices=["googlenet", "inceptionv1", "mobilenet", "mobilenetv2",
                                 "inceptionv3", "resnet50", "densenet121", "xception", "efficientnet-s",
                                 "efficientnet-m", "efficientnet-l"],
                        help="Type of image classification model.")
    parser.add_argument('image_dir', type=str, default="", nargs='?',
                        help="Path to directory of images to run inference on.")
    parser.add_argument('--loop', dest='loop', action='store_true',
                        help="Run inference on device in a loop for `num_iterations` steps.", default=True)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                        help="Batch size for inference.")
    parser.add_argument('--batches-per-step', dest='batches_per_step', type=int, default=100,
                        help="Number of batches to run per step.")
    parser.add_argument('--num-iterations', dest='num_iterations', type=int, default=100,
                        help="Number of iterations if running in a loop.")
    parser.add_argument('--mode', dest="mode", type=str, default="single_ipu",
                        help="Inference mode.", choices=["single_ipu", "replicated"])
    parser.add_argument('--data', dest="data", type=str, default="real",
                        help="Run inference on real data (transfer images host -> device) "
                             "or using on-device synthetic data",
                        choices=["real", "synthetic"])
    parser.add_argument('--num-ipus', dest='num_ipus', type=int, default=1,
                        help="Number of ipus to utilize for inference.")
    parser.add_argument('--use-ipu-model', dest="use_ipu_model",
                        action="store_true",
                        help="Run code with a CPU based IPU simulator.")
    parser.add_argument('--available-mem-prop', dest='available_mem_prop', type=float, default=None,
                        help="Float between 0.05 and 1.0: Proportion of tile memory available as temporary " +
                             "memory for matmul and convolutions execution (default:0.6 and 0.2 for Mobilenetv2)")
    parser.add_argument('--save-graph-pb', dest='save_graph_pb', type=bool, default=False,
                        help="Export frozen graph to event file to view in Tensorboard")

    args = parser.parse_args()

    if (args.mode == 'single_ipu') and (args.num_ipus > 1):
        logging.warning('num_ipus > 1 with single_ipu mode, setting num_ipus to 1. '
                        'To run on multiple ipus, use mode=replicated')
        args.num_ipus = 1

    if (args.mode == 'replicated') and (args.num_ipus == 1):
        raise ValueError(f"num_ipus must be > 1 if mode is {args.mode}.")

    if not args.available_mem_prop:
        if args.model_arch == "mobilenetv2":
            args.available_mem_prop = 0.2
        else:
            args.available_mem_prop = 0.6

    # Check if image dir exists
    if args.data == 'synthetic':
        image_filenames = ['dummy.jpg']
    elif not args.image_dir:
        raise ValueError('When running inference with real data, image directory must be '
                         'supplied as positional argument after the model name.')
    else:
        image_filenames = glob.glob(Path(args.image_dir).as_posix() + "/*.jpg")
    if len(image_filenames) == 0:
        raise ValueError(('Image directory: %s does not have images,'
                          'please run `get_images.sh` '
                          'to download sample imagenet images' % Path(args.image_dir).as_posix()))

    main(args.model_arch, image_filenames, args.batch_size, args.batches_per_step, args.loop,
         args.num_iterations, args.num_ipus, args.mode, args.data, args.available_mem_prop,
         args.save_graph_pb, args.use_ipu_model)
