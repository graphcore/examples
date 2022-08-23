# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import argparse
import logging
import os
from pathlib import Path

import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import serving
from tensorflow.python.ipu.config import IPUConfig
import yaml


from inference_networks import model_dict

# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

tf.logging.set_verbosity("INFO")


def main(model_arch: str, batch_size: int, num_iterations: int, num_ipus: int,
         available_memory_proportion: float) -> None:

    if (available_memory_proportion < 0.05) or (available_memory_proportion > 1):
        raise ValueError('Invalid "availableMemoryProportion" value: must be a float >=0.05'
                         ' and <=1 (default value is 0.6)')

    # Select model architecture
    network_class = model_dict[model_arch]
    if model_arch == 'googlenet':
        model_arch = 'inceptionv1'
    config = Path(f'configs/{model_arch}.yml')

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
                            checkpoint_dir=f"./checkpoints/{model_arch}/")

    # Configure the IPU for compilation.
    cfg = IPUConfig()
    cfg.auto_select_ipus = num_ipus
    cfg.configure_ipu_system()

    # Export for serving
    input_signature = (tf.TensorSpec(shape=(
        batch_size,
        config_dict["input_shape"][0],
        config_dict["input_shape"][1],
        config_dict["input_shape"][2]),
        dtype=config_dict['dtype'],
        name='img_input'),)
    print("Export for serving")

    def network_func(in_img):
        return tf.import_graph_def(network.optimized_graph,
                                   input_map={network.graph_input: in_img},
                                   name="optimized",
                                   return_elements=[network.graph_output])[0]
    # Directory where SavedModel will be written
    saved_model_dir = os.path.join(config_dict['network_name'], '001')
    # Export as a SavedModel.
    serving.export_single_step(network_func, saved_model_dir,
                               num_iterations, input_signature)
    print("SavedModel written to", saved_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export image classification models for serving.")
    parser.add_argument('model_arch', type=str.lower,
                        choices=["googlenet", "inceptionv1", "mobilenet", "mobilenetv2",
                                 "inceptionv3", "resnet50", "densenet121", "xception", "efficientnet-s",
                                 "efficientnet-m", "efficientnet-l"],
                        help="Type of image classification model.")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                        help="Batch size for inference.")
    parser.add_argument('--num-iterations', dest='num_iterations', type=int, default=1024,
                        help="Number of iterations if running in a loop.")
    parser.add_argument('--num-ipus', dest='num_ipus', type=int, default=1,
                        help="Number of ipus to utilize for inference.")
    parser.add_argument('--available-mem-prop', dest='available_mem_prop', type=float, default=None,
                        help="Float between 0.05 and 1.0: Proportion of tile memory available as temporary " +
                             "memory for matmul and convolutions execution (default:0.6 and 0.2 for Mobilenetv2)")

    args = parser.parse_args()

    if not args.available_mem_prop:
        if args.model_arch == "mobilenetv2":
            args.available_mem_prop = 0.2
        else:
            args.available_mem_prop = 0.6

    main(args.model_arch, args.batch_size, args.num_iterations, args.num_ipus,
         args.available_mem_prop)
