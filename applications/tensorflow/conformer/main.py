# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
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
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
from model import ConformerAM, AMConfig

tf.compat.v1.disable_eager_execution()


def seed_all(seed):
    random.seed(seed)
    tf.compat.v1.set_random_seed(random.randint(0, 2**32 - 1))
    np.random.seed(random.randint(0, 2**32 - 1))
    ipu.utils.reset_ipu_seed(random.randint(-2**16, 2**16 - 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for acoustic model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default="configs/train_fp32_kl_loss.yaml",
                        help='Model config file')
    parser.add_argument('--save-checkpoint', type=bool, default=True,
                        help="Save checkpoint or not")
    parser.add_argument('--freeze', type=bool, default=False,
                        help="Save frozen pb model")
    parser.add_argument('--logfile', type=str, default='model.log',
                        help="log file for model parameters")
    parser.add_argument('--seed', type=int, default=1991,
                        help='set random seed for all')
    parser.add_argument('--data-path', type=str, default='data/train',
                        help='set training data path')
    parser.add_argument('--dict-path', type=str, default='./sample_train_units.txt',
                        help='set training data vocab path')
    parser.add_argument('--use-synthetic-data', type=str, default=True,
                        help='if use synthetic data')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Name of the Weights&Biases run, disabled if None')

    args = parser.parse_args()
    seed_all(args.seed)
    config = AMConfig.from_yaml(args.config, **vars(args))
    conformeram = ConformerAM(config)
    conformeram.run_with_pipeline()
