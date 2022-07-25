# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

import re
import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import argparse


def find_ckpts(dir):
    filename_pattern = re.compile(".*ckpt-[0-9]+$")
    ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")
    filenames = sorted([os.path.join(dir, f[:-len(".index")])
                        for f in os.listdir(dir)
                        if filename_pattern.match(f[:-len(".index")]) and f[-len(".index"):] == ".index"],
                       key=lambda x: int(ckpt_pattern.match(x).groups()[0]))
    return filenames


def read_ckpt(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    keys = sorted(reader.get_variable_to_shape_map().keys())
    return {k: reader.get_tensor(k).astype(np.float64) for k in keys}


def get_dtypes(filename):
    return pywrap_tensorflow.NewCheckpointReader(filename).get_variable_to_dtype_map()


def get_shapes(filename):
    return pywrap_tensorflow.NewCheckpointReader(filename).get_variable_to_shape_map()


def average_ckpts(ckpts, mode='mean', decay=0.9):
    V = read_ckpt(ckpts[0])
    if mode == 'mean':
        a, b, s = 1.0, 1 / len(ckpts), 1 / len(ckpts)
    elif mode == 'exponential':
        a, b, s = decay, 1 - decay, 1.0
    else:
        raise ValueError("mode {} not recognised".format(mode))

    for k in V.keys():
        V[k] = s * V[k]

    for ckpt in ckpts[1:]:
        R = read_ckpt(ckpt)
        for k in V.keys():
            V[k] = (a * V[k]) + (b * R[k])
    return V


def correct_dtypes(V, dtypes):
    assert set(V.keys()) == set(dtypes.keys())
    conversion_dict = {tf.float16: np.float16,
                       tf.float32: np.float32,
                       tf.float64: np.float64,
                       tf.int32: np.int32,
                       tf.int64: np.int64}
    for k in V.keys():
        V[k] = V[k].astype(conversion_dict[dtypes[k]])
    return V


def save_ckpt(V, ref_ckpt, filename):
    dtypes = get_dtypes(ref_ckpt)
    shapes = get_shapes(ref_ckpt)
    V = correct_dtypes(V, dtypes)
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        with tf.device('/device:CPU:0'):
            for k in V.keys():
                var = tf.get_variable(k,
                                      dtype=dtypes[k],
                                      initializer=V[k])
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    sess = tf.Session(graph=tf_graph)
    sess.run(init)
    saver.save(sess, filename)


def add_main_arguments(parser):
    group = parser.add_argument_group('Main')
    group.add_argument('--restore-path', type=str, required=True,
                       help="Path to a directory containing multiple checkpoints")
    group.add_argument('--filename', type=str, required=True,
                       help="filename for saved checkpint")
    group.add_argument('-N', type=int,
                       help="Number of checkpoints, counted back from final, to consider")
    group.add_argument('--mode', type=str, required=True, choices=['mean', 'exponential'],
                       help="Method used for averaging")
    group.add_argument('--decay', type=float, default=0.99,
                       help="Decay factor used for exponential averaging mode")
    group.add_argument('--discard-last', type=int, default=0,
                       help="Discard last N checkpoints")
    return parser


def main():
    parser = argparse.ArgumentParser("Weight Averaging Function")
    parser = add_main_arguments(parser)
    args, unknown = parser.parse_known_args()

    ckpts = find_ckpts(args.restore_path)
    print("Found {} ckpts in {}".format(len(ckpts), args.restore_path))
    if args.discard_last:
        print("Discarding last {} checkpoints".format(args.discard_last))
        ckpts = ckpts[:-args.discard_last]
    if args.N:
        N = min(args.N, len(ckpts))
        ckpts = ckpts[-N:]
        print("Using final {}".format(N))
    print("Averaging using mode: {}".format(args.mode))
    if args.mode == 'exponential':
        print("With decay factor {}".format(args.decay))
    V = average_ckpts(ckpts, mode=args.mode, decay=args.decay)
    print("Completed averaging of {} checkpoints".format(len(ckpts)))
    save_ckpt(V, ckpts[0], args.filename)
    print("Weights saved to {}".format(args.filename))


if __name__ == '__main__':
    main()
