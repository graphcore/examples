#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ipu import ipu_compiler, scopes, utils
from tensorflow.python.framework import errors


def device_connection_type(value):
    dcts = {"ALWAYS": utils.DeviceConnectionType.ALWAYS,
            "ON_DEMAND": utils.DeviceConnectionType.ON_DEMAND,
            "NEVER": utils.DeviceConnectionType.NEVER}
    return dcts.get(value)


def my_graph(pa, pb, pc):
    o1 = pa + pb
    o2 = pa + pc
    result = o1 + o2
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection_type",
                        choices=['ALWAYS', 'ON_DEMAND', 'NEVER'],
                        help="Specify connection type")
    parser.set_defaults(connection_type='ALWAYS')
    opts = parser.parse_args()

    with tf.device("cpu"):
        pa = tf.compat.v1.placeholder(np.float32, [2], name="a")
        pb = tf.compat.v1.placeholder(np.float32, [2], name="b")
        pc = tf.compat.v1.placeholder(np.float32, [2], name="c")

    # Create the IPU section of the graph.
    with scopes.ipu_scope("/device:IPU:0"):
        out = ipu_compiler.compile(my_graph, [pa, pb, pc])

    # Define the feed_dict input data.
    fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}

    # Connection type from options.
    connection_type = device_connection_type(opts.connection_type)

    cfg = utils.create_ipu_config()
    cfg = utils.auto_select_ipus(cfg, 1)
    cfg = utils.set_ipu_connection_type(cfg,
                                        connection_type,
                                        1)
    utils.configure_ipu_system(cfg)

    # Run the session.
    # If running with DeviceConnectionType.NEVER then anticipate the
    # specific exception with message "configured for compilation only".
    with tf.compat.v1.Session() as sess:
        try:
            result = sess.run(out, fd)
            print(result)
        except tf.errors.InvalidArgumentError as invalid_arg_exception:
            if (connection_type == utils.DeviceConnectionType.NEVER) and \
               ("configured for compilation only" in invalid_arg_exception.message):
                print("Compiled")
                pass
            else:
                print("ERROR: {}".format(invalid_arg_exception.message))
        except:
            general_exception = sys.exc_info()[0]
            print("ERROR: {}".format(general_exception))


if __name__ == '__main__':
    main()
