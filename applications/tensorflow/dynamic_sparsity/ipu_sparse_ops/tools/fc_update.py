# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import argparse
import numpy as np
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from ipu_sparse_ops import sparse, layers
from tensorflow.python.ipu import ipu_outfeed_queue, ipu_compiler
from tensorflow.python.ipu.scopes import ipu_scope

tf.disable_eager_execution()
tf.disable_v2_behavior()


weights = {}


def get_program_arguments():
    parser = argparse.ArgumentParser(description='Sparse checkpoints tool')
    parser.add_argument("--input-size", type=int, default=16,
                        help="Input size of the layer.")
    parser.add_argument("--batchsize", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--output-size", type=int, default=16,
                        help="Output size of the layer.")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                        help="Floating point precision")
    parser.add_argument("--seed", default=0, type=int, help="numpy random seed")
    parser.add_argument("--print-weights", action='store_true',
                        help="Prints the dense fc weights at different "
                        "stages for debug purposes.")
    parser.add_argument('--pooling-type', default='NONE', choices=['NONE', 'SUM', 'AVG', 'MAX'],
                        help="Select dense gradients block pooling")
    return parser.parse_args()


def model(x_fc, fc, opts, outfeed_queue, dtype):
    with tf.variable_scope("SparseOps", reuse=tf.AUTO_REUSE, use_resource=True):
        y_fc = fc(x_fc, tf.constant(False))
        output = {"fc_out": y_fc}
        output['fc_weights'] = tf.convert_to_tensor(fc.get_values_var())
        out = outfeed_queue.enqueue(output)
        return out


def make_fc_weights(input_size, hidden_size, values):
    w = np.zeros([input_size, hidden_size])
    for value in values:
        w[np.random.randint(input_size), np.random.randint(hidden_size)] = value
    return w


def create_sparse_layers(opts, fc_weights):
    def create_sparse_fc_layer(hidden_size, input_shape, name='fc'):
        masked_weights = make_fc_weights(opts.input_size, opts.output_size, fc_weights)
        # Build the fc layer from the masked weights
        triplets = sparse.triplets_from_dense(masked_weights)
        fc = layers.SparseFcLayer.from_triplets(
            opts.output_size, [opts.batchsize, opts.input_size], *triplets,
            matmul_options={"metaInfoBucketOversizeProportion": 1},
            name='sparse_fc_from_triplets',
            dtype=tf.float32 if opts.dtype == 'fp32' else tf.float16,
            use_bias=False, relu=False, pooling_type=opts.pooling_type)
        return fc, masked_weights

    fc, weights = create_sparse_fc_layer(opts.output_size, [opts.batchsize, opts.input_size])

    return fc, weights


def build_update_op(fc):
    with tf.variable_scope("SparseOps", reuse=tf.AUTO_REUSE, use_resource=True):
        # Need to build update ops for each sparse layer
        update_ops = fc.update_sparsity_op()
        # Combine all layer updates into one update op:
        return update_ops


def split_sparse_data(sparse_layers, sparse_layers_data):
    split_data = {}
    for layer_name in sparse_layers.keys():
        split_data[layer_name] = {"nz": {}, "slots": {}}
    for data_name, data in sparse_layers_data.items():
        name = data_name.split("_")
        if "nz" in name:
            split_data[name[0]]["nz"][data_name] = data
    return split_data.values()


def set_up_ipu_devices(opts):
    config = ipu.utils.create_ipu_config()
    config = ipu.utils.auto_select_ipus(config, 1)
    ipu.utils.configure_ipu_system(config)
    # Set the seed for the stochastic rounding
    ipu.utils.reset_ipu_seed = opts.seed


def make_graph(fc_weights):
    graph = tf.Graph()

    with graph.as_default():
        outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="sparse_outfeed")
        fc, weights["init_weights"] = create_sparse_layers(opts, fc_weights)

        model_op = partial(model, fc=fc,
                           opts=opts, outfeed_queue=outfeed_queue,
                           dtype=dtype)

        with tf.device("cpu"):
            x_fc = tf.placeholder(dtype, shape=[opts.batchsize, opts.input_size])

        with ipu_scope('/device:IPU:0'):
            test_op = ipu_compiler.compile(model_op, inputs=[x_fc])

        with tf.device("cpu"):
            fc.create_placeholders()

        with ipu_scope('/device:IPU:0'):
            upload_sparse = build_update_op(fc)

        sparse_feed = {}
        sparse_feed.update(fc.feed_dict())

        dequeue = outfeed_queue.dequeue()
        ipu.utils.move_variable_initialization_to_cpu()


    return graph, outfeed_queue, fc, x_fc, test_op, upload_sparse, dequeue


if __name__ == "__main__":
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    tmp_path = "./tmp/test"
    opts = get_program_arguments()
    set_up_ipu_devices(opts)

    dtype = tf.float32 if opts.dtype == 'fp32' else tf.float16

    np.random.seed(opts.seed)

    x_fc_in = np.random.normal(size=[opts.batchsize, opts.input_size])

    fc_weights = np.random.rand(10)

    # Create a first graph and run it to retrieve the weights from the ipu. Then create a checkpoint
    graph, outfeed_queue, fc, x_fc, test_op, upload_sparse, dequeue = make_graph(fc_weights=fc_weights)

    with tf.Session(graph=graph) as sess:
        # init
        sess.run(tf.global_variables_initializer())

        # run and outfeed weights
        sess.run(test_op, feed_dict={x_fc: x_fc_in})
        results_1 = sess.run(dequeue)

        # Update position of the nz only
        new_triplets = sparse.triplets_from_dense(make_fc_weights(opts.input_size, opts.output_size, fc_weights))
        fc.update_triplets(new_triplets)
        sparse_feed = fc.feed_dict()
        sess.run(upload_sparse, sparse_feed)

        sess.run(test_op, feed_dict={x_fc: x_fc_in})
        results_2 = sess.run(dequeue)

        # update all weights
        fc_weights = np.random.rand(10)
        new_triplets_2 = sparse.triplets_from_dense(make_fc_weights(opts.input_size, opts.output_size, fc_weights))
        fc.update_triplets(new_triplets_2)
        sparse_feed = fc.feed_dict()
        sess.run(upload_sparse, sparse_feed)

        sess.run(test_op, feed_dict={x_fc: x_fc_in})
        results_3 = sess.run(dequeue)

    if opts.print_weights:
        print("init triplets:\n", sparse.triplets_from_dense(weights["init_weights"]),
              "\nuploaded weights:\n", new_triplets, results_1, "\nresults_1:\n",
              results_1['fc_out'], "\nresults_2:\n", results_2['fc_out'])
        print("weights_1:\n", results_1['fc_weights'], "\nweights_3:\n", results_3['fc_weights'])

    weight_change = np.allclose(results_1['fc_out'], results_3['fc_out'])
    meta_change = np.allclose(results_1['fc_out'], results_2['fc_out'])
    if meta_change or weight_change:
        debug = "The results are the same which means the weights haven't been correctly updated."
        if meta_change:
            if weight_change:
                debug += " Neither the values nor the meta-info got updated"
            else:
                debug += " Only the meta-info are not updated"
        else:
            debug += " Only values were not updated"
        raise Exception(debug)

    print("All results match")
