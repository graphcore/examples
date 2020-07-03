#!/usr/bin/env python3
# Copyright 2020 Graphcore Ltd.

"""
A simple program that uses the PopART library ONNX builder to create
a model and then trains it on the MNIST data set.
"""
import argparse
import os
import struct
import tempfile
from collections import namedtuple

import numpy as np
import popart
from time import time

Session = namedtuple('Session', ['session', 'anchors'])

ROWS = 28
COLS = 28


def kaiming_init(shape, fan_in, a=5.0, b=3.0):
    stddev = np.sqrt(a) / np.sqrt(fan_in)
    bound = np.sqrt(b) * stddev
    return np.random.uniform(-bound, bound, shape).astype(np.float32)


def load_mnist():
    def _readfile(path):
        with open(path, 'rb') as f:
            magic_number, num_items = struct.unpack('>II', f.read(8))
            if magic_number == 2051:
                rows, cols = struct.unpack('>II', f.read(8))
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.reshape([num_items, 1, rows, cols])
                data = data.astype(dtype=np.float32) / 255.0
                mean = np.mean(data)
                data = data - mean
                std = np.std(data)
                data = data - std
            else:
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.astype(dtype=np.int32)
            return data
    train_data = _readfile('data/train-images-idx3-ubyte')
    train_labels = _readfile('data/train-labels-idx1-ubyte')
    test_data = _readfile('data/t10k-images-idx3-ubyte')
    test_labels = _readfile('data/t10k-labels-idx1-ubyte')

    return train_data, train_labels, test_data, test_labels


def load_dummy(cl_opts: argparse.Namespace):
    def _generate_data(cls_opts):
        """
        Generates random data based provided command line arguments

        Args:
            cls_opts: The command line arguments

        Returns:
            Arrays with random values for data and labels
        """
        input_shape = [cl_opts.batches_per_step * cl_opts.batch_size, 1, ROWS, COLS]
        data = np.zeros(input_shape, np.float32) if cl_opts.syn_data_type == "zeros" \
            else np.random.normal(0, 1, input_shape).astype(np.float32)

        label_shape = [cl_opts.batches_per_step * cl_opts.batch_size]
        label_data = np.zeros(label_shape, np.int32) if cl_opts.syn_data_type == "zeros" \
            else np.random.uniform(0, 10, label_shape).astype(np.int32)

        return data, label_data

    train_data, train_labels = _generate_data(cl_opts)
    test_data, test_labels = _generate_data(cl_opts)

    return train_data, train_labels, test_data, test_labels


def create_model(batch_size):
    """ Create an ONNX protobuf description of a simple model.
        This function uses the popart library builder functions to create the
        ONNX description directly. An alternative would be to load an
        exported ONNX protobuf from a file.
    """
    builder = popart.Builder()

    input_shape = popart.TensorInfo('FLOAT', [batch_size, 1, ROWS, COLS])
    input_t = builder.addInputTensor(input_shape)
    x = input_t

    init_weights = kaiming_init([20, 1, 5, 5], 1 * 5 * 5)
    W1 = builder.addInitializedInputTensor(init_weights)
    init_weights = kaiming_init([20], 1 * 5 * 5, 1, 1)
    b1 = builder.addInitializedInputTensor(init_weights)

    x = builder.aiOnnx.conv([x, W1, b1],
                            dilations=[1, 1],
                            kernel_shape=[5, 5],
                            strides=[1, 1],
                            pads=[0, 0, 0, 0])

    x = builder.aiOnnx.relu([x])
    (x,) = builder.aiOnnx.maxpool([x],
                                  num_outputs=1,
                                  kernel_shape=[2, 2],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2])

    init_weights = kaiming_init([50, 20, 5, 5], 20 * 5 * 5)
    W2 = builder.addInitializedInputTensor(init_weights)
    init_weights = kaiming_init([50], 20 * 5 * 5, 1, 1)
    b2 = builder.addInitializedInputTensor(init_weights)

    x = builder.aiOnnx.conv([x, W2, b2],
                            dilations=[1, 1],
                            kernel_shape=[5, 5],
                            strides=[1, 1],
                            pads=[0, 0, 0, 0])

    x = builder.aiOnnx.relu([x])
    (x,) = builder.aiOnnx.maxpool([x],
                                  num_outputs=1,
                                  kernel_shape=[2, 2],
                                  pads=[0, 0, 0, 0],
                                  strides=[2, 2])

    shape = builder.aiOnnx.constant(np.asarray([batch_size, 50 * 4 ** 2]))
    x = builder.aiOnnx.reshape([x, shape])

    init_weights = kaiming_init([50 * 4 ** 2, 500], 50 * 4 ** 2)
    W3 = builder.addInitializedInputTensor(init_weights)
    init_weights = kaiming_init([500], 50 * 4 ** 2, 1, 1)
    b3 = builder.addInitializedInputTensor(init_weights)

    x = builder.aiOnnx.matmul([x, W3])
    x = builder.aiOnnx.add([x, b3])
    x = builder.aiOnnx.relu([x])

    init_weights = kaiming_init([500, 10], 500)
    W4 = builder.addInitializedInputTensor(init_weights)
    init_weights = kaiming_init([10], 500, 1, 1)
    b4 = builder.addInitializedInputTensor(init_weights)

    x = builder.aiOnnx.matmul([x, W4])
    output_t = builder.aiOnnx.add([x, b4])

    builder.addOutputTensor(output_t)
    probs = builder.aiOnnx.softmax([output_t])

    label_shape = popart.TensorInfo('INT32', [batch_size])
    label = builder.addInputTensor(label_shape)

    loss = builder.aiGraphcore.nllloss([probs, label], popart.ReductionType.Sum, debugPrefix='nllLossVal')

    proto = builder.getModelProto()

    return proto, input_t, label, output_t, loss


class DataSet:
    def __init__(self, batch_size, batches_per_step, data, labels):
        self.data = data
        self.labels = labels
        self.num_examples = len(data)
        self.batch_size = batch_size
        self.batches_per_step = min(
            batches_per_step, self.num_examples // self.batch_size)
        self.inputs_per_step = self.batch_size * self.batches_per_step
        self.steps_per_epoch = self.num_examples // self.inputs_per_step

    def __getitem__(self, key):
        input_begin = key * self.inputs_per_step
        input_end = input_begin + self.inputs_per_step
        data = self.data[input_begin:input_end]
        data = data.reshape(
            [self.batches_per_step, self.batch_size, 1, ROWS, COLS])
        labels = self.labels[input_begin:input_end]
        labels = labels.reshape([self.batches_per_step, self.batch_size])
        return data, labels

    def __iter__(self):
        return (self[j] for j in range(self.steps_per_epoch))

    def __len__(self):
        return self.steps_per_epoch


def get_device(sim=True):
    # Select a device
    deviceManager = popart.DeviceManager()
    if sim:
        options = {'compileIPUCode': True, 'numIPUs': 1, 'tilesPerIPU': 1216}
        device = deviceManager.createIpuModelDevice(options)
    else:
        device = deviceManager.acquireAvailableDevice()
        if device is None:
            print('Failed to acquire IPU. Exiting.')
            return None
    return device


def init_session(proto, loss, dataFlow, userOpts, device, training=True):
    # Create a session to compile and execute the graph
    if opts.test_mode:
        userOpts.instrumentWithHardwareCycleCounter = True

    if training:
        session = popart.TrainingSession(fnModel=proto,
                                         loss=loss,
                                         deviceInfo=device,
                                         optimizer=popart.ConstSGD(0.001),
                                         dataFlow=dataFlow,
                                         userOptions=userOpts)
    else:
        session = popart.InferenceSession(fnModel=proto,
                                          deviceInfo=device,
                                          dataFlow=dataFlow,
                                          userOptions=userOpts)

    print('Compiling the {} graph.'.format(
        'training' if training else 'validation'))
    session.prepareDevice()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return Session(session, anchors)


def log_run_info(session, start_time, batch_size, batches_per_step):
    duration = time() - start_time
    report_string = "{0:<8.3} sec/itr. {1:5f} images/sec.".format(
        duration,
        batch_size * batches_per_step / duration
    )
    print(report_string)
    print(
        "Hardware cycle count per 'run':",
        session.session.getCycleCount()
    )
    print("Total time: {}".format(duration))


def train(opts):
    # Do not require the mnist data to be present if running with synthetic data
    train_data, train_labels, test_data, test_labels = load_dummy(opts) \
        if opts.syn_data_type in ["random_normal", "zeros"] else load_mnist()

    if not opts.test_mode:
        max_value = len(test_data) // opts.batch_size
        if max_value < opts.batches_per_step:
            print("(batches-per-step * batch-size) is larger than test set!\n"
                  " Reduced batches-per-step to: {}\n".format(max_value))
            opts.batches_per_step = max_value
    training_set = DataSet(opts.batch_size, opts.batches_per_step, train_data, train_labels)
    test_set = DataSet(opts.batch_size, opts.batches_per_step, test_data, test_labels)

    print('Creating ONNX model.')
    proto, data_in, labels_in, output, loss = create_model(opts.batch_size)

    # Describe how to run the model
    anchor_desc = {output: popart.AnchorReturnType('ALL'),
                   loss: popart.AnchorReturnType('ALL')}
    dataFlow = popart.DataFlow(opts.batches_per_step, anchor_desc)

    # Options
    userOpts = popart.SessionOptions()

    # The validation graph by default will be optimized to change all variables to constants
    # This prevents that, which allows for checkpoints to be loaded into the model without recompiling
    userOpts.constantWeights = False

    # If requested, setup synthetic data
    if opts.syn_data_type in ["random_normal", "zeros"]:
        print(
            "Running with Synthetic Data Type '{}'".format(opts.syn_data_type)
        )
        if opts.syn_data_type == "random_normal":
            userOpts.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        elif opts.syn_data_type == "zeros":
            userOpts.syntheticDataMode = popart.SyntheticDataMode.Zeros

    # A single device is shared between training and validation sessions
    device = get_device(opts.simulation)

    training = init_session(proto, loss, dataFlow,
                            userOpts, device, training=True)
    validation = init_session(proto, loss, dataFlow,
                              userOpts, device, training=False)

    # Make weight transfer file
    _, onnx_file_name = tempfile.mkstemp()

    training.session.weightsFromHost()
    training.session.modelToHost(onnx_file_name)

    print('Running training loop.')
    for i in range(opts.epochs):
        # Training
        if i > 0:
            training.session.resetHostWeights(onnx_file_name)
        training.session.weightsFromHost()
        for data, labels in training_set:
            stepio = popart.PyStepIO(
                {data_in: data, labels_in: labels}, training.anchors)

            start = time()
            training.session.run(stepio)
            if opts.test_mode == "training":
                log_run_info(training,
                             start,
                             opts.batch_size,
                             opts.batches_per_step)

        aggregated_loss = 0
        aggregated_accuracy = 0

        training.session.modelToHost(onnx_file_name)
        validation.session.resetHostWeights(onnx_file_name)
        validation.session.weightsFromHost()

        # Evaluation
        for data, labels in test_set:
            stepio = popart.PyStepIO(
                {data_in: data, labels_in: labels}, validation.anchors)

            start = time()
            validation.session.run(stepio)
            if opts.test_mode == "inference":
                log_run_info(validation,
                             start,
                             opts.batch_size,
                             opts.batches_per_step)

            # Loss
            aggregated_loss += np.mean(validation.anchors[loss])
            # Accuracy
            results = np.argmax(validation.anchors[output].reshape(
                [test_set.inputs_per_step, 10]), 1)
            num_correct = np.sum(results == labels.reshape(
                [test_set.inputs_per_step]))
            aggregated_accuracy += num_correct / test_set.inputs_per_step

        # Log statistics
        aggregated_loss /= len(test_set)
        aggregated_accuracy /= len(test_set)
        print('Epoch #{}'.format(i + 1))
        print('   Loss={0:.4f}'.format(aggregated_loss))
        print('   Accuracy={0:.2f}%'.format(aggregated_accuracy * 100))

    # Remove weight transfer file
    os.remove(onnx_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST training in Popart',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Set the Batch size')
    parser.add_argument(
        '--batches-per-step',
        type=int,
        default=100,
        help='Number of minibatches to perform on the Device before returning to the Host.'
             ' This will be capped so the Device returns each epoch.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train for.')
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Run the example with an IPU_MODEL device.')
    parser.add_argument(
        '--log-graph-trace',
        action='store_true',
        help='Turn on ir logging to display the graph\'s ops.')
    parser.add_argument(
        "--test-mode",
        choices=['training', 'inference'],
        help="Output extra performance information, specify wit"
        "h either 'training' or 'inference'",
    )
    parser.add_argument(
        "--syn-data-type",
        choices=['random_normal', 'zeros'],
        default="off",
        help="Specify to use synthetic data with either 'random"
             "_normal' or 'zeros' (no host to IPU IO done in this mode)",
    )
    opts = parser.parse_args()

    # Set logging
    popart.getLogger('ir').setLevel('TRACE' if opts.log_graph_trace else 'CRITICAL')
    popart.getLogger('devicex').setLevel('CRITICAL')

    train(opts)
