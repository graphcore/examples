#!/usr/bin/env python3
# Copyright 2019 Graphcore Ltd.

"""
A simple program that uses the PopART library ONNX builder to create
a linear model and then trains it on the MNIST data set.
"""

import numpy as np
import popart
import struct
import argparse
from collections import namedtuple

Session = namedtuple('Session', ['session', 'anchors'])

ROWS = 28
COLS = 28


def load_mnist():
    def _readfile(path):
        with open(path, "rb") as f:
            magic_number, num_items = struct.unpack('>II', f.read(8))
            if magic_number == 2051:
                rows, cols = struct.unpack('>II', f.read(8))
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.reshape([num_items, rows * cols])
                data = data.astype(dtype=np.float32)
                data = data / 255.0
            else:
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.astype(dtype=np.int32)
            return data
    train_data = _readfile("data/train-images-idx3-ubyte")
    train_labels = _readfile("data/train-labels-idx1-ubyte")
    test_data = _readfile("data/t10k-images-idx3-ubyte")
    test_labels = _readfile("data/t10k-labels-idx1-ubyte")

    return train_data, train_labels, test_data, test_labels


def create_model(batch_size):
    """ Create an ONNX protobuf description of a simple linear model.
        This function uses the popart library builder functions to create the
        ONNX description directly. An alternative would be to load an
        exported ONNX protobuf from a file.
    """
    builder = popart.Builder()

    np.random.seed(0)  # For predictable weight initialization

    input_shape = popart.TensorInfo("FLOAT", [batch_size, ROWS * COLS])
    x = builder.addInputTensor(input_shape)

    init_weights = np.random.normal(0, 1, [ROWS * COLS, 10]).astype(np.float32)
    W = builder.addInitializedInputTensor(init_weights)

    y = builder.aiOnnx.matmul([x, W])

    init_biases = np.random.normal(0, 1, [10]).astype(np.float32)
    b = builder.addInitializedInputTensor(init_biases)

    output = builder.aiOnnx.add([y, b], "output")
    builder.addOutputTensor(output)
    probs = builder.aiOnnx.softmax([output])

    label_shape = popart.TensorInfo("INT32", [batch_size])
    label = builder.addInputTensor(label_shape)

    loss = popart.NllLoss(probs, label, "nllLossVal")

    proto = builder.getModelProto()

    return proto, x, label, output, loss


class DataSet:
    def __init__(self, batch_size, batches_per_step, data, labels):
        self.data = data
        self.labels = labels
        self.num_examples = len(data)
        self.batch_size = batch_size
        self.batches_per_step = min(batches_per_step, self.num_examples // self.batch_size)
        self.inputs_per_step = self.batch_size * self.batches_per_step
        self.steps_per_epoch = self.num_examples // self.inputs_per_step

    def __getitem__(self, key):
        input_begin = key * self.inputs_per_step
        input_end = input_begin + self.inputs_per_step
        data = self.data[input_begin:input_end]
        data = data.reshape([self.batches_per_step, self.batch_size, -1])
        labels = self.labels[input_begin:input_end]
        labels = labels.reshape([self.batches_per_step, self.batch_size])
        return data, labels

    def __iter__(self):
        return (self[j] for j in range(self.steps_per_epoch))

    def __len__(self):
        return self.steps_per_epoch


def get_device(num_ipus, sim=True):
    # Select a device
    deviceManager = popart.DeviceManager()
    if sim:
        options = {
            "compileIPUCode": True,
            "numIPUs": num_ipus,
            "tilesPerIPU": 1216
        }
        device = deviceManager.createIpuModelDevice(options)
    else:
        device = deviceManager.acquireAvailableDevice(num_ipus)
        if device is None:
            print("Failed to acquire IPU. Exiting.")
            return None
    return device


def init_session(proto, loss, dataFlow, userOpts, device, training=True):
    # Create a session to compile and execute the graph
    if training:
        session = popart.TrainingSession(fnModel=proto,
                                         losses=[loss],
                                         deviceInfo=device,
                                         optimizer=popart.ConstSGD(0.01),
                                         dataFeed=dataFlow,
                                         userOptions=userOpts)
    else:
        session = popart.InferenceSession(fnModel=proto,
                                          losses=[loss],
                                          deviceInfo=device,
                                          dataFeed=dataFlow,
                                          userOptions=userOpts)

    print("Compiling the {} graph.".format("training" if training else "validation"))

    session.prepareDevice()
    session.setRandomSeed(1)
    if training:
        session.optimizerFromHost()

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return Session(session, anchors)


def train(opts):
    train_data, train_labels, test_data, test_labels = load_mnist()

    # Limit batches_per_step so the test set isn't evaluated more than once.
    max_value = len(test_data) // opts.batch_size
    if max_value < opts.batches_per_step:
        print("(batches-per-step * batch-size) is larger than test set!\n"
              " Reduced batches-per-step to: {}\n".format(max_value))
        opts.batches_per_step = max_value

    training_set = DataSet(opts.batch_size, opts.batches_per_step, train_data, train_labels)
    test_set = DataSet(opts.batch_size, opts.batches_per_step, test_data, test_labels)

    print("Creating ONNX model.")
    proto, data_in, labels_in, output, loss = create_model(opts.batch_size)

    # Describe how to run the model
    anchor_desc = {output: popart.AnchorReturnType("ALL"),
                   loss.output(0): popart.AnchorReturnType("ALL")}
    dataFlow = popart.DataFlow(opts.batches_per_step, anchor_desc)

    # Options
    userOpts = popart.SessionOptions()
    userOpts.logging = {
        'ir': 'TRACE' if opts.log_graph_trace else 'CRITICAL',
        'devicex': 'CRITICAL'
    }
    # The validation graph by default will be optimized to change all variables to constants
    # This prevents that, which allows for checkpoints to be loaded into the model without recompiling
    userOpts.constantWeights = False

    # Enable auto-sharding
    if opts.num_ipus > 1:
        userOpts.enableVirtualGraphs = True
        userOpts.virtualGraphMode = popart.VirtualGraphMode.Auto

    # Enable pipelining
    if opts.pipeline:
        userOpts.enablePipelining = True

    # A single device is shared between training and validation sessions
    device = get_device(opts.num_ipus, opts.simulation)

    training = init_session(proto, loss, dataFlow, userOpts, device, training=True)
    validation = init_session(proto, loss, dataFlow, userOpts, device, training=False)

    print("Running training loop.")
    for i in range(opts.epochs):
        # Training
        if i > 0:
            training.session.resetHostWeights('ckpt.onnx')
        training.session.weightsFromHost()
        for data, labels in training_set:
            stepio = popart.PyStepIO({data_in: data, labels_in: labels}, training.anchors)
            training.session.run(stepio)

        aggregated_loss = 0
        aggregated_accuracy = 0

        training.session.modelToHost('ckpt.onnx')
        validation.session.resetHostWeights('ckpt.onnx')
        validation.session.weightsFromHost()

        # Evaluation
        for data, labels in test_set:
            stepio = popart.PyStepIO({data_in: data, labels_in: labels}, validation.anchors)
            validation.session.run(stepio)
            # Loss
            aggregated_loss += np.mean(validation.anchors["nllLossVal"])
            # Accuracy
            results = np.argmax(validation.anchors[output].reshape([test_set.inputs_per_step, 10]), 1)
            num_correct = np.sum(results == labels.reshape([test_set.inputs_per_step]))
            aggregated_accuracy += num_correct / test_set.inputs_per_step

        # Log statistics
        aggregated_loss /= len(test_set)
        aggregated_accuracy /= len(test_set)
        print("Epoch #{}".format(i + 1))
        print("   Loss={0:.4f}".format(aggregated_loss))
        print("   Accuracy={0:.2f}%".format(aggregated_accuracy * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST training in Popart',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Set the Batch size")
    parser.add_argument('--batches-per-step', type=int, default=100,
                        help="Number of minibatches to perform on the Device before returning to the Host."
                        " This will be capped so the Device returns each epoch.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs to train for.")
    parser.add_argument('--num-ipus', type=int, default=1,
                        help="Number of IPU's")
    parser.add_argument('--pipeline', action="store_true", default=False,
                        help="Pipeline the model over IPUs")
    parser.add_argument('--simulation', action='store_true',
                        help="Run the example with an IPU_MODEL device.")
    parser.add_argument('--log-graph-trace', action='store_true',
                        help="Turn on ir logging to display the graph's ops.")
    opts = parser.parse_args()

    train(opts)
