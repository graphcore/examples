#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
A simple program that uses the PopART library ONNX builder to create
a linear model and then trains it on the MNIST data set.
"""
import argparse
import os
import struct
import tempfile
from collections import namedtuple
from time import time

import numpy as np
import popart

Session = namedtuple("Session", ["session", "anchors"])

ROWS = 28
COLS = 28


def load_mnist():
    def _readfile(path):
        with open(path, "rb") as f:
            magic_number, num_items = struct.unpack(">II", f.read(8))
            if magic_number == 2051:
                rows, cols = struct.unpack(">II", f.read(8))
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.reshape([num_items, rows * cols])
                data = data.astype(dtype=np.float32)
                data = data / 255.0
            else:
                data = np.fromstring(f.read(), dtype=np.uint8)
                data = data.astype(dtype=np.int32)
            return data

    goal_dir = os.path.join(os.getcwd(), "data")
    train_data = _readfile(os.path.join(goal_dir, "train-images-idx3-ubyte"))
    train_labels = _readfile(os.path.join(goal_dir, "train-labels-idx1-ubyte"))
    test_data = _readfile(os.path.join(goal_dir, "t10k-images-idx3-ubyte"))
    test_labels = _readfile(os.path.join(goal_dir, "t10k-labels-idx1-ubyte"))

    return train_data, train_labels, test_data, test_labels


def load_dummy(cl_opts: argparse.Namespace):
    def _generate_data(cls_opts):
        input_shape = [cl_opts.batches_per_step * cl_opts.batch_size, 1, ROWS, COLS]
        data = (
            np.zeros(input_shape, np.float32)
            if cl_opts.syn_data_type == "zeros"
            else np.random.normal(0, 1, input_shape).astype(np.float32)
        )

        label_shape = [cl_opts.batches_per_step * cl_opts.batch_size]
        label_data = (
            np.zeros(label_shape, np.int32)
            if cl_opts.syn_data_type == "zeros"
            else np.random.uniform(0, 10, label_shape).astype(np.int32)
        )

        return data, label_data

    train_data, train_labels = _generate_data(cl_opts)
    test_data, test_labels = _generate_data(cl_opts)

    return train_data, train_labels, test_data, test_labels


def create_model(batch_size):
    """Create an ONNX protobuf description of a simple linear model.
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

    loss = builder.aiGraphcore.nllloss([probs, label], popart.ReductionType.Sum, debugContext="nllLossVal")

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
        options = {"compileIPUCode": True, "numIPUs": num_ipus, "tilesPerIPU": 1216}
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
        session = popart.TrainingSession(
            fnModel=proto,
            loss=loss,
            deviceInfo=device,
            optimizer=popart.ConstSGD(0.01),
            dataFlow=dataFlow,
            userOptions=userOpts,
        )
    else:
        session = popart.InferenceSession(fnModel=proto, deviceInfo=device, dataFlow=dataFlow, userOptions=userOpts)

    print(f"Compiling the {'training' if training else 'validation'} graph.")

    session.prepareDevice()
    session.setRandomSeed(1)

    # Create buffers to receive results from the execution
    anchors = session.initAnchorArrays()

    return Session(session, anchors)


def log_run_info(session, start_time, cl_opts):
    duration = time() - start_time
    image_rate = cl_opts.batch_size * cl_opts.batches_per_step / duration
    report_string = f"{duration:<8.3} sec/itr. {image_rate:5f} images/sec."
    print(report_string)
    print("Hardware cycle count per 'run':", session.session.getCycleCount())
    print(f"Total time: {duration}")


def train():
    batch_size = 32
    batches_per_step = 100
    epochs = 10
    pipeline = False
    replication_factor = 1
    samples_per_device = 32
    num_ipus = 1
    simulation = False

    train_data, train_labels, test_data, test_labels = load_mnist()

    max_value = len(test_data) // batch_size
    if max_value < batches_per_step:
        print(
            "(batches-per-step * batch-size) is larger than test set!\n" f" Reduced batches-per-step to: {max_value}\n"
        )
        batches_per_step = max_value
    training_set = DataSet(batch_size, batches_per_step, train_data, train_labels)
    test_set = DataSet(batch_size, batches_per_step, test_data, test_labels)

    print("Creating ONNX model.")
    proto, data_in, labels_in, output, loss = create_model(samples_per_device)

    # Describe how to run the model
    anchor_desc = {
        output: popart.AnchorReturnType("ALL"),
        loss: popart.AnchorReturnType("ALL"),
    }
    dataFlow = popart.DataFlow(batches_per_step, anchor_desc)

    # Options
    userOpts = popart.SessionOptions()

    # The validation graph by default will be optimized to change all variables to constants
    # This prevents that, which allows for checkpoints to be loaded into the model without recompiling
    userOpts.constantWeights = False

    # A single device is shared between training and validation sessions
    device = get_device(num_ipus, simulation)

    training = init_session(proto, loss, dataFlow, userOpts, device, training=True)
    validation = init_session(proto, loss, dataFlow, userOpts, device, training=False)

    return (
        training,
        validation,
        training_set,
        test_set,
        data_in,
        labels_in,
        loss,
        output,
    )
