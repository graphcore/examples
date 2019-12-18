#!/usr/bin/env python3
# Copyright 2019 Graphcore Ltd.

"""
A simple program that uses PyTorch to create a linear model and then
trains it on the MNIST data set using the popart library.

"""

import argparse
import numpy as np
import popart
import struct
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Tuple
from collections import namedtuple

# Constants for the MNIST dataset
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NUM_CLASSES = 10
NUM_TEST_SAMPLES = 10000

# Constants for IPU emulator
TILES_PER_IPU = 1216


class Net(nn.Module):
    """Neural network module that defines the simple linear model to
    classify MNIST digits.

    Attributes:
        fc: Fully connected layer between input and output.
    """

    def __init__(self) -> None:
        """Initialize.
        """

        super(Net, self).__init__()
        self.fc = nn.Linear(IMAGE_WIDTH * IMAGE_HEIGHT, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass.

        Args:
            x : Image input tensor.

        Returns:
            Softmax output probabilities per class.
        """
        x = self.fc(x)
        return F.softmax(x, dim=1)


def create_model(batch_size: int) -> Tuple[str, str, str]:
    """Create Pytorch model and export as an ONNX protobuf.

    Args:
        batch_size : Batch size of the model.

    Returns:
        Filename of onnx binary protobuf file,
        image_input name, output_name
    """
    net = Net()
    image_input = "input_1"
    output = "output_1"
    input_names = [image_input] + [
        "learned_%d" % i for i, _ in enumerate(net.parameters())
    ]
    dummy_input = torch.randn(batch_size, IMAGE_WIDTH * IMAGE_HEIGHT)
    protobuf_file = "net.onnx"
    torch.onnx.export(
        net, dummy_input, protobuf_file, input_names=input_names, output_names=[output]
    )
    return protobuf_file, image_input, output


def get_data_loader(
    batch_size: int, batches_per_step: int, is_train: bool
) -> torch.utils.data.DataLoader:
    """Get dataloader for training/testing.

    Args:
        batch_size: Number of samples in one batch.
        batches_per_step: Number of mini-batches to process before returning to the host.
        is_train: Flag is True if training.

    Returns:
        Dataloader for the split requested.
    """

    return torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=is_train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))]),),
        batch_size=batch_size * batches_per_step,
        shuffle=is_train,
    )


def convert_model(batch_size: int, protobuf_file: str, output_name: str) -> Tuple[bytes, str, popart.NllLoss]:
    """Create popart builder and loss for model.

    Args:
        batch_size : Batch size per inference.
        protobuf_file : ONNX binary protobuf filename.
        output_name: Name of the output Tensor using which loss must be computed.

    Returns:
        Modelproto, label and loss.

    """
    # Create builder from onnx protobuf file
    builder = popart.Builder(protobuf_file)

    # Set up label Tensor
    label_shape = popart.TensorInfo("INT32", [batch_size])
    label = builder.addInputTensor(label_shape)
    proto = builder.getModelProto()

    # Add loss
    loss = popart.NllLoss(output_name, label, "nllLossVal")
    return proto, label, loss


def preprocess_data(data: torch.Tensor, label: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data from data loader.

    Args:
        data: image input
        label: corresponding output

    Returns: pre-processed data and label in numpy format.

    """
    data, label = data.numpy(), label.numpy()
    data = data.reshape(opts.batches_per_step, opts.batch_size, -1)
    label = label.reshape(opts.batches_per_step, opts.batch_size)
    label = label.astype(np.int32)
    return data, label


def train(opts) -> None:
    """Train MNIST model using command line args."""

    # Limit batches_per_step so the test set isn't evaluated more than once.
    max_value = NUM_TEST_SAMPLES // opts.batch_size
    if max_value < opts.batches_per_step:
        print("(batches-per-step * batch-size) is larger than test set!\n"
              " Reduced batches-per-step to: {}\n".format(max_value))
        opts.batches_per_step = max_value

    # Construct MNIST data loaders
    train_loader = get_data_loader(
        opts.batch_size, opts.batches_per_step, is_train=True
    )

    test_loader = get_data_loader(
        opts.batch_size, opts.batches_per_step, is_train=False
    )

    print("Creating ONNX model.")
    proto_filename, data_in, output = create_model(opts.batch_size)

    print("Converting model.")
    proto, label_in, loss = convert_model(opts.batch_size, proto_filename, output)

    # Describe how to run the model
    anchor_desc = {
        output: popart.AnchorReturnType("ALL"),
        loss.output(0): popart.AnchorReturnType("ALL"),
    }
    dataFlow = popart.DataFlow(opts.batches_per_step, anchor_desc)
    optimizer = popart.SGD(0.01)

    # Options
    userOpts = popart.SessionOptions()

    # Ensure weight tensors in the validation model are not modified by the IR
    userOpts.constantWeights = False

    # Select a device
    deviceManager = popart.DeviceManager()
    if opts.simulation:
        options = {"compileIPUCode": True, "numIPUs": 1,
                   "tilesPerIPU": TILES_PER_IPU}
        device = deviceManager.createIpuModelDevice(options)
    else:
        device = deviceManager.acquireAvailableDevice()
        if device is None:
            print("Failed to acquire IPU. Exiting.")
            return

    def init_session(proto, loss, dataFlow, userOpts, device, training):
        # Create a session to compile and execute the graph
        if training:
            session = popart.TrainingSession(
                fnModel=proto,
                losses=[loss],
                optimizer=optimizer,
                dataFeed=dataFlow,
                userOptions=userOpts,
                deviceInfo=device
            )
        else:
            session = popart.InferenceSession(
                fnModel=proto,
                losses=[loss],
                dataFeed=dataFlow,
                userOptions=userOpts,
                deviceInfo=device
            )

        print("Compiling the {} graph.".format("training" if training else "validation"))
        session.prepareDevice()

        # Create buffers to receive results from the execution
        anchors = session.initAnchorArrays()

        Session = namedtuple('Session', ['session', 'anchors'])
        return Session(session, anchors)

    training = init_session(proto, loss, dataFlow, userOpts, device, training=True)
    validation = init_session(proto, loss, dataFlow, userOpts, device, training=False)

    print("Running training loop.")
    inputs_per_step = opts.batch_size * opts.batches_per_step
    for i in range(opts.epochs):
        # Training
        if i > 0:
            training.session.resetHostWeights('ckpt.onnx')
        training.session.weightsFromHost()
        training.session.optimizerFromHost()
        for data, label in train_loader:
            if len(label) != inputs_per_step:
                continue
            data, label = preprocess_data(data, label)
            stepio = popart.PyStepIO({data_in: data, label_in: label}, training.anchors)
            training.session.run(stepio)

        # Evaluation
        aggregated_loss = 0
        num_correct = 0

        training.session.modelToHost('ckpt.onnx')
        validation.session.resetHostWeights('ckpt.onnx')
        validation.session.weightsFromHost()

        for data, label in test_loader:
            if len(label) != inputs_per_step:
                continue
            data, label = preprocess_data(data, label)
            stepio = popart.PyStepIO({data_in: data, label_in: label}, validation.anchors)
            validation.session.run(stepio)
            aggregated_loss += np.mean(validation.anchors[loss.output(0)])
            results = np.argmax(
                validation.anchors[output].reshape([inputs_per_step, NUM_CLASSES]), 1
            )
            score = results == label.reshape([inputs_per_step])
            num_correct += np.sum(score)
        aggregated_loss /= len(test_loader)
        accuracy = num_correct / len(test_loader.dataset)

        # Log statistics
        print("Epoch #{}".format(i))
        print("   Loss={0:.4f}".format(aggregated_loss))
        print("   Accuracy={0:.2f}%".format(accuracy * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST training in PyTorch with popart backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Set the Batch size")
    parser.add_argument(
        "--batches-per-step",
        type=int,
        default=100,
        help="Number of minibatches to perform on the Device before returning to the Host."
        " This will be capped so the Device returns each epoch.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run the example with an IPU_MODEL device.",
    )
    parser.add_argument(
        "--log-graph-trace",
        action="store_true",
        help="Turn on ir logging to display the graph's ops.",
    )
    opts = parser.parse_args()

    # Set logging
    popart.getLogger('ir').setLevel('TRACE' if opts.log_graph_trace else 'CRITICAL')
    popart.getLogger('devicex').setLevel('CRITICAL')

    train(opts)
