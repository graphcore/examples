#!/usr/bin/env python3
# Copyright (c) 2019 Graphcore Ltd. All rights reserved.


"""
A simple program that uses PyTorch to create a linear model and then
trains it on the MNIST data set using the popart library.

"""

import argparse
import numpy as np
import popart
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import datasets, transforms
from typing import Tuple
from collections import namedtuple
from time import time
import tempfile

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
        return func.softmax(x, dim=1)


def create_model(
    batch_size: int, temp_file: tempfile.NamedTemporaryFile
) -> Tuple[str, str]:
    """Create Pytorch model and export as an ONNX protobuf.

    Args:
        batch_size : Batch size of the model.
        temp_file : To hold the model

    Returns:
        image_input name, output_name
    """
    net = Net()
    image_input = "input_1"
    output = "output_1"
    input_names = [image_input] + [
        "learned_%d" % i for i, _ in enumerate(net.parameters())
    ]
    dummy_input = torch.randn(batch_size, IMAGE_WIDTH * IMAGE_HEIGHT)
    torch.onnx.export(
        net,
        dummy_input,
        temp_file.name,
        input_names=input_names,
        output_names=[output],
    )
    return image_input, output


def convert_model(
    batch_size: int, protobuf_file: str, output_name: str
) -> Tuple[bytes, str, str]:
    """Create popart builder and loss for model.

    Args:
        batch_size : Batch size per inference.
        protobuf_file : ONNX binary protobuf filename.
        output_name: Name of the output Tensor using which loss must be computed

    Returns:
        Modelproto, label and loss.

    """
    # Create builder from onnx protobuf file
    builder = popart.Builder(protobuf_file)

    # Set up label Tensor
    label_shape = popart.TensorInfo("INT32", [batch_size])
    label = builder.addInputTensor(label_shape)

    # Add loss
    loss = builder.aiGraphcore.nllloss([output_name, label], popart.ReductionType.Sum, debugPrefix="nllLossVal")
    proto = builder.getModelProto()
    return proto, label, loss


def get_data_loader(
    cl_opts: argparse.Namespace, is_train: bool
) -> torch.utils.data.DataLoader:
    """Get dataloader for training/testing.

    Args:
        cl_opts: The command line arguments
        is_train: Flag is True if training.

    Returns:
        Dataloader for the split requested.
    """
    if cl_opts.syn_data_type in ["random_normal", "zeros"]:
        print(
            "Loading FAKE data {}".format(
                "for training" if is_train else "for inference"
            )
        )
        data_set = datasets.FakeData(
            size=cl_opts.batch_size * cl_opts.batches_per_step,
            image_size=(1, 28, 28),
            num_classes=NUM_CLASSES,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
            ),
        )
    else:
        print(
            "Loading MNIST data {}".format(
                "for training" if is_train else "for inference"
            )
        )
        data_set = datasets.MNIST(
            "../data",
            train=is_train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
            ),
        )
    return torch.utils.data.DataLoader(
        data_set,
        batch_size=cl_opts.batch_size * cl_opts.batches_per_step,
        shuffle=is_train,
    )


def preprocess_data(
    data: torch.Tensor, label: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
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


def train(opts, model_file, ckpt_file) -> None:
    """
    Train MNIST model using command line args.

    Args:
        opts: The command line options
        model_file: Temporary file for holding the model
        ckpt_file: Temporary file for holding the weights

    """
    if not opts.test_mode:
        max_value = NUM_TEST_SAMPLES // opts.batch_size
        if max_value < opts.batches_per_step:
            print(
                "(batches-per-step * batch-size) is larger than test set!\n"
                " Reduced batches-per-step to: {}\n".format(max_value)
            )
            opts.batches_per_step = max_value

    # Construct MNIST data loaders
    train_loader = get_data_loader(opts, is_train=True)

    test_loader = get_data_loader(opts, is_train=False)
    print("Creating ONNX model.")
    data_in, output = create_model(opts.batch_size, model_file)
    print("Converting model.")
    proto, label_in, loss = convert_model(
        opts.batch_size, model_file.name, output
    )

    # Describe how to run the model
    anchor_desc = {
        output: popart.AnchorReturnType("ALL"),
        loss: popart.AnchorReturnType("ALL"),
    }
    dataFlow = popart.DataFlow(opts.batches_per_step, anchor_desc)
    optimizer = popart.ConstSGD(0.01)

    # Options
    userOpts = popart.SessionOptions()

    # Ensure weight tensors in the validation model are not modified by the IR
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

    # Select a device
    deviceManager = popart.DeviceManager()
    if opts.simulation:
        print("Running using IPU MODEL")
        options = {
            "compileIPUCode": True,
            "numIPUs": 1,
            "tilesPerIPU": TILES_PER_IPU,
        }
        device = deviceManager.createIpuModelDevice(options)
    else:
        print("Running using Hardware")
        device = deviceManager.acquireAvailableDevice()
        if device is None:
            print("Failed to acquire IPU. Exiting.")
            return
        if opts.test_mode:
            print(" IPU IDs: {}".format(device.driverIds))

    def init_session(proto, loss, dataFlow, userOpts, device, training, opts):
        # Create a session to compile and execute the graph
        if opts.test_mode:
            userOpts.instrumentWithHardwareCycleCounter = True
        if training:
            session = popart.TrainingSession(
                fnModel=proto,
                loss=loss,
                optimizer=optimizer,
                dataFlow=dataFlow,
                userOptions=userOpts,
                deviceInfo=device,
            )
        else:
            session = popart.InferenceSession(
                fnModel=proto,
                dataFlow=dataFlow,
                userOptions=userOpts,
                deviceInfo=device,
            )

        print(
            "Compiling the {} graph.".format(
                "training" if training else "validation"
            )
        )
        session.prepareDevice()

        # Create buffers to receive results from the execution
        anchors = session.initAnchorArrays()

        Session = namedtuple("Session", ["session", "anchors"])
        return Session(session, anchors)

    training = init_session(proto, loss, dataFlow, userOpts, device, True, opts)
    validation = init_session(
        proto, loss, dataFlow, userOpts, device, False, opts
    )

    inputs_per_step = opts.batch_size * opts.batches_per_step
    for i in range(opts.epochs):
        # Training
        if i > 0:
            training.session.resetHostWeights(ckpt_file.name)
        training.session.weightsFromHost()
        for data, label in train_loader:
            if len(label) != inputs_per_step:
                continue
            data, label = preprocess_data(data, label)
            stepio = popart.PyStepIO(
                {data_in: data, label_in: label}, training.anchors
            )
            if opts.test_mode == "training":
                start = time()
            training.session.run(stepio)
            if opts.test_mode == "training":
                duration = time() - start
                report_string = "{:<8.3} sec/itr.".format(duration)
                report_string += "   " + iteration_report(opts, duration)
                print(report_string)
                print(
                    "Hardware cycle count per 'run':",
                    training.session.getCycleCount(),
                )
                print("Total time: {}".format(duration))
        # Evaluation
        aggregated_loss = 0
        num_correct = 0

        training.session.modelToHost(ckpt_file.name)
        validation.session.resetHostWeights(ckpt_file.name)
        validation.session.weightsFromHost()

        for data, label in test_loader:
            if len(label) != inputs_per_step:
                continue

            data, label = preprocess_data(data, label)
            stepio = popart.PyStepIO(
                {data_in: data, label_in: label}, validation.anchors
            )
            if opts.test_mode == "inference":
                start = time()
            validation.session.run(stepio)
            if opts.test_mode == "inference":
                duration = time() - start
                report_string = "{:<8.3} sec/itr.".format(duration)
                report_string += "   " + iteration_report(opts, duration)
                print(report_string)
                print(
                    "Hardware cycle count per 'run':",
                    validation.session.getCycleCount(),
                )
                print("Total time: {}".format(duration))
            aggregated_loss += np.mean(validation.anchors[loss])
            results = np.argmax(
                validation.anchors[output].reshape(
                    [inputs_per_step, NUM_CLASSES]
                ),
                1,
            )
            score = results == label.reshape([inputs_per_step])
            num_correct += np.sum(score)
        aggregated_loss /= len(test_loader)
        accuracy = num_correct / len(test_loader.dataset)

        # Log statistics
        print("Epoch #{}".format(i))
        print("   Loss={0:.4f}".format(aggregated_loss))
        print("   Accuracy={0:.2f}%".format(accuracy * 100))


def iteration_report(opts, time):
    return "{:5f} images/sec.".format(
        opts.batch_size * opts.batches_per_step / time
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST training in PyTorch with popart backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Set the Batch size"
    )
    parser.add_argument(
        "--batches-per-step",
        type=int,
        default=100,
        help="Number of minibatches to perform on the Device before returning t"
        "o the Host. This will be capped so the Device returns each epoch.",
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
    parser.add_argument(
        "--test-mode",
        type=str,
        help="Output extra performance information, specify wit"
        "h either 'training' or 'inference'",
    )

    parser.add_argument(
        "--syn-data-type",
        type=str,
        default="off",
        help="Specify to use synthetic data with either 'random"
        "_normal' or 'zeros'",
    )

    opts = parser.parse_args()

    # Validate synthetic data argument given
    if opts.syn_data_type:
        valids = ["random_normal", "zeros", "off"]
        if opts.syn_data_type not in valids:
            raise ValueError(
                "'--syn-data-type' must be one of {}".format(valids)
            )
    # Validate test mode given
    if opts.test_mode:
        valids = ["training", "inference"]
        if opts.test_mode not in valids:
            raise ValueError("'--test-mode' must be one of {}".format(valids))

    # Validate the given batch size and batches per step
    total = opts.batch_size * opts.batches_per_step
    if NUM_TEST_SAMPLES < total or total < 1:
        raise ValueError(
            "'--batch-size' ({}) multiplied by '--batches-per-step"
            "' ({}) comes to {} which is not in the range of avail"
            "able images ({})".format(
                opts.batch_size, opts.batches_per_step, total, NUM_TEST_SAMPLES
            )
        )
    # Set logging
    popart.getLogger("ir").setLevel(
        "TRACE" if opts.log_graph_trace else "CRITICAL"
    )
    popart.getLogger("devicex").setLevel("CRITICAL")

    with tempfile.NamedTemporaryFile() as model_file:
        with tempfile.NamedTemporaryFile() as ckpt_file:
            train(opts, model_file, ckpt_file)
