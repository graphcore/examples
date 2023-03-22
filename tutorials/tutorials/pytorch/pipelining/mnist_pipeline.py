#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
"""Demonstrate parallel execution methods for convolutional networks on MNIST."""
import os
import json
import argparse
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import poptorch

# set the random seed to obtain deterministic results
torch.manual_seed(0)


def get_mnist_data(opts):
    options = poptorch.Options()
    options.Training.gradientAccumulation(opts.gradient_accumulation)
    options.deviceIterations(opts.device_iterations)
    training_data = poptorch.DataLoader(
        options,
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        mode=poptorch.DataLoaderMode.Async,
        num_workers=16,
    )

    validation_data = poptorch.DataLoader(
        poptorch.Options().deviceIterations(opts.gradient_accumulation * opts.device_iterations),
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=opts.test_batch_size,
        shuffle=True,
        drop_last=True,
        mode=poptorch.DataLoaderMode.Async,
        num_workers=16,
    )

    return training_data, validation_data


class ConvLayer(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvLayer(1, 10, 5, 2)
        self.layer2 = ConvLayer(10, 20, 5, 2)
        self.layer3 = nn.Linear(320, 256)
        self.layer3_act = nn.ReLU()
        self.layer4 = nn.Linear(256, 10)

    def forward(self, x):
        with poptorch.Block("B1"):
            x = self.layer1(x)
        with poptorch.Block("B2"):
            x = self.layer2(x)
        with poptorch.Block("B3"):
            x = x.view(-1, 320)
            x = self.layer3_act(self.layer3(x))
        with poptorch.Block("B4"):
            x = self.layer4(x)
        return x


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, labels=None):
        output = self.model(args)
        if labels is None:
            return output
        with poptorch.Block("B4"):
            loss = self.loss(output, labels)
        return output, loss


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    # provide labels only for samples, where prediction is available
    # (during the training, not every samples prediction is returned
    # for efficiency reasons by default)
    labels = labels[-predictions.size()[0] :]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy


def train(training_model, training_data, opts):
    nr_batches = len(training_data)
    for epoch in range(1, opts.epochs + 1):
        print(f"Epoch {epoch}/{opts.epochs}")
        bar = tqdm(training_data, total=nr_batches)
        for data, labels in bar:
            preds, losses = training_model(data, labels)
            mean_loss = torch.mean(losses).item()
            acc = accuracy(preds, labels)
            bar.set_description(f"Loss:{mean_loss:0.4f} | Accuracy:{acc:0.2f}%")


def test(inference_model, test_data):
    nr_batches = len(test_data)
    sum_acc = 0.0
    for data, labels in tqdm(test_data, total=nr_batches):
        output = inference_model(data)
        sum_acc += accuracy(output, labels)
    print(f"Accuracy on test set: {sum_acc / len(test_data):0.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST training in PopTorch")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="batch size for training (default: 40)",
    )
    parser.add_argument(
        "--device-iterations",
        type=int,
        default=10,
        help="device iterations for training (default:10)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=10,
        help="gradient accumulation count (default:10)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="batch size for testing (default: 8)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs to train (default: 3)")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="turn on or off profiling (default: False)",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profile",
        help="where to save the profile (default: ./profile)",
    )
    parser.add_argument(
        "--strategy",
        choices=["sharded", "pipelined"],
        default="pipelined",
        help="execution strategy",
    )
    parser.add_argument("--offload-optimiser", action="store_true", help="offload optimiser state")
    parser.add_argument("--debug", action="store_true", help="print out the debug logging while running")
    opts = parser.parse_args()

    if opts.debug:
        poptorch.setLogLevel(1)  # Force debug logging

    training_data, test_data = get_mnist_data(opts)
    model = Network()
    model_with_loss = TrainingModelWithLoss(model)
    options = poptorch.Options()
    options._Popart.set("defaultPrefetchBufferingDepth", 3)
    model_opts = options.deviceIterations(opts.device_iterations)
    if opts.strategy == "pipelined":
        strategy = poptorch.PipelinedExecution(
            poptorch.Stage("B1").ipu(0),
            poptorch.Stage("B2").ipu(1),
            poptorch.Stage("B3").ipu(2),
            poptorch.Stage("B4").ipu(3),
        )
        model_opts.setExecutionStrategy(strategy)
        model_opts.Training.gradientAccumulation(opts.gradient_accumulation)
    else:
        strategy = poptorch.ShardedExecution(
            poptorch.Stage("B1").ipu(0),
            poptorch.Stage("B2").ipu(1),
            poptorch.Stage("B3").ipu(2),
            poptorch.Stage("B4").ipu(3),
        )
        model_opts.setExecutionStrategy(strategy)

    if opts.offload_optimiser:
        # Set the storage for activations, weights, accumulator explicitly.
        # Show how to use these options.
        model_opts.TensorLocations.setActivationLocation(poptorch.TensorLocationSettings().useOnChipStorage(True))
        model_opts.TensorLocations.setWeightLocation(poptorch.TensorLocationSettings().useOnChipStorage(True))
        model_opts.TensorLocations.setAccumulatorLocation(poptorch.TensorLocationSettings().useOnChipStorage(True))
        # Stores the optimiser state in remote buffers instead of IPUs' memory.
        model_opts.TensorLocations.setOptimizerLocation(poptorch.TensorLocationSettings().useOnChipStorage(False))

    if opts.profile:
        print("Profiling is enabled.")
        profile_opts = {
            "autoReport.all": "true",
            "autoReport.directory": opts.profile_dir,
        }
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(profile_opts)

    training_model = poptorch.trainingModel(
        model_with_loss,
        model_opts,
        optimizer=poptorch.optim.AdamW(model.parameters(), lr=opts.learning_rate),
    )

    # run training, on IPU
    print("Training...")
    train(training_model, training_data, opts)

    # Update the weights in model by copying from the training IPU implicitly.

    # Check validation loss on IPU once trained.
    # Because PopTorch will be compiled on first call the
    # weights in model.parameters() will be copied implicitly.
    # Reuse the parallel execution strategy and options of the training model.
    # Keep the same number of batches per step by changing deviceIterations.
    model_opts = model_opts.clone()
    model_opts.Training.gradientAccumulation(1)
    model_opts.deviceIterations(opts.device_iterations * opts.gradient_accumulation)
    inference_model = poptorch.inferenceModel(model, model_opts)
    training_model.detachFromDevice()
    print("Testing...")
    test(inference_model, test_data)
    inference_model.detachFromDevice()
