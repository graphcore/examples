# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
from tqdm import tqdm
import os
import sys
import json
import torch
import torch.nn as nn
import torchvision
import poptorch
import popart
import torch.optim as optim
import octconv


def cifar10(data_dir, train=True):
    """
    Get the normalized CIFAR-10 dataset
    """
    (mean, std) = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    dataset = torchvision.datasets.CIFAR10(data_dir,
                                           train=train,
                                           download=True,
                                           transform=transforms)

    return dataset


def createConvBlock(in_channels, out_channels):
    """
    Creates a conv --> batchnorm --> relu --> maxpool block
    """
    conv = nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     padding=1,
                     bias=False)
    norm = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()
    pool = nn.MaxPool2d(2)

    return nn.Sequential(conv, norm, relu, pool)


def applyMultiConv(module):
    """
    Applies a poptorch.MultiConv to a module

    Any data-independent convolutions in the module will be executed in parallel
    on the IPU using the PopLibs multi-convolution implementation.
    """
    forward_impl = module.forward

    def forwardWithMultiConv(*args, **kwargs):
        with poptorch.MultiConv():
            return forward_impl(*args, **kwargs)

    module.forward = forwardWithMultiConv


class OctConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, use_multi=True):
        """
        Module containing an octave convolution --> batchnorm --> relu --> maxpool

        Uses the Octave convolution as described in the paper
        "Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution"
        https://arxiv.org/pdf/1904.05049.pdf

        This module ensures that both the high and low frequency feature outputs
        (when both are present) have the same operations applied to them.
        """
        super().__init__()
        self.octconv = octconv.OctConv2d(in_channels,
                                         out_channels,
                                         kernel_size=3,
                                         padding=1,
                                         alpha=alpha)

        if use_multi:
            applyMultiConv(self.octconv)

        norm_high = nn.BatchNorm2d(self.octconv.out_channels["high"])
        relu = nn.ReLU()
        pool = nn.MaxPool2d(2)
        self.high_seq = nn.Sequential(norm_high, relu, pool)
        self.has_low = self.octconv.alpha_out > 0.

        if self.has_low:
            norm_low = nn.BatchNorm2d(self.octconv.out_channels["low"])
            self.low_seq = nn.Sequential(norm_low, relu, pool)

    def forward(self, input):
        out = self.octconv(input)

        if self.has_low:
            # Propagate both high and low frequency features
            y_high, y_low = out
            return self.high_seq(y_high), self.low_seq(y_low)
        else:
            return self.high_seq(out)


class ClassificationModel(nn.Module):
    def __init__(self, conv_mode="vanilla", alpha=0.5, expansion=1):
        """
        CNN model for image classification tasks.

        conv_mode: Selects the convolution implementation used in the model:
            * "vanilla": Uses the standard torch.nn.Conv2d
            * "octave": Uses octconv.OctConv2d
            * "multi-octave: Uses poptorch.MultiConv to accelerate "octave"

        alpha: Ratio of low-frequency features used in the octave convolutions.

        expansion: Factor for parametrizing the width of the model.
        """
        super().__init__()

        assert isinstance(expansion, int) and expansion > 0, \
            f"Invalid expansion \"{expansion}\". Must be a positive integer."

        self.num_channels = 16 * expansion

        if conv_mode == "vanilla":
            self._makeVanilla()
        elif conv_mode == "octave":
            self._makeOctave(alpha, use_multi=False)
        elif conv_mode == "multi-octave":
            self._makeOctave(alpha, use_multi=True)
        else:
            raise AssertionError((f"Invalid conv_mode=\"{conv_mode}\"."
                                  "Must be vanilla, octave, or multi-octave"))

        self.fc = nn.Linear(self.num_channels * 4 * 4, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def _makeVanilla(self):
        self.convlayers = nn.Sequential(
            createConvBlock(3, self.num_channels),
            createConvBlock(self.num_channels, self.num_channels * 2),
            createConvBlock(self.num_channels * 2, self.num_channels))

    def _makeOctave(self, alpha, use_multi):
        self.convlayers = nn.Sequential(
            OctConvBlock(3, self.num_channels, (0., alpha), use_multi),
            OctConvBlock(self.num_channels, self.num_channels * 2, alpha,
                         use_multi),
            OctConvBlock(self.num_channels * 2, self.num_channels, (alpha, 0.),
                         use_multi))

    def forward(self, x, labels=None):
        out = self.convlayers(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        out = self.log_softmax(out)

        if labels is None:
            # Inference model, just return the prediction
            return out
        else:
            # Training model, calculate the loss and return it along with the prediction
            loss = self.loss(out, labels)
            return out, loss


def setupOptions(args, train=True):
    """
    Setup poptorch options for either training or inference runs.
    """
    opts = poptorch.Options().deviceIterations(args.batches_per_step)

    if args.cache_dir:
        # Separate caches for training/inference to prevent overwriting.
        prefix = args.conv_mode
        suffix = "-train" if train else "-inference"
        cache = args.cache_dir + f"/{prefix}{suffix}"
        opts.enableExecutableCaching(cache)

    if args.profile_dir:
        # Enable profiling if supported
        assert not args.cache_dir, "Profiling is not supported with executable caching"

        engine_opts = {
            "autoReport.all": "true",
            "autoReport.directory": args.profile_dir,
            "profiler.format": "v3"
        }

        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(engine_opts)

        # Use synthetic data when profiling
        data_mode = int(popart.SyntheticDataMode.RandomNormal)
        opts.Popart.set("syntheticDataMode", data_mode)

    return opts


def accuracy(predictions, labels):
    """
    Evaluate accuracy from model predictions against ground truth labels.
    """
    ind = torch.argmax(predictions, 1)
    # provide labels only for samples, where prediction is available (during the training, not every samples prediction is returned for efficiency reasons)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / \
        labels.size()[0] * 100.0
    return accuracy


def setupTraining(model, args):
    """
    Setup a training run using the CIFAR-10 training dataset.

    Uses the poptorch.DataLoader so that each training iteration executed on the
    IPU will incorporate:

        * (mini-)batch size
        * device iterations
        * replica factor
        * gradient accumulation factor

    Using poptorch.DataLoaderMode.Async allows loading the dataset on a separate
    thread.  This reduces the host/IPU communication overhead by using the time
    that the IPU is running to load the next batch on the CPU.
    """
    opts = setupOptions(args, train=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    training_model = poptorch.trainingModel(model, opts, optimizer)
    dataset = cifar10(args.data_dir, train=True)

    loader = poptorch.DataLoader(opts,
                                 dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=8,
                                 mode=poptorch.DataLoaderMode.Async)

    return training_model, loader


def train(model, loader, num_epochs):
    """
    Train the model on the IPU.
    """
    num_batches = len(loader)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        bar = tqdm(loader, total=num_batches)

        for data, labels in bar:
            predictions, losses = model(data, labels)

            with torch.no_grad():
                mean_loss = torch.mean(losses).item()
                acc = accuracy(predictions, labels)

            bar.set_description("Loss:{:0.4f} | Accuracy:{:0.2f}%".format(
                mean_loss, acc))


def setupInference(model, args):
    """
    Setup a training run using the CIFAR-10 training dataset.

    Uses the poptorch.DataLoader so that each training iteration executed on the
    IPU will incorporate:

        * (mini-)batch size
        * device iterations
        * replica factor
        * gradient accumulation factor

    Applying the poptorch.AsynchronousDataAccessor allows loading the dataset on
    a separate thread.  This reduces the host/IPU communication overhead by
    using the time that the IPU is running to load the next batch on the CPU.
    """
    opts = setupOptions(args, train=False)
    inference_model = poptorch.inferenceModel(model, opts)
    dataset = cifar10(args.data_dir, train=False)

    loader = poptorch.DataLoader(opts,
                                 dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=8)
    loader = poptorch.AsynchronousDataAccessor(loader)

    return inference_model, loader


def test(inference_model, loader):
    """
    Test the model on the IPU.
    """
    num_batches = len(loader)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(loader, total=num_batches):
            output = inference_model(data)
            sum_acc += accuracy(output, labels)

    print("Accuracy on test set: {:0.2f}%".format(sum_acc / num_batches))


def profile(model, args):
    """
    Profile a single training iteration on the IPU using synthetic data
    """
    opts = setupOptions(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    training_model = poptorch.trainingModel(model, opts, optimizer)

    # Generate a random dataset for profiling
    device_batch_size = args.batch_size * args.batches_per_step
    torch.manual_seed(0)
    data = torch.randn(device_batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (device_batch_size,))
    _, _ = training_model(data, labels)


def parseArgs():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Octave Convolution in PopTorch")
    parser.add_argument("--conv-mode",
                        choices=["vanilla", "octave", "multi-octave"],
                        default="vanilla",
                        help="Convolution implementation used in the classification model (default: vanilla)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Ratio of low-frequency features used in octave convolutions (default: 0.5)")
    parser.add_argument("--batch-size",
                        type=int,
                        default=8,
                        help="batch size for training (default: 8)")
    parser.add_argument("--batches-per-step",
                        type=int,
                        default=50,
                        help="device iteration (default:50)")
    parser.add_argument("--test-batch-size",
                        type=int,
                        default=80,
                        help="batch size for testing (default: 80)")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr",
                        type=float,
                        default=0.05,
                        help="learning rate (default: 0.05)")
    parser.add_argument(
        "--profile-dir",
        type=str,
        help="Perform a single iteration of training for profiling and place in specified folder."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Enable executable caching in the specified folder")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Location to use for loading the CIFAR-10 dataset from.")

    parser.add_argument("--expansion",
                        type=int,
                        default=1,
                        help="Expansion factor for tuning model width.")

    return parser.parse_args()


if __name__ == "__main__":
    # Create the model from command line args
    args = parseArgs()
    model = ClassificationModel(conv_mode=args.conv_mode,
                                alpha=args.alpha,
                                expansion=args.expansion)

    if args.profile_dir:
        profile(model, args)
        sys.exit(0)

    # Train the model
    training_model, train_loader = setupTraining(model, args)
    train(training_model, train_loader, args.epochs)

    # Update the weights in model by copying from the training IPU. This updates (model.parameters())
    training_model.copyWeightsToHost()

    # Evaluate the trained model
    inference_model, test_loader = setupInference(model, args)
    test(inference_model, test_loader)
