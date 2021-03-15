#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import poptorch
import torch.optim as optim

# The following is a workaround for pytorch issue #1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


def get_mnist_data(opts):
    training_data = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST('mnist_data/', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])),
                    batch_size=opts.batch_size * opts.batches_per_step, shuffle=True, drop_last=True)

    validation_data = torch.utils.data.DataLoader(
                      torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                                 transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])),
                      batch_size=opts.test_batch_size, shuffle=True, drop_last=True)
    return training_data, validation_data


class Block(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              num_filters,
                              kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 32, 3, 2)
        self.layer2 = Block(32, 64, 3, 2)
        self.layer3 = nn.Linear(1600, 128)
        self.layer3_act = nn.ReLU()
        self.layer3_dropout = torch.nn.Dropout(0.5)
        self.layer4 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Flatten layer
        x = x.view(-1, 1600)
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(self.layer3_dropout(x))
        x = self.softmax(x)
        return x


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, loss_inputs=None):
        output = self.model(args)
        if loss_inputs is None:
            return output
        else:
            loss = self.loss(output, loss_inputs)
            return output, loss


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    # provide labels only for samples, where prediction is available (during the training, not every samples prediction is returned for efficiency reasons)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy


def train(training_model, training_data, opts):
    nr_batches = len(training_data)
    for epoch in range(1, opts.epochs+1):
        print("Epoch {0}/{1}".format(epoch, opts.epochs))
        bar = tqdm(training_data, total=nr_batches)
        for data, labels in bar:
            preds, losses = training_model(data, labels)
            with torch.no_grad():
                mean_loss = torch.mean(losses).item()
                acc = accuracy(preds, labels)
            bar.set_description("Loss:{:0.4f} | Accuracy:{:0.2f}%".format(mean_loss, acc))


def test(inference_model, test_data):
    nr_batches = len(test_data)
    sum_acc = 0.0
    with torch.no_grad():
        for data, labels in tqdm(test_data, total=nr_batches):
            output = inference_model(data)
            sum_acc += accuracy(output, labels)
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST training in PopTorch')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training (default: 8)')
    parser.add_argument('--batches-per-step', type=int, default=50, help='device iteration (default:50)')
    parser.add_argument('--test-batch-size', type=int, default=80, help='batch size for testing (default: 80)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.05)')
    opts = parser.parse_args()

    training_data, test_data = get_mnist_data(opts)
    model = Network()
    model_with_loss = TrainingModelWithLoss(model)
    model_opts = poptorch.Options().deviceIterations(opts.batches_per_step)
    training_model = poptorch.trainingModel(model_with_loss, model_opts, optimizer=optim.SGD(model.parameters(), lr=opts.lr))

    inference_model = poptorch.inferenceModel(model)

    # run training, on IPU
    train(training_model, training_data, opts)

    # Update the weights in model by copying from the training IPU. This updates (model.parameters())
    training_model.copyWeightsToHost()

    # Check validation loss on IPU once trained. Because PopTorch will be compiled on first call the
    # weights in model.parameters() will be copied implicitly. Subsequent calls will need to call
    # inference_model.copyWeightsToDevice()
    test(inference_model, test_data)
