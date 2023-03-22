# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
from torch import nn

# The simple PyTorch model used in each of these examples
class SimpleTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3)
        )

        conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3)
        )

        self.the_network = nn.Sequential(
            conv_block,
            conv_block2,
            torch.nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        return self.the_network(x)
