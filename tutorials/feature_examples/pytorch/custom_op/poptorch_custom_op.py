# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pathlib
import torch
import torch.nn as nn
import torchvision
import poptorch
import os
from pathlib import Path

# This example shows the process of loading in and using in a custom op in PopTorch. The
# code shows a simple CNN trained on the FashionMNIST dataset, replacing the activations
# in the model with a LeakyReLU custom activation, and then shows a short training process.

# Once you have implemented a custom op in C++ along with a Makefile, pathlib is used to verify
# the path to the generated .so file, and the op is loaded into the code.
myso = pathlib.Path("leaky_relu_example/build/custom_ops.so")
assert (
    myso.exists()
), "Failed to find Leaky ReLU Custom op file: `custom_ops.so`. Have you run the `make` command to generate this file?"

# Load the shared library file (.so) to allow the C++ functionalities
# to be called from this Python file when using poptorch.custom_ops.
torch.ops.load_library(myso)

# Loading and preprocessing of the dataset
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_data = torchvision.datasets.FashionMNIST("~/.torch/datasets", transform=transform, download=True, train=True)


# Load the custom op in as a function here for easy reusability. This function could also be defined
# in the model class. The return value of the custom op is the output tensor generated from the LeakyReLU
# computation. The main parameters of the custom op include:
#    * The input tensor to the op `[input]`
#    * The name of the custom op, defined in the identifier in the C++ code, in the format {'<domain>', '<name>', '<version_num>'}
#    * The domain of the custom op, this is defined in the identifier in the C++ code.
#    * The version number of the custom op, also defined in the identifier.
#    * An example output for the operation to know what format to return an output tensor as (in this case, it can just be a copy of the input)
#    * The attributes for the custom op: specific parameters that may define it's operation. For example the alpha value for a LeakyReLU activation.
def leakyrelu(inp, alpha=0.01):
    return poptorch.custom_op(
        [inp],
        "LeakyRelu",
        "custom.ops",
        1,
        example_outputs=[inp],
        attributes={"alpha": alpha},
    )[0]


# Define the model class as in any PyTorch model.
class CNNwithLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.linear_1 = nn.Linear(64 * 5 * 5, 256)
        self.linear_2 = nn.Linear(256, 10)

        self.pool = nn.MaxPool2d((2, 2))
        self.softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        # Then, you can use the custom op in the feedforward stage in a similar way to other functions
        l1 = leakyrelu(self.pool(self.conv_1(x)))
        l2 = leakyrelu(self.pool(self.conv_2(l1)))
        flt = torch.flatten(l2, start_dim=1)
        l3 = leakyrelu(self.linear_1(flt), alpha=0.005)
        pred = self.softmax(self.linear_2(l3))
        loss = self.loss(pred, y)

        return pred, loss


# Define the model
model = CNNwithLeakyReLU()
model.train()  # Switch the model to training mode
# Models are initialised in training mode by default, so the line above will
# have no effect. Its purpose is to show how the mode can be set explicitly.

# Use PopTorch's dataloader for efficient data batching for the train stage
opts = poptorch.Options()
batch_size = 16
train_dataloader = poptorch.DataLoader(opts, train_data, batch_size=batch_size, shuffle=True, num_workers=20)

# Define the training parameters
epochs = 5
optimizer = poptorch.optim.SGD(model.parameters(), lr=1e-3)

# Define the PopTorch model with poptorch.trainingModel as it is being used for training
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

# A simple training loop over 5 epochs to demonstrate
for epoch in range(epochs):
    total_loss = 0.0
    total_acc = 0.0
    for n, (x, y) in enumerate(train_dataloader):
        preds, loss = poptorch_model(x, y)
        total_loss += loss
        total_acc += 100 * torch.sum(torch.eq(torch.max(preds, 1)[1], y)) / batch_size
        # Count batches processed directly
        n_batches = n + 1

    mean_loss = (total_loss / n_batches).item()
    mean_acc = (total_acc / n_batches).item()

    print(f"Epoch {epoch} | Loss: {mean_loss:.2f} | Accuracy: {mean_acc:.2f}")
