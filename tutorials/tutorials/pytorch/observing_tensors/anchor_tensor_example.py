# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# !/usr/bin/env python3

import torch
import torchvision
import torch.nn as nn
import poptorch

from tqdm import tqdm
import matplotlib.pyplot as plt


class BasicLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 320)
        self.fc2 = nn.Linear(320, 100)
        self.fc3 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()
        self.relu = nn.ReLU()

    def forward(self, x, labels=None):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.log_softmax(self.fc3(x))
        if self.training:
            return x, self.loss(x, labels)
        return x


batch_size_train = 1000
tensors = [
    "Gradient___fc1.weight",
    "Gradient___fc3.weight",
    "Gradient___fc2.bias",
    "Gradient___fc1.bias",
    "Gradient___fc3.bias",
    "Gradient___fc2.weight",
]
pictureName = "GradientHistogram.png"

# Initialize the PopTorch options and anchor the tensors of interest
opts = poptorch.Options()
for t in tensors:
    opts.anchorTensor(t, t)

# Download the dataset then load it into the dataloaders
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_data = torchvision.datasets.MNIST("~/.torch/datasets", transform=transform, download=True, train=True)
train_loader = poptorch.DataLoader(opts, train_data, batch_size=batch_size_train, shuffle=True)

# Initialize the PopTorch training model
model = BasicLinearModel()
model.train()  # Switch the model to training mode
# Models are initialised in training mode by default, so the line above will
# have no effect. Its purpose is to show how the mode can be set explicitly.
optimizer = poptorch.optim.SGD(model.parameters(), lr=0.01)
poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

# Compile the model so we can use getTensorNames to print out all the tensor names
poptorch_model.compile(torch.zeros(batch_size_train, 1, 28, 28), torch.zeros(batch_size_train).long())
print("tensor names:", poptorch_model.getTensorNames())

# Train the model
total_loss = 0.0
predictions = []
for data, labels in tqdm(train_loader, desc="batches", leave=False):
    output, loss = poptorch_model(data, labels)
    total_loss += loss
    predictions += output
print(f"Loss: {total_loss.item():.4f}")

# Retrieve the tensors using getAnchoredTensor
gradient = []
for t in tensors:
    gradient += poptorch_model.getAnchoredTensor(t).flatten()
gradient = torch.stack(gradient).abs()
idx = gradient != 0
gradient[idx] = torch.log2(gradient[idx])
gradient[~idx] = torch.min(gradient) - 2
gradient_data = gradient.numpy()

# We can process the gradient data to create the histogram.  Don't hesitate to
# experiment: you can then compare your histogram with the
# [histogram provided in the `static` folder](static/GradientHistogram.png).
fig, axs = plt.subplots(tight_layout=True)
axs.hist(gradient_data, bins=50)
axs.set(title="Gradient Histogram", ylabel="Frequency")
plt.savefig(pictureName)
print(f"Saved histogram to ./{pictureName}")
