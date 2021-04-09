#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Half and mixed precision in PopTorch

# This tutorial shows how to use half and mixed precision in PopTorch with the
# example task of fine-tuning a pretrained ResNet-18 model on a single
# Graphcore Mk2 IPU.

# Requirements:
#   - an installed Poplar SDK. See the Getting Started guide for your IPU
#     hardware for details of how to install the SDK;
#   - Other Python modules: `pip install -r requirements.txt`

# Import the packages
import torch
import poptorch
import torchvision
from torchvision import transforms
from tqdm import tqdm


# Build the model
class CustomResNet18(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Create a fresh RestNet-18 model
        self.resnet18 = torchvision.models.resnet18()
        # Replace the last layer with a layer with num_classes output units
        self.resnet18.fc = torch.nn.Linear(512, num_classes)
        # Add a loss function
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        out = self.resnet18(x)
        if self.training:
            return out, self.loss(out, labels)
        return out
model = CustomResNet18(10)

# Casting a model's parameters
model = model.half()

# Prepare the data
transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.ConvertImageDtype(torch.half)])
train_dataset = torchvision.datasets.FashionMNIST("./datasets/",
                                                  transform=transform,
                                                  download=True,
                                                  train=True)
test_dataset = torchvision.datasets.FashionMNIST("./datasets/",
                                                 transform=transform,
                                                 download=True,
                                                 train=False)

# Optimizer and loss scaling
optimizer = poptorch.optim.SGD(model.parameters(),
                               lr=0.001,
                               loss_scaling=1000)

# Set PopTorch's options
opts = poptorch.Options()
# Stochastic rounding
opts.Precision.enableStochasticRounding(True)
# Partials data type
opts.Precision.setPartialsType(torch.float)

# Train the model
train_dataloader = poptorch.DataLoader(opts,
                                       train_dataset,
                                       batch_size=12,
                                       shuffle=True,
                                       num_workers=40)
poptorch_model = poptorch.trainingModel(model,
                                        options=opts,
                                        optimizer=optimizer)

epochs = 10
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss

# Evaluate the model
model.eval()
poptorch_model_inf = poptorch.inferenceModel(model, options=opts)
test_dataloader = poptorch.DataLoader(opts,
                                      test_dataset,
                                      batch_size=32,
                                      num_workers=40)

predictions, labels = [], []
for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.float().max(dim=1).indices
    labels += label

print(f"""Eval accuracy on IPU: {100 *
                (1 - torch.count_nonzero(torch.sub(torch.tensor(labels),
                torch.tensor(predictions))) / len(labels)):.2f}%""")
