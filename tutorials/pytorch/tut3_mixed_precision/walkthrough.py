#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import poptorch
import torchvision
from torchvision import transforms
from tqdm import tqdm


if __name__ == '__main__':
    os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all": "true", "autoReport.directory": "mixed", "debug.allowOutOfMemory": "true"}'

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.ConvertImageDtype(torch.float16)])

    train_dataset = torchvision.datasets.FashionMNIST("./datasets/", transform=transform, download=True, train=True)
    test_dataset = torchvision.datasets.FashionMNIST("./datasets/", transform=transform, download=True, train=False)
    classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
    num_classes = len(classes)


    class CustomResNet18(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            # Download/upload a pretrained RestNet-18 model
            self.resnet18 = torchvision.models.resnet18(pretrained=True)
            # Freeze all the parameters in the model
            for param in self.resnet18.parameters():
                param.requires_grad = False
            # Replace with the last layer `fc` with a trainable `Linear` layer
            self.resnet18.fc = torch.nn.Linear(512, num_classes)
            # Add a loss function
            self.loss = torch.nn.CrossEntropyLoss()

        def forward(self, x, labels=None):
            out = self.resnet18(x)
            if self.training:
                return out, self.loss(out, labels)
            return out
    model = CustomResNet18(num_classes)

    # model = model.half()

    opts = poptorch.Options().deviceIterations(100)
    opts.Popart.set("enableStochasticRounding", True)

    train_dataloader = poptorch.DataLoader(opts,
                                           train_dataset,
                                           batch_size=12,
                                           shuffle=True,
                                           num_workers=40,
                                           mode=poptorch.DataLoaderMode.Async)

    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

    epochs = 5
    for epoch in tqdm(range(epochs), desc="epochs"):
        total_loss = 0.0
        for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
            output, loss = poptorch_model(data, labels)
            total_loss += loss
        print (f"Epoch {epoch+1}/{epochs} - Loss: {total_loss}")
