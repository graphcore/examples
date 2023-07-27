# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import libpvti as pvti
import torch
import poptorch
import torchvision
import torch.nn as nn
import psutil
from tqdm import tqdm, trange
import os
from metrics import accuracy

# Set torch random seed for reproducibility
torch.manual_seed(42)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = torchvision.datasets.FashionMNIST("~/.torch/datasets", transform=transform, download=True, train=True)

test_dataset = torchvision.datasets.FashionMNIST("~/.torch/datasets", transform=transform, download=True, train=False)

train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(1)
)

classes = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(972, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        # The model is responsible for the calculation
        # of the loss when using an IPU. We do it this way:
        if self.training:
            return x, self.loss(x, labels)
        return x


model = ClassificationModel()
model.train()

opts = poptorch.Options()
opts.anchorTensor("conv1_weight", "conv1.weight")
opts.anchorTensor("conv2_weight", "conv2.weight")
opts = opts.outputMode(poptorch.OutputMode.All)

train_dataloader = poptorch.DataLoader(opts, train_subset, batch_size=32, shuffle=True, num_workers=20)

validation_dataloader = poptorch.DataLoader(opts, val_subset, batch_size=32, shuffle=False, num_workers=20)

optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)

mnistPvtiChannel = pvti.createTraceChannel("MNIST Application")
loss_graph = pvti.Graph("Loss", "")
accuracy_graph = pvti.Graph("Accuracy", "%")
cpu_usage_graph = pvti.Graph("CPU Load", "%")
cpu_load = cpu_usage_graph.addSeries("CPU Usage")
training_loss_series = loss_graph.addSeries("Training Loss")
validation_loss_series = loss_graph.addSeries("Validation Loss")
training_accuracy_series = accuracy_graph.addSeries("Training Accuracy")
validation_accuracy_series = accuracy_graph.addSeries("Validation Accuracy")
mnistInstantEventsChannel = pvti.createTraceChannel("Instant Events")
cpu_load.add(psutil.cpu_percent())
conv1_heatmap = pvti.HeatmapDouble("conv1.weight", torch.linspace(-16, 16, 33).tolist(), "2^x")
conv2_heatmap = pvti.HeatmapDouble("conv2.weight", torch.linspace(-16, 16, 33).tolist(), "2^x")

epochs = 10
print("Running training loop.")
for epoch in trange(epochs, desc="epochs"):
    pvti.Tracepoint.event(mnistInstantEventsChannel, f"Epoch {epoch} begin")
    with pvti.Tracepoint(mnistPvtiChannel, f"Epoch:{epoch}"):
        training_loss_total = 0.0
        training_accuracy_total = 0.0
        validation_loss_total = 0.0
        validation_accuracy_total = 0.0
        for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
            output, loss = poptorch_model(data, labels)
            training_loss_total += loss
            training_accuracy_total += accuracy(output, labels)

            conv1_tensor = torch.abs(poptorch_model.getAnchoredTensor("conv1_weight")).flatten()
            conv1_heatmap.add(conv1_tensor[conv1_tensor != 0].tolist())
            conv2_tensor = torch.abs(poptorch_model.getAnchoredTensor("conv2_weight")).flatten()
            conv2_heatmap.add(conv1_tensor[conv1_tensor != 0].tolist())
            # Record the training loss
            training_loss_series.add(round(loss.item(), 2))

        model.eval()
        for data, labels in tqdm(validation_dataloader, desc="batches", leave=False):
            output, loss = poptorch_model(data, labels)
            validation_loss_total += loss
            validation_accuracy_total += accuracy(output, labels)
            # Record the validation loss
            validation_loss_series.add(round(loss.item(), 2))
        model.train()

    training_loss_total /= len(train_dataloader)
    validation_loss_total /= len(validation_dataloader)
    training_accuracy_total /= len(train_dataloader)
    validation_accuracy_total /= len(validation_dataloader)

    # Record the training and validation accuracy
    training_accuracy_series.add(round(training_accuracy_total, 2))
    validation_accuracy_series.add(round(validation_accuracy_total, 2))

    print(f"Epoch #{epoch + 1}")
    print(f"   Loss={training_loss_total:.4f}")
    print(f"   Loss_Accuracy={training_accuracy_total:.2%}")
    print(f"   Validation_Loss={validation_loss_total:.4f}")
    print(f"   Validation_Loss_Accuracy={validation_accuracy_total:.2%}")
    cpu_load.add(psutil.cpu_percent())

model.eval()

test_dataloader = poptorch.DataLoader(opts, test_dataset, batch_size=32, num_workers=10)
test_accuracy = 0.0
for data, labels in test_dataloader:
    output = poptorch_model(data, labels)[0]
    test_accuracy += accuracy(output, labels)

test_accuracy /= len(test_dataloader)

poptorch_model.detachFromDevice()
print(f"Eval accuracy: {test_accuracy:.2%}")
