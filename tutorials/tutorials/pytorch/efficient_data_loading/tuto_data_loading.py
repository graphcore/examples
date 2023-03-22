# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import time

import poptorch
import torch
import torch.nn as nn


device_iterations = 50
batch_size = 16
num_workers = 32


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(41772, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        if self.training:
            return x, self.loss(x, labels)
        return x


def print_parameters(args):
    if args.synthetic_data:
        print("SYNTHETIC DATA. The IPU Throughput will not include the cost of IO")

    print(
        f"mini-batch size: {batch_size}\n",
        f"replication factor: {args.replicas}\n",
        f"device-iterations: {device_iterations}\n",
        f"workers: {num_workers}\n",
        f"--> Global batch size: {args.replicas*batch_size}",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-data",
        dest="synthetic_data",
        action="store_true",
        help="Use IPU-generated synthetic data",
    )
    parser.add_argument("--replicas", type=int, default=1, help="IPU replication factor")
    args = parser.parse_args()
    replicas = args.replicas

    opts = poptorch.Options()
    opts.deviceIterations(device_iterations)
    opts.replicationFactor(replicas)
    if args.synthetic_data:
        opts.enableSyntheticData(True)

    model = ClassificationModel()
    model.train()  # Switch the model to training mode
    # Models are initialised in training mode by default, so the line above will
    # have no effect. Its purpose is to show how the mode can be set explicitly.

    print_parameters(args)

    # Setup a PopTorch training model
    training_model = poptorch.trainingModel(model, opts, poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9))

    # Create a dataset from random data
    features = torch.randn([10000, 1, 128, 128])
    labels = torch.empty([10000], dtype=torch.long).random_(10)
    dataset = torch.utils.data.TensorDataset(features, labels)
    print("Dataset size: ", len(dataset))

    # PopTorch Dataloader
    training_data = poptorch.DataLoader(
        opts,
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        mode=poptorch.DataLoaderMode.Async,
        async_options={"early_preload": True},
    )

    # Number of steps necessary to consume the whole dataset
    steps = len(training_data)

    # Assess asynchronous dataloader throughput on CPU
    print("Evaluating Dataloader: ", steps, "steps")
    t0 = time.time()
    for data, labels in training_data:
        pass
    t1 = time.time()
    total_time = t1 - t0
    print("Total execution Time:", total_time, "s")
    print(
        "DataLoader throughput:",
        (steps * device_iterations * batch_size * replicas) / total_time,
        "items/s",
    )

    # IPU evaluation:
    # Warmup
    print("Compiling + Warmup ...")
    training_model.compile(data, labels)

    print(
        "Evaluating: ",
        steps,
        "steps of ",
        device_iterations * batch_size * replicas,
        " items",
    )

    if args.synthetic_data:
        # With synthetic data enabled, no data is copied from the host to the IPU, so we don't use
        # the dataloader, to prevent influencing the execution time and therefore the IPU throughput calculation
        t0 = time.time()
        for _ in range(steps):
            training_model(data, labels)
        t1 = time.time()
    else:
        t0 = time.time()
        for data, labels in training_data:
            training_model(data, labels)
        t1 = time.time()
    total_time = t1 - t0
    print("Total execution Time:", total_time, "s")
    print(
        "IPU throughput:",
        (steps * device_iterations * batch_size * replicas) / total_time,
        "items/s",
    )
