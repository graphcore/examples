#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import time
import torch
import poptorch
import tqdm
import argparse
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """

    def __init__(self, shape, size=60000, half_precision=False):
        self.size = size
        self.half_precision = half_precision
        self.data_shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        synthetic_data = torch.rand(self.data_shape)
        if self.half_precision:
            synthetic_data = synthetic_data.half()
        return synthetic_data, 0


class ResNextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnext = models.resnext50_32x4d(pretrained=True, progress=True)

    def forward(self, x):
        x = self.resnext(x)
        return x


def run_model(batch_size=20, availableMemoryProportion=0.8):

    model = ResNextModel()
    model.eval()  # Switch the model to inference mode

    opts = poptorch.Options().deviceIterations(1)
    opts.setAvailableMemoryProportion({"IPU0": availableMemoryProportion})

    test_dataloader = poptorch.DataLoader(
        opts,
        GeneratedDataset(shape=[3, 224, 224]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
    )

    poptorch_model = poptorch.inferenceModel(model, options=opts)

    tput_acc = 0

    for data, _ in test_dataloader:
        start = time.time()
        result = poptorch_model(data)
        end = time.time()
        tput_acc += data.size()[0] / (end - start)

    return tput_acc / len(test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    parser.add_argument(
        "--available-memory-proportion",
        type=float,
        default=0.8,
        help="Available memory proportion",
    )
    opts = parser.parse_args()

    average_tput = run_model(
        batch_size=opts.batch_size,
        availableMemoryProportion=opts.available_memory_proportion,
    )

    print(f"bs:{opts.batch_size},amp:{opts.available_memory_proportion},mean_throughput:{average_tput}")
