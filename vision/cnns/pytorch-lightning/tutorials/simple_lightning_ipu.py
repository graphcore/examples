# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl

import torch

import torchvision
import torchvision.transforms as transforms

from simple_torch_model import SimpleTorchModel


# This class shows a minimal lightning example. This example uses our own
# SimpleTorchModel which is a basic 2 conv, 2 FC torch network. It can be
# found in simple_torch_model.py.
class SimpleLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleTorchModel()

    def training_step(self, batch, _):
        x, label = batch
        prediction = self.model(x)
        loss = torch.nn.functional.nll_loss(prediction, label)
        return loss

    def validation_step(self, batch, _):
        x, label = batch
        prediction = self.model(x)
        preds = torch.argmax(prediction, dim=1)
        acc = torch.sum(preds == label).float() / len(label)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


class SimpleData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage="train"):
        # Retrieving the normal PyTorch datasets
        self.train_data = torchvision.datasets.FashionMNIST(
            "FashionMNIST", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )

        self.validation_data = torchvision.datasets.FashionMNIST(
            "FashionMNIST", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
        )

    # Normal PyTorch dataloader.
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=16, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=16, shuffle=False)


if __name__ == "__main__":
    # Create the model as usual.
    model = SimpleLightning()

    # Run on IPU using accelerator='ipu' and devices=1. This will run on IPU but
    # will not include any custom PopTorch Options. Changing IPUs from 1 to
    # devices=N will replicate the graph N times. This can lead to issues with
    # the DataLoader batching - the script ipu_strategy_and_dataloading.py shows
    # how these can be avoided through the use of IPUStrategy.
    datamodule = SimpleData()
    trainer = pl.Trainer(accelerator="ipu", devices=1, max_epochs=3, log_every_n_steps=1)

    # When fit is called the model will be compiled for IPU and will run on the available IPU devices.
    trainer.fit(model, datamodule)
