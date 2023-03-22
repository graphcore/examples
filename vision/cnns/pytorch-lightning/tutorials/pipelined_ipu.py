# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl

import poptorch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.strategies import IPUStrategy
from simple_lightning_ipu import SimpleLightning


if __name__ == "__main__":
    # Create the model as usual.
    model = SimpleLightning()

    # Normal PyTorch dataset.
    data_set = torchvision.datasets.FashionMNIST(
        "FashionMNIST", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    train_options = poptorch.Options()
    train_options.deviceIterations(50)

    # When training with pipelined execution you should turn on gradient accumulation
    # otherwise you will be training out of sync. With it on the weight updates will be
    # scheduled at the end of the pipeline, essentially each stage calculating the gradient
    # with frozen weights and only applying them when there is no race condition.
    train_options.Training.gradientAccumulation(8)

    train_data = poptorch.DataLoader(
        train_options, data_set, batch_size=64, shuffle=True, mode=poptorch.DataLoaderMode.Async
    )

    # Model is a PyTorch Lightning model which looks like this:
    #
    # class SimpleLightning(pl.LightningModule):
    #    def __init__(self):
    #        self.model = SimpleTorchModel()
    #
    # And SimpleTorchModel looks like:
    #
    # class SimpleTorchModel(torch.nn.Module):
    #   def __init__(self):
    #      ...
    #        self.the_network = nn.Sequential(...)
    #
    # So to access a layer in this model we use:
    #   model.model.the_network[i]
    #

    # We mark `model.model.the_network[1]` and all layers after that as going on
    # ipu=1. By default layers will be allocated onto IPU 0. This means the layers
    # before the annotation will be on IPU 0 implicitly.
    poptorch.BeginBlock(model.model.the_network[1], ipu_id=1)

    # Adding this overrides the setting for `model.model.the_network[3]`
    # and all layers after it.
    poptorch.BeginBlock(model.model.the_network[3], ipu_id=2)

    # And so on.
    poptorch.BeginBlock(model.model.the_network[4], ipu_id=3)

    trainer = pl.Trainer(
        accelerator="ipu",
        devices=1,
        max_epochs=1,
        log_every_n_steps=1,
        strategy=IPUStrategy(training_opts=train_options, autoreport=True)
        # Autoreport will produce Poplar graph and execution
        # report. Here we use it so you can see in the report
        # the multiple IPUs in use.
    )

    trainer.fit(model, train_data)
