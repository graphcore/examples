# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytorch_lightning as pl
import torch
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

    # PopTorch includes an Strategy class which exposes additional IPU specific hardware and software options.
    # See the documentation on [session control options]
    # (https://docs.graphcore.ai/projects/popart-user-guide/en/latest/importing.html#session-control-options)
    # and [batching]
    # (https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#efficient-data-batching)
    # for further information on specific options.

    # Firstly we start by creating the options class.
    train_options = poptorch.Options()

    # One useful option is device iterations. This allows the IPU to eliminate some
    # host overhead by pulling more elements from the dataloader at a time while
    # still running with the normal model batchsize.
    train_options.deviceIterations(300)

    # Replication factor will replicate the program across multiple IPUs
    # automatically. This can also be done using the ipus=N option. However the
    # dataloader will not automatically pull in the additional elements leading
    # to it pulling in batch_size/replication factor elements at a time.
    train_options.replicationFactor(4)

    # To avoid this we provide a poptorch.Dataloader class. This is almost the
    # same as `torch.utils.data.DataLoader` which it wraps. The difference
    # is that it takes in a `poptorch.Options` class which is then used to calculate
    # the correct batchsize to pull in.
    # It also supports a `poptorch.DataLoaderMode.Async` which will load in the data
    # asynchronously to further reduce host overhead.
    train_data_loader = poptorch.DataLoader(
        options=train_options,
        dataset=data_set,
        batch_size=16,
        shuffle=True,
        mode=poptorch.DataLoaderMode.Async,
        num_workers=64,
    )

    # PyTorch Lightning provides an `IPUStrategy` class which takes in these options
    # and will pass them to PopTorch under the hood.

    trainer = pl.Trainer(
        accelerator="ipu",
        devices=1,
        max_epochs=1,
        log_every_n_steps=1,
        strategy=IPUStrategy(training_opts=train_options),
    )

    trainer.fit(model, train_data_loader)
