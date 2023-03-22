# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import poptorch
import pytorch_lightning as pl

import torch


# A simple regression model for function approximation.
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 100)
        self.fc2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        return self.fc2(x)

    def training_step(self, batch, _):
        x, target = batch
        prediction = self.forward(x)

        # Manually calculate the mean squared error.
        loss = (target - prediction).pow(2)

        # If the loss is not a normal torch loss PopTorch requires an identity
        # loss wrapper. This identifies it as the final loss. Other than non
        # torch losses we also require it in the case of multiple losses
        # combined together.
        return poptorch.identity_loss(loss, reduction="mean")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


# For simplicity we will train the network to approximate a simple function: sigmoid.
class Sigmoid(torch.utils.data.Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, _):
        element = torch.rand(1)
        return element, torch.sigmoid(element)


if __name__ == "__main__":
    # Create the model
    model = Model()

    # Get the training data
    train_set = Sigmoid()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

    trainer = pl.Trainer(accelerator="ipu", devices=1, max_epochs=3, log_every_n_steps=1)

    trainer.fit(model, train_loader)

    # Check how close we are.
    for _ in range(0, 100):
        x = torch.rand(1)
        actual = torch.sigmoid(x)
        pred = model(x)
        print("pred {}, actual {}, difference {}".format(pred.item(), actual.item(), (pred - actual).item()))
