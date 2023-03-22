# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import fcntl
import logging
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torchvision
import poptorch
import popdist
import popdist.poptorch


class ModelWithLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.loss = torch.nn.NLLLoss(reduction="mean")

    def forward(self, x, y=None):
        logits = self.model(x)
        if y is None:
            return logits
        log_preds = F.log_softmax(logits, dim=1)
        loss = self.loss(log_preds, y)
        return logits, poptorch.identity_loss(loss, reduction="none")


def train(args):
    if popdist.isPopdistEnvSet():
        instance = popdist.getInstanceIndex()
    else:
        instance = 0

    opts = create_options(args)
    data_train = load_data(args, opts, train=True)
    model = ModelWithLoss()

    # Training the model.
    model.train()
    optimizer = poptorch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        use_combined_accum=True,
    )
    poptorch_model = poptorch.trainingModel(
        model,
        opts,
        optimizer=optimizer,
    )

    log_in_single_instance("Training the model...")
    for epoch in range(args.epochs):
        log_in_single_instance(f"Epoch {epoch}")
        for x, y in data_train:
            _, loss = poptorch_model(x, y)
            # Loss contains a value for each replica.
            loss = torch.mean(loss)
            log_in_single_instance(f"Loss: {loss}")
    poptorch_model.destroy()

    # Persist checkpoints from all instances even though they are identical,
    # this is useful for testing purposes.
    torch.save(model.state_dict(), f"checkpoint-instance-{instance}.pt")

    # Validation
    data_validation = load_data(args, opts, train=False)
    model.eval()
    poptorch_model = poptorch.inferenceModel(model, opts)

    logging.info("Validating the model...")
    num_correct_predictions = 0
    for x, y in data_validation:
        y_pred = poptorch_model(x)
        _, y_pred = torch.max(y_pred, 1)
        num_correct_predictions += torch.eq(y_pred, y).long().sum().item()
    accuracy = num_correct_predictions / len(data_validation.dataset)
    logging.info(f"Validation accuracy: {accuracy}")


def create_options(args):
    if popdist.isPopdistEnvSet():
        opts = popdist.poptorch.Options()
        # When using the dataloader with 'auto_distributed_partitioning=True'
        # and 'shuffle=True' we must set the random seed to ensure that tensors
        # are in the same order in all processes.
        opts.randomSeed(args.seed)
        # Replication factor is already set via PopRun so
        # we ignore 'args.num_replicas'.
        logging.info(f"Num of local replicas: {popdist.getNumLocalReplicas()}")
    else:
        opts = poptorch.Options()
        opts.replicationFactor(args.num_replicas)
    return opts


def load_data(args, opts, train):
    # We need to lock the directory to avoid race conditions related to
    # downloading and writing a dataset.
    datasets_dir = os.path.expanduser("~/.torch/datasets")
    with create_and_lock_directory(datasets_dir):
        dataset = torchvision.datasets.CIFAR10(
            root=datasets_dir,
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5),
                    ),
                ]
            ),
        )

    # When using a dataloader with 'auto_distributed_partitioning=True',
    # PopTorch partitions the dataset for distributed execution (with PopRun)
    # automatically.
    return poptorch.DataLoader(
        opts,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        shuffle=train,
        auto_distributed_partitioning=True,
    )


def log_in_single_instance(string):
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        logging.info(string)


@contextmanager
def create_and_lock_directory(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass
    dir_fd = os.open(dir, os.O_RDONLY)
    fcntl.flock(dir_fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(dir_fd, fcntl.LOCK_UN)
        os.close(dir_fd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-replicas", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    if popdist.isPopdistEnvSet():
        popdist.init()
    train(parser.parse_args())
