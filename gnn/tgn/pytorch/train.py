# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


import argparse
import torch
import poptorch
from pathlib import Path
import time
import numpy as np
import warnings
from poptorch import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from tgn_modules import TGN, Data, DataWrapper, init_weights


def build_tgn(
        data: Path,
        dtype: str,
        batch_size: int,
        nodes_size: int,
        edges_size: int,
        dropout: float,
        target: str,
):
    model_dtype = torch.float32 if dtype == "float32" else torch.float16

    train_data = DataWrapper(
        Data(data, torch.float32, batch_size, nodes_size, edges_size), 'train'
    )
    test_data = DataWrapper(
        Data(data, torch.float32, batch_size, nodes_size, edges_size), 'val'
    )

    tgn = TGN(
        num_nodes=9227,
        raw_msg_dim=172,
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        dtype=model_dtype,
        dropout=dropout,
        target=target,
    )
    tgn.apply(init_weights)
    return train_data, test_data, tgn


def run_train(model, train_data, optim, target, do_reset) -> float:
    """Trains TGN for one epoch"""
    total_loss = 0
    num_events = 0
    model.train()

    if do_reset:
        model.memory.reset_state()  # Start with a fresh memory.
        if target == 'ipu':
            model.copyWeightsToDevice()
    for n, batch in enumerate(train_data):
        if target == 'ipu':
            count, loss = model(**batch)
        else:
            optim.zero_grad()
            count, loss = model(**batch)
            loss.backward()
            optim.step()
        total_loss += float(loss) * count
        num_events += count
        model.memory.detach()

    return total_loss / num_events


@torch.no_grad()
def run_test(model, inference_data) -> (float, float):
    """Inference over one epoch"""

    model.eval()
    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs
    aps = 0.0
    aucs = 0.0
    num_events = 0
    for batch in inference_data:
        count, y_true, y_pred = model(**batch)
        aps += count * average_precision_score(y_true, y_pred)
        aucs += count * roc_auc_score(y_true, y_pred)
        num_events += count
    return aps / num_events, aucs / num_events


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the TGN example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/JODIE",
        type=Path,
        help="directory to load/save the data"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=40,
        type=int,
        help="batch size for training and validation/test"
    )
    parser.add_argument(
        "--nodes-size",
        default=400,
        type=int,
        help="padding for nodes"
    )
    parser.add_argument(
        "--edges-size",
        default=1200,
        type=int,
        help="padding for edges"
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=("cpu", "ipu"),
        default="ipu",
        help="device to run on",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        choices=("float16", "float32"),
        default="float32",
        help="floating point format",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=25,
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--validate-every",
        default=1,
        type=int,
        help="run validation every n-th epoch",
    )
    parser.add_argument(
        "--lr",
        default=0.000075,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout rate in the attention module",
    )
    parser.add_argument(
        "--optimizer",
        choices=("SGD", "Adam"),
        default="Adam",
        help="Optimizer",
    )
    parser.add_argument(
        "-di",
        "--device-iterations",
        default=212,
        type=int,
        help="number of iterations to do on ipu before returning to host",
    )

    args = vars(parser.parse_args())

    epochs = args.pop("epochs")
    validate_every = args.pop("validate_every")
    optim = args.pop("optimizer")
    lr = args.pop("lr")
    device_iterations = args.pop("device_iterations")

    # Build dataloader and model
    train_data, test_data, model = build_tgn(**args)

    num_batches = len(train_data)
    if args["target"] == "cpu":
        device_iterations = int(1)
        warnings.warn(f"device_iterations was set to 1 for training on CPU")
    if not num_batches % device_iterations == 0:
        factors = [x for x in np.arange(1, num_batches + 1) if num_batches % x == 0]
        device_iterations = int(min(factors, key=lambda x: abs(x - device_iterations)))
        warnings.warn(f"device_iterations was set to {device_iterations}, the closest "
                      f"factor of the training batch count")

    train_opts = poptorch.Options()
    train_opts.deviceIterations(device_iterations)
    test_opts = poptorch.Options()
    test_opts.deviceIterations(1)

    torch.multiprocessing.set_sharing_strategy('file_system')

    async_options = {
            "sharing_strategy": poptorch.SharingStrategy.SharedMemory,
            "load_indefinitely": True,
            "early_preload": True,
            "buffer_size": 2}
    train_dl = DataLoader(options=train_opts, dataset=train_data, batch_size=1,
                          mode=poptorch.DataLoaderMode.Async,
                          async_options=async_options)
    test_dl = DataLoader(options=test_opts, dataset=test_data, batch_size=1)

    if args["target"] == "ipu":
        if optim.lower() == "sgd":
            optim = poptorch.optim.SGD(model.parameters(), lr=lr)
        elif optim.lower() == "adam":
            optim = poptorch.optim.AdamW(model.parameters(), lr=lr,
                                         bias_correction=True,
                                         weight_decay=0.0,
                                         eps=1e-8,
                                         betas=(0.9, 0.999))
        else:
            raise NotImplementedError(f"Optimizer {optim}")
        model_train = poptorch.trainingModel(model, options=train_opts, optimizer=optim)
        model_eval = poptorch.inferenceModel(model, options=test_opts)
    else:
        if optim.lower() == "sgd":
            optim = torch.optim.SGD(model.parameters(), lr=lr)
        elif optim.lower() == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optim}")
        model_train = model
        model_eval = model

    dataset_size = len(train_dl) * (device_iterations * args['batch_size'])

    # Run training
    for epoch in range(1, epochs+1):
        t0 = time.time()
        loss = run_train(model_train, train_dl, optim, args["target"],
                         do_reset=(epoch > 1))
        duration = time.time() - t0
        tput = dataset_size / duration
        print(f'Epoch {epoch}: Loss {loss:.4f}, Time {duration:.4f}, '
                f'Throughput: {tput:.4f} samples/s')

        if epoch % validate_every == 0 or epoch == epochs:
            aps, aucs = run_test(model_eval, test_dl)
            print(f'Validation APS {aps:.4f}, AUCS {aucs:.4f}')

    print(f'Training finished \n'
          f'APS {aps:.4f}, AUCS {aucs:.4f}')


if __name__ == "__main__":
    _main()
