# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


import argparse
import torch
import poptorch
from pathlib import Path
import time
from poptorch import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from tgn_modules import TGN, Data, DataWrapper


def build_tgn(
        data: Path,
        dtype: str,
        batchsize,
        dropout: float,
):
    model_dtype = torch.float32 if dtype == "float32" else torch.float16

    opts = poptorch.Options()
    train_data = DataLoader(
        options=opts,
        dataset=DataWrapper(Data(data, torch.float32, batchsize), 'train'),
        batch_size=1,
    )
    test_data = DataLoader(
        options=opts,
        dataset=DataWrapper(Data(data, torch.float32, batchsize), 'test'),
        batch_size=1,
    )
    tgn = TGN(
        num_nodes=9227,
        raw_msg_dim=172,
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        dtype=model_dtype,
        dropout=dropout,
    )
    return train_data, test_data, tgn


def train(train_data, model, optim, target) -> float:
    """Trains TGN for one epoch"""

    total_loss = 0
    num_events = 0
    model.train()

    for n, batch in enumerate(train_data):
        batch = {nm: x.squeeze(0) for nm, x in batch.items()}
        if n == 0:
            model.memory.reset_state()  # Start with a fresh memory.
            if target == 'ipu':
                model.compile(**batch)
                model.copyWeightsToDevice()

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
def test(inference_data, model) -> (float, float):
    """Inference over one epoch"""

    model.eval()
    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs
    aps = 0.0
    aucs = 0.0
    num_events = 0
    for batch in inference_data:
        batch = {nm: x.squeeze(0) for nm, x in batch.items()}
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
        default="/localdata/research/datasets/tgn/JODIE",
        type=Path,
        help="directory to load/save the data"
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        default=200,
        type=int,
        help="batch size for training and validation/test"
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
        default=50,
        type=int,
        help="number of epochs to train for",
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
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

    args = vars(parser.parse_args())

    epochs = args.pop("epochs")
    target = args.pop("target")
    optim = args.pop("optimizer")
    lr = args.pop("lr")

    # Build dataloader and model
    train_data, test_data, model = build_tgn(**args)

    if target == "ipu":
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
        model_train = poptorch.trainingModel(model, optimizer=optim)
    else:
        if optim.lower() == "sgd":
            optim = torch.optim.SGD(model.parameters(), lr=lr)
        elif optim.lower() == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optim}")
        model_train = model

    # Run training
    for epoch in range(epochs):
        t0 = time.time()
        loss = train(train_data, model_train, optim, target)
        print(f'Epoch {epoch}: Loss {loss:.4f}, Time {time.time()-t0:.4f}')

    # import pickle, copy
    # trained_params = copy.deepcopy(dict(list(model.named_parameters())))
    # pickle.dump(trained_params, open("trained_params.pkl", "wb"))
    # pickle.dump(model.memory._memory, open("trained_memorybuffer.pkl", "wb"))

    aps, aucs = test(test_data, model)
    print(f'Training finished \n'
          f'APS {aps:.4f}, AUCS {aucs:.4f}')


if __name__ == "__main__":
    _main()
