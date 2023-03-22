# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl import ops
from data.data_utils import StatefulRandomSampler, WorkerInit, StatefulDataLoader
from typing import Tuple
import numpy as np
from pathlib import Path
import os
import torch
import popdist


def sample_program(input_shape: Tuple, replicas: int):
    ir = popxl.Ir(replication=replicas)

    with ir.main_graph:
        x_h2d = popxl.h2d_stream(input_shape, dtype=popxl.float32, name="x_in")
        x_d2h = popxl.d2h_stream(input_shape, dtype=popxl.float32, name="x_out")
        x = ops.host_load(x_h2d)
        ops.host_store(x_d2h, x)

    ir.num_host_transfers = 1
    return popxl.Session(ir, "ipu_hw"), x_h2d, x_d2h


def test_epochs():
    bs = 2
    inps = 5
    dataset_size = 2 * 10
    worker_seed = 47
    workers = 4
    epochs = 3
    replicas = 2
    seed = int(torch.empty((), dtype=torch.int64).random_().item())

    dataset = np.arange(0, dataset_size * inps).astype(np.float32).reshape(dataset_size, inps)

    dl = StatefulDataLoader(
        dataset,
        seed=seed,
        batch_size=bs,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=WorkerInit(worker_seed),
        persistent_workers=workers > 0,
    )

    epoch_first_element = []
    for epoch in range(epochs):
        step = 0
        for x in dl:
            if step == 0:
                epoch_first_element.append(x.numpy())
            step += 1

    for first_sample in epoch_first_element[1:]:
        not np.all(epoch_first_element[0] == first_sample)


def test_dataloader_checkpoints():
    bs = 2
    inps = 5
    dataset_size = 2 * 10
    worker_seed = 47
    workers = 4
    epochs = 3
    replicas = 2
    seed = int(torch.empty((), dtype=torch.int64).random_().item())

    dataset = np.arange(0, dataset_size * inps).astype(np.float32).reshape(dataset_size, inps)
    session, in_stream, out_stream = sample_program((bs, inps), replicas)
    test_dir = Path(__file__).parent.resolve()
    # dataloader - full iteration
    dl = StatefulDataLoader(
        dataset,
        seed=seed,
        batch_size=bs,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=WorkerInit(worker_seed),
        persistent_workers=workers > 0,
    )
    full_sampled_elements = []
    for epoch in range(epochs):
        step = 0
        epoch_samples = []
        print(f"epoch {epoch}\n")
        for x in dl:
            print(x.numpy())
            step += 1
            epoch_samples.append(x.numpy().astype(np.float32))
        full_sampled_elements.append(epoch_samples)

    # iterate dl, then interrupt iteration
    dl1 = StatefulDataLoader(
        dataset,
        seed=seed,
        batch_size=bs,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=WorkerInit(worker_seed),
        persistent_workers=workers > 0,
    )
    sampled_elements = []
    for epoch in range(epochs):
        # set epoch before iterating dl
        step = 0
        epoch_samples = []
        print(f"epoch {epoch}\n")
        for data in dl1:
            x = data
            x = np.repeat(x.reshape(1, *x.shape), replicas, axis=0)
            with session:
                out = session.run({in_stream: x})[out_stream]
                print(out[0])
                epoch_samples.append(out[0].copy())
                step += 1
            if epoch == 1 and step == 2:
                sampled_elements.append(epoch_samples)
                break
        else:
            sampled_elements.append(epoch_samples)
            continue
        break
    filename = os.path.join(test_dir, "dataloader_state.bin")
    dl1.save(filename)

    # resume from previous epoch and step
    dl2 = StatefulDataLoader(
        dataset,
        seed=seed,
        batch_size=bs,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=WorkerInit(worker_seed),
        persistent_workers=workers > 0,
    )
    dl2.resume(filename)
    start_epoch = dl2.sampler.epoch
    print("RESUMED")
    for epoch in range(dl2.sampler.epoch, epochs):
        # set epoch before iterating dl
        step = 0
        epoch_samples = []
        print(f"epoch {epoch}\n")
        for data in dl2:
            x = data
            x = np.repeat(x.reshape(1, *x.shape), replicas, axis=0)

            with session:
                out = session.run({in_stream: x})[out_stream]
                print(out[0])
                step += 1
                epoch_samples.append(out[0].copy())
        if epoch == start_epoch:
            sampled_elements[start_epoch].extend(epoch_samples)
        else:
            sampled_elements.append(epoch_samples)

    np.testing.assert_allclose(full_sampled_elements, sampled_elements, 10e-7)
    os.remove(os.path.join(test_dir, f"dataloader_state.bin"))


if __name__ == "__main__":
    test_epochs()
    test_dataloader_checkpoints()
