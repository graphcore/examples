# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl import ops
from data.data_utils import DistributedSampler, WorkerInit, StatefulDataLoader
from typing import Tuple
import numpy as np
from pathlib import Path
import os
from mpi4py import MPI
import torch
import popdist


def sample_program(input_shape: Tuple, replicas: int):
    ir = popxl.Ir("popdist")

    with ir.main_graph:
        x_h2d = popxl.h2d_stream(input_shape, dtype=popxl.float32, name="x_in")
        x_d2h = popxl.d2h_stream(input_shape, dtype=popxl.float32, name="x_out")
        x = ops.host_load(x_h2d)
        ops.host_store(x_d2h, x)

    ir.num_host_transfers = 1
    return popxl.Session(ir, "ipu_hw"), x_h2d, x_d2h


def dataloader_checkpoints_popdist():
    bs = 2
    inps = 5
    dataset_size = 2 * 10
    worker_seed = 47
    workers = 4
    epochs = 3
    replicas = 2
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if popdist.getInstanceIndex() == 0:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    else:
        seed = 0
    seed = comm.bcast(seed, root=0)

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
        for x in dl:
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
        for data in dl1:
            x = data
            with session:
                out = session.run({in_stream: x})[out_stream]
                epoch_samples.append(out)
                step += 1
            if epoch == 1 and step == 2:
                sampled_elements.append(epoch_samples)
                break
        else:
            sampled_elements.append(epoch_samples)
            continue
        break

    rank = popdist.getInstanceIndex()
    filename = os.path.join(test_dir, f"dataloader_state_{rank}.bin")
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
    for epoch in range(dl2.sampler.epoch, epochs):
        # set epoch before iterating dl
        step = 0
        epoch_samples = []
        for data in dl2:
            x = data

            with session:
                out = session.run({in_stream: x})[out_stream]
                step += 1
                epoch_samples.append(out)
        if epoch == start_epoch:
            sampled_elements[start_epoch].extend(epoch_samples)
        else:
            sampled_elements.append(epoch_samples)

    np.testing.assert_allclose(full_sampled_elements, sampled_elements, 10e-7)
    os.remove(filename)
    print("Passed test: distributed dataloader checkpoint")


if __name__ == "__main__":
    dataloader_checkpoints_popdist()
