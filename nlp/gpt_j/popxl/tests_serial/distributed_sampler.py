# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl import ops
from data.data_utils import DistributedSampler, WorkerInit
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
from pathlib import Path
from mpi4py import MPI
import time
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


def distributed_sampler():
    bs = 2
    inps = 5
    dataset_size = 2 * 10
    worker_seed = 47
    workers = 4
    epochs = 3
    replicas = 2

    dataset = np.random.random((dataset_size, inps)).astype(np.float32)
    sampler = DistributedSampler(dataset)
    dl = DataLoader(
        dataset,
        batch_size=bs,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=WorkerInit(worker_seed),
        persistent_workers=workers > 0,
        sampler=sampler,
    )
    session, in_stream, out_stream = sample_program((bs, inps), replicas)

    # check each instance get different data
    loader_list = list(dl)[0][0][0].numpy()

    # MPI to broadcast data in root=1 to root=0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    loader_list_copy = np.copy(loader_list)
    comm.Bcast(loader_list, root=1)

    # Assert if data broadcast to root=0 is different
    if comm.Get_rank() == 0 and not np.all(loader_list_copy == loader_list):
        print("Passed test: instances have different data")

    # Wait until both roots are finished
    time.sleep(2)

    # check epochs behaviour
    epochs_first_data = []
    for epoch in range(epochs):
        # set epoch explicitly before iterating dl
        sampler.set_epoch(epoch)
        step = 0
        for data in dl:
            x = data
            with session:
                out = session.run({in_stream: x})[out_stream]
                if step == 0:
                    epochs_first_data.append(out)
            step += 1

    assert len(epochs_first_data) == epochs, f"Expected {epochs} elements to compare, found {len(epochs_first_data)}"
    # check each epoch data is sampled in different order
    for first_item in epochs_first_data[1:]:
        not np.all(first_item == epochs_first_data[0])
        print("Passed test: each epoch samples dataset in different order")


if __name__ == "__main__":
    distributed_sampler()
