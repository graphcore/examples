# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD


def mpi_reduce(input, average=False):
    if type(input) in [int, float]:
        out = comm.allreduce(input, op=MPI.SUM)
        if average:
            out /= comm.Get_size()
    elif type(input) == np.ndarray:
        out = np.hstack(comm.allgather(input))
    else:
        raise NotImplementedError("The input provided is not supported.")
    return out
