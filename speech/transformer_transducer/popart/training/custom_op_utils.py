# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import ctypes


def load_custom_lib(lib):
    " loads the custom library of given name from the build directory "
    rnnt_wd = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(rnnt_wd)

    so_path = "build/" + lib + ".so"
    if not os.path.isfile(so_path):
        print("Build the custom ops library with `make` before running this script")
        exit(1)
    libc = ctypes.cdll.LoadLibrary(so_path)
    return libc


def load_custom_rnnt_op():
    return load_custom_lib("rnnt_loss")


def load_custom_sparse_logsoftmax_op():
    return load_custom_lib("sparse_logsoftmax")


def load_exp_avg_custom_op():
    return load_custom_lib("exp_avg_custom_op")
