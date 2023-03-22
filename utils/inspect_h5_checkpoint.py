# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import h5py
import numpy as np
import argparse
import os
from typing import Union


def inspect_checkpoint(
    file_name: str = "", tensor_names: Union[list, str] = "", all_tensors: bool = False, all_tensor_names: bool = False
):
    tensors = {}

    if not (len(tensor_names) > 0 or all_tensors or all_tensor_names):
        raise ValueError(
            "No tensors names given to inspect, "
            "and --all-tensors and --all-tensor-names are both False. What do you want to do?"
        )

    if not os.path.exists(file_name):
        raise ValueError(f"{file_name} does not exist.")

    with h5py.File(file_name, "r") as f:

        def process_file(name, content):
            weight_requested = name in tensor_names or all_tensors
            if all_tensor_names or weight_requested:
                if weight_requested and isinstance(content, h5py.Dataset):
                    tensor = np.zeros(shape=content.shape)
                    content.read_direct(tensor)
                    tensors[name] = tensor
                else:
                    tensors[name] = None

        f.visititems(process_file)
        return tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--file-name", type=str, default="", help="Checkpoint filename.")
    parser.add_argument(
        "--tensor-names",
        type=str,
        nargs="*",
        default="",
        help="Name of the tensors to inspect, e.g. --tensor-names tensor1 tensor2 --..",
    )
    parser.add_argument(
        "--all-tensors",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="If True, print the names and values of all the tensors.",
    )
    parser.add_argument(
        "--all-tensor-names",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="If True, print the names of all the tensors.",
    )
    args = parser.parse_args()
    print(args)

    tensors = inspect_checkpoint(**vars(args))
    if args.all_tensor_names:
        for tensor_name in tensors.keys():
            print(tensor_name)
    else:
        for name, value in tensors.items():
            print(f"{name}\n{value}")
