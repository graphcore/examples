# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import glob
import numpy as np
from os import path
import argparse
from termcolor import colored
from inspect_h5_checkpoint import inspect_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-set1", type=str, default="", help="Base directory with checkpoints.")
    parser.add_argument("--path-set2", type=str, default="", help="Base directory with checkpoints.")
    args = parser.parse_args()
    print(args)

    ckpt_filenames1 = glob.glob(path.join(args.path_set1, "*.h5"))
    ckpt_filenames2 = glob.glob(path.join(args.path_set2, "*.h5"))

    for ckpt1, ckpt2 in zip(ckpt_filenames1, ckpt_filenames2):

        print(f'evaluating {ckpt1.split("/")[-1]}')
        layers_from_all_ckpts = []

        weights_ckpt1 = inspect_checkpoint(file_name=ckpt1, all_tensors=True)
        weights_ckpt2 = inspect_checkpoint(file_name=ckpt2, all_tensors=True)

        for (name1, weights1), (name2, weights2) in zip(weights_ckpt1.items(), weights_ckpt2.items()):
            if name1 != name2:
                raise ValueError(
                    f"In comparison of {ckpt1} and {ckpt2} layer names do not match {name1} != {name2}. "
                    f"Compared models must have the same architecture and follow the same names."
                )

            if np.array_equal(weights1, weights2):
                print(colored(f"\t{name1}", "green"))
            else:
                total_difference = float(np.sum(np.abs(weights1 - weights2)))
                print(colored(f"\t{name1} {total_difference:.5f}", "red"))
