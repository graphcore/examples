# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from glob import glob

import numpy as np
import tensorflow.compat.v1 as tf

from typing import Dict, List, NamedTuple, Optional


class LoadedCheckpoint(NamedTuple):
    filename: str
    checkpoint: "TFCheckpoint"


def get_filenames_and_checkpoints(restore_path: str) -> List[LoadedCheckpoint]:

    ckpt_pattern_idx = re.compile(".*ckpt-([0-9]+).index$")
    ckpt_pattern = re.compile(".*ckpt-([0-9]+)$")

    # search to a maximum depth of 1
    ckpts = glob(os.path.join(restore_path, '*.index')) \
        + glob(os.path.join(restore_path, 'ckpt', '*.index'))

    training_ckpts = sorted([c for c in ckpts if ckpt_pattern_idx.match(c)],
                            key=lambda x: int(ckpt_pattern_idx.match(x).groups()[0]))

    weight_avg_ckpts = [c for c in ckpts if not ckpt_pattern_idx.match(c)]
    filenames = training_ckpts + weight_avg_ckpts
    filenames = [f[:-len(".index")] for f in filenames]

    return [(fn, tf.train.load_checkpoint(fn)) for fn in filenames]


def find_nans(fns_and_ckpts: List[LoadedCheckpoint]) -> Dict["VariableName", "NanIndices"]:

    print("Finding NaN values in checkpoints...")

    c0 = fns_and_ckpts[0][1]

    vs = set(c0.get_variable_to_dtype_map().keys())
    vs_no_nan_yet = vs
    nan_seen = {}

    for fn, c in fns_and_ckpts:

        to_discard = []

        for v in vs:

            t = c.get_tensor(v)

            if np.isnan(t).any():

                nan_locs = np.argwhere(np.isnan(t))

                if v in vs_no_nan_yet or nan_seen[v] != nan_locs:

                    print(fn.split('/')[-1], v)
                    print(nan_locs)
                    to_discard.append(v)
                    nan_seen[v] = nan_locs

        for v in to_discard:
            vs_no_nan_yet.discard(v)

    return nan_seen


def count_nans(file_path: str) -> int:
    print(f"Counting NaN values in checkpoint {file_path}...")
    checkpoint = tf.train.load_checkpoint(file_path)
    variables = set(checkpoint.get_variable_to_dtype_map().keys())
    nan_count = 0

    for variable in variables:
        tensor = checkpoint.get_tensor(variable)
        if np.isnan(tensor).any():
            nan_count += np.count_nonzero(np.isnan(tensor))

    return nan_count


def first_fail(
        fns_and_ckpts: List[LoadedCheckpoint],
        ffvar: str,
        ffidx: int) -> Dict["VarName", "ReferenceValue"]:
    """
    For a particular variable and index,
    get the value of that variable at that index
    for every checkpoint. Written with 1D vars in mind,
    will return/print slices for vars with 2+ dimensions
    """

    print("Analysing first failure...")

    progression = []

    for fn, c in fns_and_ckpts:

        ff = c.get_tensor(ffvar)[ffidx]
        progression.append((fn.split("/")[-1], ff))
        print(fn.split('/')[-1], ff)

    return progression


class NanStats(NamedTuple):
    count: int
    where: List[int]
    variable: str

    @classmethod
    def from_checkpoint(
        cls, ckpt, filter_variable: Optional[str] = None
    ) -> List["NanStats"]:
        variables = ckpt.get_variable_to_dtype_map().keys()
        stats = []
        for v in variables:
            if filter_variable is not None and filter_variable not in v:
                continue
            t = ckpt.get_tensor(v)
            t_nan = np.isnan(t)
            stats.append(
                cls(
                    count=np.count_nonzero(t_nan),
                    where=np.argwhere(t_nan),
                    variable=v,
                )
            )
        return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--restore-path', type=str, required=True)
    parser.add_argument('--find-nans', action='store_true')
    parser.add_argument('--first-fail', action='store_true')

    parser.add_argument('--first-fail-var', type=str)
    parser.add_argument('--first-fail-index', type=int)

    args = parser.parse_args()

    fns_and_ckpts = get_filenames_and_checkpoints(args.restore_path)

    if args.find_nans:
        find_nans(fns_and_ckpts)

    if args.first_fail:
        first_fail(fns_and_ckpts, args.first_fail_var, args.first_fail_index)
