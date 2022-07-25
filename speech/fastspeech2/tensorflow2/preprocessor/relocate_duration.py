# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np


def relocate_duration(root_path, duration_path):
    """Relocate duration datasets according to train/valid utt ids."""
    train_ids = np.load(os.path.join(root_path, "train_utt_ids.npy"))
    valid_ids = np.load(os.path.join(root_path, "valid_utt_ids.npy"))

    train_duration_path = os.path.join(root_path, "train", "duration")
    valid_duration_path = os.path.join(root_path, "valid", "duration")
    if not os.path.exists(train_duration_path):
        os.makedirs(train_duration_path)
    if not os.path.exists(valid_duration_path):
        os.makedirs(valid_duration_path)

    num_files = len(os.listdir(duration_path))
    for dur in tqdm(os.listdir(duration_path), total=num_files, desc="[Moving duration data]"):
        if not dur.endswith(".npy"):
            continue
        prefix = "-".join(dur.split("-")[:2])
        full_dur_path = os.path.join(duration_path, dur)
        if prefix in train_ids:
            des_path = os.path.join(train_duration_path, dur)
        elif prefix in valid_ids:
            des_path = os.path.join(valid_duration_path, dur)
        else:
            raise Exception(
                f"File {dur} not in either train or validation files.")
        shutil.move(full_dur_path, des_path)


def parser():
    """Parse arguments and set configuration parameters."""
    parser = argparse.ArgumentParser(
        description="Relocate duration datasets"
    )
    parser.add_argument(
        "--root-path",
        default=None,
        type=str,
        required=True,
        help="Root directory of preprocessed LJSpeech dataset.",
    )
    parser.add_argument(
        "--duration-path",
        default=None,
        type=str,
        required=True,
        help="Duration dataset path.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    relocate_duration(args.root_path, args.duration_path)
