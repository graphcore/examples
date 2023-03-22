# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
#
# This file has been modified by Graphcore Ltd.

from itertools import chain
from config import GPTJConfig


def group_texts(config: GPTJConfig):
    seq_len_1 = config.model.sequence_length + 1

    def func(examples):
        # Concatenate all texts.
        inputs = list(chain(*examples["input_ids"]))
        total_length = len(inputs)
        # We drop the small remainder instead of padding
        if total_length >= seq_len_1:
            total_length = (total_length // seq_len_1) * seq_len_1
        # Split by chunks of max_len.
        data = [inputs[i : i + seq_len_1] for i in range(0, total_length, seq_len_1)]
        result = {
            "input_ids": [d[:-1] for d in data],
            "labels": [d[1:] for d in data],
        }
        return result

    return func
