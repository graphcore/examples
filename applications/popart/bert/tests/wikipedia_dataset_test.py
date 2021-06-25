# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

import json
from tqdm import tqdm
from bert_data.pretraining_dataset import CachedDataLoader
import pytest


@pytest.mark.skip(reason="Extremely long, about 90 hours")
def test_wikipedia_dataset():
    '''
    Go through binary files, cached and shuffle them, then check that all examples are properly formatted.
    '''

    # Read config and get inputs
    json_config = "configs/mk1/pretrain_large_128.json"
    with open(json_config, "r") as cfg:
        config = json.loads(cfg.read())
    sequence_length = config['sequence_length']
    mask_tokens = config['mask_tokens']
    vocab_length = 30522  # Original vocab_length used to generate dataset

    # Data loader
    dl = CachedDataLoader(
        config['input_files'],
        [sequence_length, sequence_length, sequence_length, 1, 1, mask_tokens, 1],
        start_data_at_epoch=4,
        shuffle=True)
    dl_itr = iter(dl)

    # Go through each data point and check that it is correctly formed
    total_remplacements = 0
    examples = 0

    for i in tqdm(range(len(dl))):

        tokens, positions, sentence_ids, padding_0_start, padding_1_start, mlm_labels, nsp_label = next(dl_itr)

        success, replacements = check_tokens(tokens, padding_0_start, padding_1_start, vocab_length, mask_tokens)
        assert (success)
        assert(check_positions(positions, padding_0_start, padding_1_start, sequence_length))
        assert(check_sentence_ids(sentence_ids, padding_1_start))
        assert(check_labels(mlm_labels, nsp_label, vocab_length))


        total_remplacements += replacements
        examples += 1

    assert (examples == len(dl))
    assert (0.19 < (total_remplacements / (examples * mask_tokens) < 0.21))  # Assumption 20% replacements


def check_tokens(tokens, padding_0_start, padding_1_start, vocab_length, mask_tokens):

        seen_101 = False
        seen_102 = False
        replacements = 0
        for index, value in enumerate(tokens[0]):

                if value == 100:  # [UNK]
                    continue

                if index < padding_0_start:
                    if value != 103:  # [MSK]
                        replacements += 1

                elif seen_101 is False:  # [CLS]
                    if value == 101:
                        seen_101 = True
                    elif value != 0:
                        print(tokens, f"Value {value} at index {index} is incorrect")
                        return False, replacements

                elif seen_102 is False:  # [SEP]
                    if value == 102:
                        seen_102 = True
                    elif not (999 <= value < vocab_length):
                        print(tokens, f"Value {value} at index {index} is incorrect")
                        return False, replacements

                elif seen_102 is True and index >= padding_1_start:
                    if value != 0:
                        print(tokens, f"Value {value} at index {index} is incorrect")
                        return False, replacements

        return True, replacements


def check_positions(positions, padding_0_start, padding_1_start, sequence_length):
    for index, value in enumerate(positions[0]):

        if index < padding_0_start:
            if not(1 <= value < sequence_length):
                print(positions, f"Value {value} at index {index} is incorrect")
                return False

        elif index < padding_1_start:
            if value != 384 and not(0 <= value < sequence_length):
                print(positions, f"Value {value} at index {index} is incorrect")
                return False

        else:
            if value != 384:
                print(positions, f"Value {value} at index {index} is incorrect")
                return False

    return True


def check_sentence_ids(sentence_ids, padding_1_start):
    for index, value in enumerate(sentence_ids[0]):

        if index < padding_1_start:
                    if value != 0 and value != 1:
                        print(sentence_ids, f"Value {value} at index {index} is incorrect")
                        return False
        else:
            if value != 0:
                print(sentence_ids, f"Value {value} at index {index} is incorrect")
                return False

    return True


def check_labels(mlm_labels, nsp_label, vocab_length):
    for index, value in enumerate(mlm_labels[0]):
        if value != 0 and value != 100 and not(999 <= value < vocab_length):
            print(mlm_labels, f"Value {value} at index {index} is incorrect")
            return False

    if nsp_label[0] != 0 and nsp_label[0] != 1:
        print(nsp_label, f"Value {nsp_label[0]} of nsp_label is incorrect")
        return False

    return True
