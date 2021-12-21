# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy
import editdistance
import torch
from itertools import groupby


def get_kl_acc(y_pred, y_true, ignore_id=0):
    """Calculate the accuracy of the conformer model in each validate step"""

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    mask = y_true != ignore_id
    numerator = torch.sum(y_pred[mask] == y_true[mask], axis = 0)
    denominator = torch.sum(mask, axis = 0)

    return float(numerator) / float(denominator)


def get_char_dict(dict_path):
    char_dict = {}
    index = 0
    with open(dict_path, 'r') as fp:
        for item in fp.readlines():
            item = item.strip()
            char_dict[int(index)] = item
            index += 1
    return char_dict


def get_cer(y_pred, y_true, dict_path, blank_id=0):
    y_pred = numpy.reshape(y_pred, [-1, y_pred.shape[-1]])
    y_true = numpy.reshape(y_true, [-1, y_true.shape[-1]])
    char_dict = get_char_dict(dict_path)

    cers, char_ref_lens = [], []
    for i, y in enumerate(y_pred):
        y_hat_i = [x[0] for x in groupby(y)]
        y_true_i = y_true[i]
        seq_hat, seq_true = [], []
        for idx in y_hat_i:
            idx = int(idx)
            if idx in char_dict.keys():
                seq_hat.append(char_dict[int(idx)])

        for idx in y_true_i:
            idx = int(idx)
            if idx in char_dict.keys():
                seq_true.append(char_dict[int(idx)])

        hyp_chars = "".join(seq_hat)
        ref_chars = "".join(seq_true)
        # Here only compare the ones before <sos/eos> which is the predicted value of the correctly identified part and the label
        hyp_chars = hyp_chars.split('<sos/eos>')[0]
        ref_chars = ref_chars.split('<sos/eos>')[0]
        if len(ref_chars) > 0:
            cers.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))

    cer = float(sum(cers)) / sum(char_ref_lens) if cers else None

    return cer
