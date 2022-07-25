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


def get_char_dict(dict_path):
    char_dict = {}
    with open(dict_path, 'r') as fp:
        for item in fp.readlines():
            char, index = item.strip().split(" ")
            char_dict[int(index)] = char
    return char_dict


def get_recog_predict(y_pred, char_dict, key):
    y_pred = numpy.array(y_pred)

    y_pred = numpy.reshape(y_pred, [-1, y_pred.shape[-1]])
    pre_l = []
    for i, y in enumerate(y_pred):
        key_ = key[i]
        hyp_chars = "".join(char_dict[int(idx)] for idx in y if char_dict.get(int(idx)))
        # Here only compare the ones before <sos/eos> which is the predicted value of the correctly identified part and the ref_l
        hyp_chars = hyp_chars.split('<sos/eos>')[0]
        # Here append the predict and reference's key in order to compute_cer
        pre_l.append(key_+" "+hyp_chars+"\n")
    return pre_l
