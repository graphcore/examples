# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
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


import os
import json
import kaldiio
import numpy as np


class Dataloader(object):
    def __init__(self, data_path, max_wav_len, max_lab_len,
                 vocab_size, feat_dim, training=False, dtype='FLOAT16', use_synthetic_data=False):
        self.data_path = data_path
        self.max_wav_len = max_wav_len
        self.max_lab_len = max_lab_len
        self.training = training
        self.use_synthetic_data = use_synthetic_data  # use fake data
        self.dtype = np.float16 if dtype == 'FLOAT16' else np.float32

        self.sos = vocab_size - 1
        self.eos = self.sos
        self.out_of_len = 0
        self.feat_dim = feat_dim
        self.utts = {}


    def load_data(self):
        if self.use_synthetic_data is True:
            print('using syntheticdata.')
            return
        self.filename = self.data_path + '/data.json'

        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as fp:
                raw_data = json.load(fp)['utts']
                for key in raw_data.keys():
                    if raw_data[key]['input'][0]['shape'][0] > self.max_wav_len:
                        self.out_of_len += 1
                        continue
                    utt_feat = raw_data[key]['input'][0]['feat'].replace('/data//', 'data/')
                    utt_target = raw_data[key]['output'][0]['tokenid']
                    self.utts[key] = (utt_feat, utt_target)
            print("Un-Valid {} utts, Valid {}".format(self.out_of_len, len(self.utts)))
        else:
            raise IOError(f"Dataset file {self.filename} not found")


    @property
    def num_utts(self):
        if self.use_synthetic_data is True:
            return 400
        else:
            return len(self.utts.keys())


    def __call__(self):
        while True:
            if self.use_synthetic_data is True:
                feat_pad = np.random.rand(self.max_wav_len, self.feat_dim, 1).astype(self.dtype)
                feat_len = (((self.max_wav_len - 3) // 2 + 1) - 3) // 2 + 1
                feat_len = np.array(feat_len, dtype=np.int32)
                label_pad = np.random.randint(low=0, high=10, size=(self.max_lab_len,)).astype(np.int32)
                label_len = np.array(label_pad.shape[0], dtype=np.int32)
                yield feat_pad, feat_len, label_pad, label_len
            else:
                utt_keys = list(self.utts.keys())
                if self.training:
                    np.random.shuffle(utt_keys)

                for key in utt_keys:
                    feat = kaldiio.load_mat(self.utts[key][0])
                    label = [int(lab) for lab in self.utts[key][1].strip().split(' ')]

                    feat_pad = np.zeros((self.max_wav_len, self.feat_dim, 1)).astype(feat.dtype)
                    label_pad = np.zeros((self.max_lab_len)).astype(np.int32)

                    feat_pad[:feat.shape[0], :, 0] = feat
                    feat_pad = feat_pad.astype(self.dtype)
                    label_pad[0] = self.sos
                    label_pad[1:len(label)+1] = np.asarray(label).astype(np.int32)
                    label_pad[len(label)+1] = self.eos
                    # conv2d sub-sampling, kernel size 3, stride 2
                    feat_len = (((self.max_wav_len - 3) // 2 + 1) - 3) // 2 + 1
                    feat_len = np.array(feat_len, dtype=np.int32)
                    label_len = np.array(len(label) + 1, dtype=np.int32)
                    yield feat_pad, feat_len, label_pad, label_len
