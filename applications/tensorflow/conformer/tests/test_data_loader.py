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


import pytest
import kaldiio
import os


class TestData:

    def test_loader_with_sample_data(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from data_loader import Dataloader
        loader = Dataloader('datas/train', 768, 49, 4233, 83, use_synthetic_data=True)
        loader.load_data()
        (feat, feat_len, label, label_len) = next(loader())
        assert feat.shape[0] == 768
        assert feat.shape[2] == 1

    def test_loader_precision(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.chdir('../')
        from data_loader import Dataloader
        loader = Dataloader('datas/train', 768, 49, 4233, 83, dtype='FLOAT16', use_synthetic_data=True)
        loader.load_data()
        (feat, feat_len, label, label_len) = next(loader())
        assert feat.dtype == 'float16'

        loader = Dataloader('datas/train', 768, 49, 4233, 83, dtype='FLOAT32', use_synthetic_data=True)
        loader.load_data()
        (feat, feat_len, label, label_len) = next(loader())
        assert feat.dtype == 'float32'
