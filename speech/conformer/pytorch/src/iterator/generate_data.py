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

import torch
import popdist
from torch.utils.data import Dataset


class GenerateDataset(Dataset):
    def __init__(self, args):
        super(GenerateDataset, self).__init__()
        '''
        Args:
            args(dict): config_file is 'configs/train.yaml'
        '''
        self.args = args
        grad_accum = self.args['ipu_options']['gradient_accumulation']
        replicas = self.args['ipu_options']['num_replicas']
        batch_size = self.args['train_iterator']['batch_size']
        device_iterations = self.args['ipu_options']['device_iterations']
        num_workers = self.args['train_iterator']['num_workers']

        self.length = grad_accum * replicas * batch_size * device_iterations * num_workers * 2
        if popdist.isPopdistEnvSet():
            self.length *= self.args['NumInstances']
        self.random_seed = self.args['train_dataset']['random_seed']
        self.feature, self.feature_max_len, self.target, self.target_max_len = self.get_fixed_data()

    def get_fixed_data(self):
        torch.manual_seed(self.random_seed)
        input_size = self.args['encoder']['input_size']
        feature_max_len = self.args['encoder']['max_len']
        target_max_len = self.args['decoder']['max_len']
        feature = torch.randn(
            size=[self.args['encoder']['max_len'], input_size])
        if self.args['train_dataset']['dtype'] == "FLOAT16":
            feature = feature.half()
        target = torch.ones(
            size=[self.args['decoder']['max_len']], dtype=torch.long)
        return feature, feature_max_len, target, target_max_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (0, self.feature, self.feature_max_len, self.target, self.target, self.target_max_len)
