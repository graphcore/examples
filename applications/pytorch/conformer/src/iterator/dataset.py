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

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from espnet2.asr.specaug.specaug import SpecAug
import popdist


def load_feat_length_npy(path):
    return np.load(path).item()


class AishellDataset(Dataset):
    def __init__(
        self,
        vocab,
        args,
        dataset,
    ):
        super(AishellDataset, self).__init__()
        '''
        Args:
            vocab: vocab module instance
            args(dict): config_file is 'configs/train.yaml'
            dataset(dict): Choose one of the two from the train.yaml('train_dataset' or 'val_dataset')
        '''
        self.data_root = dataset['data_root']
        self.feats_file = dataset['feats_file']
        self.feats_length_file = dataset['feats_length_file']
        self.target_file = dataset['target_file']
        self.target_length_file = dataset['target_length_file']
        self.feature_max_length = dataset['feature_max_length']
        self.target_max_length = dataset['target_max_length']
        self.vocab = vocab
        self.cached_data_path = dataset['cached_data_path']
        if 'use_generated_data' in dataset:
            self.use_generated_data = dataset['use_generated_data']
        else:
            self.use_generated_data = False
        self.data = None
        self.args = args
        self.length = self.args['ipu_options']['gradient_accumulation'] * self.args['ipu_options']['num_replicas'] * self.args['train_iterator']['batch_size'] * self.args['ipu_options']['batches_per_step'] * 2
        if popdist.isPopdistEnvSet():
            self.length *= self.args['NumInstances']

        if not self.use_generated_data:
            self.load_real_data()
        else:
            self.random_seed = dataset['random_seed']
            self.feature, self.feature_max_len, self.target, self.target_max_len = self.get_fixed_data()
        self.dtype = dataset['dtype']


    def get_fixed_data(self):
        torch.manual_seed(self.random_seed)
        input_size = self.args['encoder']['input_size']
        feature_max_len = self.args['encoder']['max_len']
        target_max_len = self.args['decoder']['max_len']
        feature = torch.randn(feature_max_len, input_size)
        return feature, feature_max_len, torch.ones(target_max_len).int(), target_max_len - 1

    def get_generated_data(self):
        return (self.feature, self.feature_max_len, self.target, self.target_max_len)

    def load_real_data(self):
        if self.cached_data_path:
            if os.path.exists(self.cached_data_path):
                self.data = pd.read_json(self.cached_data_path, orient='index')
                self.length = len(self.data)
                self.data = self.data.to_dict('index')
            else:
                self._load_data()
                self._save_data()
        else:
            self._load_data()

    def _load_data(self):
        feats = pd.read_csv(self.feats_file, sep=' ', names=['feat_dir'], index_col=0)
        feats_length = pd.read_csv(
            self.feats_length_file, sep=' ', names=['feat_length'], index_col=0
        )
        target = pd.read_csv(self.target_file, sep=' ', names=['target'], index_col=0)
        target_length = pd.read_csv(
            self.target_length_file, sep=' ', names=['target_length'], index_col=0
        )
        data = feats.join(feats_length).join(target).join(target_length)
        data['feat_dir'] = data.feat_dir.apply(
            lambda x: os.path.join(self.data_root, x)
        )
        data['feat_length'] = data.feat_length.apply(
            lambda x: os.path.join(self.data_root, x)
        )
        data['feat_length'] = data.feat_length.apply(lambda x: load_feat_length_npy(x))
        data['target'] = data.target.apply(lambda x: self.vocab.str2id(x))
        data = data[data.feat_length < self.feature_max_length]
        data = data[data.target_length < self.target_max_length - 1]
        self.length = len(data)
        self.data = data.reset_index(drop=True).to_dict('index')

    def _save_data(self):
        df = pd.DataFrame.from_dict(self.data, orient='index')
        df.to_json(self.cached_data_path, orient='index')

    def get_real_data(self, index):
        sample = self.data[index]
        feature = np.load(sample['feat_dir'])
        pad_feature = np.zeros(
            shape=(self.feature_max_length, feature.shape[1]), dtype=np.float32
        )
        pad_feature[: feature.shape[0], :] = feature
        feature_length = sample['feat_length']
        target = sample['target']
        pad_target = np.zeros(shape=(self.target_max_length), dtype=np.int32)
        pad_target[: len(target)] = target
        target_length = sample['target_length']
        feature_input = torch.from_numpy(pad_feature)
        return feature_input, feature_length, torch.from_numpy(pad_target), target_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.use_generated_data:
            return self.get_generated_data()
        else:
            return self.get_real_data(index)


class CollateFn:
    def __init__(self, sos, eos, is_spec_aug=False, dtype='FLOAT16'):
        '''
        Args:
            sos: sos token id
            eos: eos token id
            is_spec_aug(bool): Whether to use feature spec augment
        '''
        self.sos = sos
        self.eos = eos
        self.is_spec_aug = is_spec_aug
        self.spec_aug = SpecAug()
        self.dtype = dtype

    def __call__(self, batch):
        batch = list(zip(*batch))
        feature = torch.stack(batch[0])
        feature_length = torch.IntTensor(batch[1])
        target = torch.stack(batch[2])
        target_length = torch.IntTensor(batch[3])
        target_out = target.scatter(1, target_length.unsqueeze(1).long(), self.eos)
        target_in = torch.nn.functional.pad(
            target, (1, 0), mode='constant', value=self.sos
        )[:, :-1]
        if self.is_spec_aug:
            feature, length = self.spec_aug(feature)
        feature = feature.half() if self.dtype == 'FLOAT16' else feature
        return feature, feature_length, target_in, target_out, target_length + 1
