# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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


import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np


class CachedDataset(Dataset):
    def __init__(self, folder="cached_data"):
        super().__init__()
        self.folder = folder
        self.file_list = [os.path.join(self.folder, i) for i in os.listdir(folder)]
        self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = np.load(self.file_list[index], allow_pickle=True)
        ret = []
        for i in sample:
            try:
                tensor = torch.from_numpy(i)
            except:
                tensor = i
            ret.append(tensor)
        return ret


class GenCachedDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.generated_sample = [torch.zeros(189), torch.zeros(80, 870), 0, torch.zeros(189), torch.zeros(189), 0]

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        # Leaving index as a placeholder input for compatibility
        # Exact dimensions of a single sample
        return self.generated_sample
