# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
import sys


def save_checkpoint(epoch, model, optimizer, path):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
    torch.save(save_state, f'{path}/model_{epoch}.pth')
    torch.save(save_state, f'{path}/checkpoint.pth')


def load_checkpoint(model, optimizer, path):
    assert os.path.exists(path), f'{path} not exists'
    model_state = torch.load(path)
    epoch = model_state['epoch']
    weights = model_state['model']
    optimizer_weights = model_state['optimizer']

    model.load_state_dict(weights)
    optimizer.load_state_dict(optimizer_weights)
    return epoch
