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
import os
import numpy
import torch


class CheckPoint:
    def __init__(self, epoch, logger, experiment_root=''):
        self.epoch = epoch
        self.logger = logger
        self.infermodel = None

    def load(self, pretrained_checkpoint, optimizer, model, is_train=True):
        if pretrained_checkpoint != '':
            assert os.path.exists(pretrained_checkpoint)
            training_state = torch.load(pretrained_checkpoint)
            if is_train:
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
            model_one_iter = training_state['model_weight']
            model.load_state_dict(model_one_iter)

            return training_state, model, optimizer

    def save(self, save_checkpoint_output_dir, model, optimizer, epoch, args, global_step):
        path = os.path.join(
            os.path.abspath(save_checkpoint_output_dir), f'epoch_{epoch}'
        )
        os.makedirs(path, exist_ok=True)
        self.logger.info(f'Saving checkpoint for epoch {epoch} to: {path}\n')
        torch.save(
            {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_weight': model.state_dict(),
                'config': args,
                'global_step': global_step,
            },
            os.path.join(path, 'training_state.pt'),
        )
