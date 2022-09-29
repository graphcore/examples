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

import os
import sys
import re
import time
import unittest
import pytest
import ctypes
from pathlib import Path
import subprocess
import numpy as np
import torch
import poptorch
from poptorch.optim import AdamW
from options import train_options
import timm.optim.optim_factory as optim_factory
from core import models_mae

# Append mae directory
mae_root_path = str(Path(__file__).parent.parent)
sys.path.append(mae_root_path)


def alignment():
    cpu_grad = torch.load(f'{mae_root_path}/scripts/alignment/cpu/cpu_grad0.pt')
    ipu_grad = torch.load(f'{mae_root_path}/scripts/alignment/ipu/ipu_grad0.pt')
    for key in cpu_grad.keys():
        if key in ipu_grad.keys():
            if cpu_grad[key] is not None:
                grad_cpu = cpu_grad[key]
                grad_ipu = ipu_grad[key]
                np.testing.assert_allclose(
                    grad_cpu, grad_ipu, atol=1e-4, rtol=1e-6)


def get_data(bs, mask_ratio):
    img = torch.randn(bs, 3, 224, 224)
    N, L, D = bs, 196, 768
    noise = torch.rand(N, L)  # noise in [0, 1]
    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    len_keep = int(L * (1 - mask_ratio))
    keep_mat = torch.zeros(N, L, len_keep)
    ids_keep = ids_shuffle[:, :len_keep]
    keep_mat.scatter_(dim=1, index=ids_keep.view(N, 1, len_keep), value=1)
    restore_mat = torch.zeros(N, L, L)
    restore_mat.scatter_(dim=1, index=ids_restore.view(N, 1, L), value=1)

    mask = torch.ones([N, L])
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return img, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask


class TestMAE(unittest.TestCase):
    def test_alignment(self):
        cmd = "cd scripts; sh alignment.sh"
        ret = subprocess.check_call(cmd, shell=True, cwd=mae_root_path)
        assert ret == 0, f'alignment script execute error, ret code = {ret}'
        alignment()

    @pytest.mark.ipus(2)
    def test_mae_model(self):
        mask_ratio = 0.75
        bs = 1
        replica = 1
        accumulate = 16
        pipeline = [3, 3, 3, 3, 2, 2, 2, 2]
        half = False
        assert os.path.exists('./remap/remap_ops.so'), 'please compile custom op remap'
        ctypes.cdll.LoadLibrary('./remap/remap_ops.so')
        opts = train_options(
            False,
            gradient_accumulation_count=accumulate,
            replica=replica,
            half=half,
            ipu_per_replica=8)
        # ============ building  networks ... ============

        model = models_mae.__dict__[
            'mae_vit_base_patch16'](
            norm_pix_loss=True,
            pipeline=pipeline,
            device='ipu',
            mask_ratio=0.75,
            half=False)

        model.train()

        # ============ preparing optimizer ... ============
        param_groups = optim_factory.add_weight_decay(model, 0.05)
        optimizer = AdamW(param_groups, lr=None, betas=(0.9, 0.95))

        global_count = accumulate * replica
        ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        input_data, noise, ids_shuffle, ids_restore, keep_mat, restore_mat, mask = get_data(
            global_count * bs, mask_ratio)
        for i in range(100):
            s0 = time.time()
            _, loss = ipu_model(input_data, input_data, ids_restore,
                                keep_mat, restore_mat, mask)
            # batch_center = torch.mean(batch_center, dim=0, keepdim=True)
            s1 = time.time()
            tput = (global_count * bs) / (s1 - s0)
            with open(f'{mae_root_path}/test.log', 'a') as fw:
                fw.write(f'current tput is {int(tput)}\n')
