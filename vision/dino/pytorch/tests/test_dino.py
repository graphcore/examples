# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
from options import get_options, train_options
from core import utils
from core import vision_transformer as vits
from core.dino import DINOLoss, DINOHead, MultiCropWrapper
from core.gelu import ERF_GELU

# Append dino directory
dino_root_path = str(Path(__file__).parent.parent)
sys.path.append(dino_root_path)


def alignment():
    cpu_grad = torch.load(f'{dino_root_path}/script/alignment/cpu/cpu_grad0.pt')
    ipu_grad = torch.load(f'{dino_root_path}/script/alignment/ipu/ipu_grad0.pt')
    for key in cpu_grad.keys():
        if key in ipu_grad.keys():
            grad_cpu = cpu_grad[key]
            grad_ipu = ipu_grad[key]
            np.testing.assert_allclose(
                grad_cpu, grad_ipu, atol=1e-4, rtol=1e-6)


class TestDINO(unittest.TestCase):
    def test_alignment(self):
        cmd = "cd script; sh alignment.sh"
        ret = subprocess.check_call(cmd, shell=True, cwd=dino_root_path)
        assert ret == 0, f'alignment script execute error, ret code = {ret}'
        alignment()

    @pytest.mark.ipus(2)
    def test_dino_model(self):
        bs = 1
        replic = 1
        accumulate = 16
        ema_so = f'{dino_root_path}/ema/build/exp_avg_custom_op.so'
        pipeline = [3, 0]
        half = True
        crops = 4
        assert os.path.exists(ema_so), 'please compile custom op ema'
        libc = ctypes.cdll.LoadLibrary(ema_so)
        opts = train_options(
            pipeline=pipeline,
            ga=accumulate,
            replic=replic,
            half=half)
        # ============ building student and teacher networks ... ============
        student = vits.vit_mini()
        teacher = vits.vit_mini()
        embed_dim = student.embed_dim

        out_dim = 32768
        # ============ preparing loss ... ============
        dino_loss = DINOLoss(
            crops + 2,
        )
        # multi-crop wrapper handles forward with inputs of different
        # resolutions
        model = MultiCropWrapper(
            student,
            teacher,
            DINOHead(
                embed_dim,
                out_dim,
                act_layer=ERF_GELU
            ),
            DINOHead(
                embed_dim,
                out_dim,
                act_layer=ERF_GELU),
            dino_loss,
            0.996,
            device='ipu',
            pipeline=pipeline,
            half=True)

        model.student.half()
        model.teacher.half()
        model.train()

        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(model)
        optimizer = AdamW(params_groups,
                          lr=5e-4,
                          loss_scaling=128)

        center = torch.zeros(1, out_dim)
        global_count = accumulate * replic
        ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        ema_factor_base = torch.ones((global_count))
        ema_factor = ema_factor_base * 0.996
        teacher_temp_factor = 0.04 * torch.ones((global_count))
        input_224 = torch.randint(
            0, 255, (global_count * bs, 2, 3, 224, 224), dtype=torch.uint8)
        input_96 = torch.randint(
            0, 255, (global_count * bs, crops, 3, 96, 96), dtype=torch.uint8)
        for i in range(500):
            global_center = center.repeat(global_count, 1)
            s0 = time.time()
            batch_center, loss = ipu_model(
                input_224, input_96, ema_factor, global_center, teacher_temp_factor)
            batch_center = torch.mean(batch_center, dim=0, keepdim=True)
            center = center * 0.9 + batch_center * 0.1
            s1 = time.time()
            tput = (global_count * bs) / (s1 - s0)
            with open(f'{dino_root_path}/test.log', 'a') as fw:
                fw.write(f'current tput is {int(tput)}\n')
