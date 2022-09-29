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
import yaml
import ctypes
from pathlib import Path
from functools import partial
import numpy as np
import torch
import poptorch
from poptorch.optim import AdamW
from options import get_options, train_options
from core import utils
from core import vision_transformer as vits
from core.dino import DINOLoss, DINOHead, MultiCropWrapper
from core.gelu import ERF_GELU
from core.utils import Precision
from script.alignment import *

# Append dino directory
dino_root_path = str(Path(__file__).parent.parent)
sys.path.append(dino_root_path)


def alignment():
    config = default_config()
    config['ema_so'] = f'{dino_root_path}/ema/build/exp_avg_custom_op.so'
    Path(config['output']).mkdir(parents=True, exist_ok=True)
    config['arch'] = 'vit_mini'
    config['pipeline'] = [3, 0]
    config['device'] = 'ipu'
    source_ipu, ipu_weights = process(config)
    # extract cpu grad and weights
    config['device'] = 'cpu'
    config['pipeline'] = None
    source_cpu, cpu_weights = process(config)

    all_pass = True
    max_diff = 0.01
    mean_diff = 0.0001
    wrong_number_thresh = 1e-4

    def get_number(x):
        if len(nonzero_ipu) == 0:
            return 0
        prod = 1
        for dim in x.shape:
            prod *= dim
        return prod

    for key in source_ipu.keys():
        diff_ipu = ipu_weights[key] - source_ipu[key]
        diff_cpu = cpu_weights[key] - source_cpu[key]
        #assert diff_ipu.shape == diff_cpu.shape
        zero_index = np.abs(diff_ipu) > 1e-6
        nonzero_cpu = diff_cpu[zero_index]
        nonzero_ipu = diff_ipu[zero_index]
        #diff = np.abs(np.where(nonzero_ipu - nonzero_cpu < 1e-4, 0, nonzero_ipu - nonzero_cpu) / nonzero_ipu)
        # if all zero
        if len(nonzero_ipu) == 0:
            continue
        # if mean is zero
        if np.abs(nonzero_ipu).mean() == 0:
            assert((nonzero_ipu - nonzero_cpu).mean() < 1e-8)
        diff = np.abs((nonzero_ipu - nonzero_cpu) / np.abs(nonzero_ipu).mean())
        if (diff < max_diff).all() and diff.mean() < mean_diff:
            continue

        wrong_percent = (diff > max_diff).sum() / get_number(diff_ipu)
        if wrong_percent < wrong_number_thresh:
            # if wrong nubmer is not too many
            print('WARN, some wrong numbers:')
        else:
            print('ERROR, some wrong numbers:')
            all_pass = False
        print(f'{key} mean: {diff.mean()} , std:{diff.std()}, max:{diff.max()}, wrong percent:{wrong_percent}')
        print('cpu values:', nonzero_cpu[diff > max_diff])
        print('ipu values:', nonzero_ipu[diff > max_diff])
    assert all_pass


class TestDINO(unittest.TestCase):
    def test_alignment(self):
        alignment()

    def test_dino_model(self):
        # Load the yaml
        args = dict()
        config_name = "vit_base_pytest"
        with open(f"{dino_root_path}/configs.yml", "r") as f:
            try:
                args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

        bs = args['batch_size']
        replica = args['replica']
        accumulate = args['ga']
        ema_so = f'{dino_root_path}/ema/build/exp_avg_custom_op.so'
        pipeline = args['pipeline']
        precision = Precision(args['precision'])
        gelu = partial(ERF_GELU, precision=precision)
        assert os.path.exists(ema_so), 'please compile custom op ema'
        libc = ctypes.cdll.LoadLibrary(ema_so)
        opts = train_options(
            pipeline=pipeline,
            ga=accumulate,
            replica=replica,
            precision=precision)
        # ============ building student and teacher networks ... ============
        student = vits.__dict__[args['arch']](
            drop_path_rate=args['drop_path_rate'],
            act_layer=gelu,
            precision=precision
        )
        teacher = vits.__dict__[args['arch']](
            act_layer=gelu,
            precision=precision
        )
        embed_dim = student.embed_dim

        out_dim = 1024
        # ============ preparing loss ... ============
        dino_loss = DINOLoss(
            out_dim,
            args['local_crops_number'] + 2,
        )
        # multi-crop wrapper handles forward with inputs of different
        # resolutions
        model = MultiCropWrapper(
            student,
            teacher,
            DINOHead(
                embed_dim,
                out_dim,
                act_layer=gelu,
                precision=precision,
            ),
            DINOHead(
                embed_dim,
                out_dim,
                act_layer=gelu,
                precision=precision,
            ),
            dino_loss,
            args['momentum_teacher'],
            device='ipu',
            pipeline=pipeline,
            precision=precision)

        model.half()
        model.train()

        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(model)
        optimizer = AdamW(params_groups,
                          lr=5e-4,
                          loss_scaling=128)

        center = torch.zeros(1, out_dim)
        global_count = accumulate * replica
        ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        ema_factor_base = torch.ones((global_count))
        ema_factor = ema_factor_base * 0.996
        teacher_temp_factor = 0.04 * torch.ones((global_count))
        input_224 = torch.randint(
            0, 255, (global_count * bs, 2, 3, 224, 224), dtype=torch.uint8)
        input_96 = torch.randint(
            0,
            255,
            (global_count * bs,
             args['local_crops_number'],
                3,
                96,
                96),
            dtype=torch.uint8)
        for i in range(50):
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

