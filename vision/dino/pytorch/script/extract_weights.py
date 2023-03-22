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
import datetime
import time
import math
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.append("..")
from core import utils
from core.gelu import ERF_GELU
from core.dino import DINOLoss, DINOHead, MultiCropWrapper
from core import vision_transformer as vits


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_base",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--out_dim",
        default=65536,
        type=int,
        help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this parameter to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=0,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument(
        "--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up."
    )
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training""",
    )
    parser.add_argument("--ema_so", type=str, default="./ema/build/exp_avg_custom_op.so", help="custom ema, path of so")
    parser.add_argument("--device", type=str, default="ipu", help="device to use")
    parser.add_argument("--pipeline", type=int, nargs="+", help="set modules on multi ipus")

    parser.add_argument("--weights", type=str, default="checkpoint.pth")
    parser.add_argument("--output", type=str)
    return parser


def extract_weight(args):
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknown architecture: {args.arch}")

    model = MultiCropWrapper(
        student,
        teacher,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
            act_layer=ERF_GELU,
        ),
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, act_layer=ERF_GELU),
        DINOLoss(
            # total number of crops = 2 global crops + local_crops_number
            args.local_crops_number
            + 2,
        ),
        args.momentum_teacher,
        device=args.device,
        pipeline=args.pipeline,
    )

    weight_path = args.weights
    assert os.path.exists(weight_path), f"{weight_path} not exists"
    model_state = torch.load(weight_path)
    state_dict = model_state["model"]
    model.load_state_dict(state_dict)
    teacher_weight = model.teacher.state_dict()
    torch.save(teacher_weight, f"{args.output}")
    print(f"Extract {args.output} from {weight_path} succeeded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()
    extract_weight(args)
