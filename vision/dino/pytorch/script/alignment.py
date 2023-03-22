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
import math
import json
import ctypes
import time
import copy
import datetime
from pathlib import Path
import numpy as np
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import poptorch
from poptorch.optim import SGD, AdamW

sys.path.append("..")
from core.gelu import ERF_GELU
from core import utils
from core import vision_transformer as vits
from core.dino import DINOLoss, DINOHead, MultiCropWrapper
from options import alignment_options, get_options


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)
    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_base",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base", "vit_mini"],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--out_dim",
        default=1024,
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
    parser.add_argument("--pipeline", default=None, type=int, nargs="+", help="set modules on multi ipus")
    parser.add_argument("--alignment", default=True, type=utils.bool_flag)
    parser.add_argument("--extract_name", action="store_true")
    parser.add_argument("--grad_compare", action="store_true")
    parser.add_argument("--device", type=str, default="ipu", help="device to use")
    parser.add_argument(
        "--ema_so", type=str, default="../ema/build/exp_avg_custom_op.so", help="custom ema, path of so"
    )
    parser.add_argument("--output", type=str, default="./alignment")
    parser.add_argument("--grad", type=str, default="grad_names.pt")
    parser.add_argument("--alignment_pipeline", action="store_true")
    parser.add_argument("--ga", default=12, type=int)
    return parser


def default_config(args=None):
    config = {}
    config["arch"] = "vit_base"
    config["out_dim"] = 1024
    config["norm_last_layer"] = True
    config["momentum_teacher"] = 0.0
    config["batch_size"] = 1
    config["drop_path_rate"] = 0.0
    config["local_crops_number"] = 2
    config["pipeline"] = None
    config["alignment"] = True
    config["extract_name"] = False
    config["grad_compare"] = False
    config["seed"] = 0
    config["device"] = "ipu"
    config["ema_so"] = "../ema/build/exp_avg_custom_op.so"
    config["output"] = "./alignment"
    config["grad"] = "grad_names.pt"
    config["alignment_pipeline"] = False
    config["ga"] = 12
    config["count"] = 1
    if args is not None:
        config["arch"] = args.arch
        config["out_dim"] = args.out_dim
        config["norm_last_layer"] = args.norm_last_layer
        config["pipeline"] = args.pipeline
        config["alignment"] = args.alignment
        config["extract_name"] = args.extract_name
        config["grad_compare"] = args.grad_compare
        config["device"] = args.device
        config["ema_so"] = args.ema_so
        config["output"] = args.output
        config["grad"] = args.grad
        config["alignment_pipeline"] = args.alignment_pipeline
        config["ga"] = args.ga
    return config


def extract_name(config, model, optimizer, center):
    path = os.path.join(config["output"], config["grad"])
    if os.path.exists(path):
        return
    assert os.path.exists(config["ema_so"]), "please compile custom op ema"
    libc = ctypes.cdll.LoadLibrary(config["ema_so"])
    opts = alignment_options()
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    img1 = torch.randint(0, 255, (1, 2, 3, 224, 224), dtype=torch.uint8)
    img2 = torch.randint(0, 255, (1, config["local_crops_number"], 3, 96, 96), dtype=torch.uint8)
    ema_factor_base = torch.ones((1))
    ema_factor = ema_factor_base * config["momentum_teacher"]
    teacher_temp_factor = 0.04 * torch.ones((1))
    _, loss = ipu_model(img1, img2, ema_factor, center, teacher_temp_factor)
    tensor_names = ipu_model.getTensorNames()
    torch.save(tensor_names, path)
    return tensor_names


def load_weight(model, path):
    state_dict = torch.load(path)
    new_dict = OrderedDict()
    for n, v in model.state_dict().items():
        if n in state_dict:
            new_dict[n] = state_dict[n]
        else:
            print(f"{n} not in state dict")
            new_dict[n] = v
    model.load_state_dict(new_dict)


def shard_alignment(config, model, optimizer, center):
    img_name = os.path.join(config["output"], "image.pth")
    if os.path.exists(img_name):
        img1, img2 = torch.load(img_name)
        print("load image")
    else:
        bs = config["batch_size"]
        img1 = torch.randint(0, 255, (bs, 2, 3, 224, 224), dtype=torch.uint8)
        img2 = torch.randint(0, 255, (bs, config["local_crops_number"], 3, 96, 96), dtype=torch.uint8)
        torch.save([img1, img2], img_name)

    compare_weights = False
    dir_path = os.path.join(config["output"], config["device"])
    os.makedirs(dir_path, exist_ok=True)
    count = config["count"]
    grad_count = config["count"]
    ema_factor_base = torch.ones((1))
    ema_factor = ema_factor_base * config["momentum_teacher"]
    global_center = center.repeat(config["batch_size"], 1)
    teacher_temp_factor = 0.04 * torch.ones((config["batch_size"]))
    result = None
    if config["device"] == "ipu":
        assert os.path.exists(config["ema_so"]), "please compile custom op ema"
        libc = ctypes.cdll.LoadLibrary(config["ema_so"])
        opts = alignment_options()
        if config["grad_compare"]:
            result = grad_compare(
                model,
                opts,
                optimizer,
                center,
                teacher_temp_factor,
                img1,
                img2,
                ema_factor,
                os.path.join(config["output"], config["grad"]),
                dir_path,
                grad_count,
            )
        else:
            ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
            for i in range(count):
                logits, loss = ipu_model(img1, img2, ema_factor, global_center, teacher_temp_factor)
                print(loss)
                torch.save(logits, f"{dir_path}/logits_{i}.pth")
                torch.save(ipu_model.state_dict(), f"{dir_path}/model{i}.pth")
                if i == 0:
                    result = ipu_model.state_dict()

    else:
        if config["grad_compare"]:
            for i in range(grad_count):
                grad_dict = {}
                optimizer.zero_grad()
                logits, loss = model(img1, img2, ema_factor, center, teacher_temp_factor)
                loss.backward()
                for n, v in model.named_parameters():
                    grad_dict[f"Gradient___model.{n}"] = v.grad
                optimizer.step()
                model.gpu_update_teacher()
                torch.save(grad_dict, f"{dir_path}/cpu_grad{i}.pt")
                torch.save(model.state_dict(), f"{dir_path}/cpu{i}.pt")
                if i == 0:
                    result = copy.deepcopy(grad_dict)
        else:
            for i in range(count):
                optimizer.zero_grad()
                logits, loss = model(img1, img2, ema_factor, center, teacher_temp_factor)
                print(loss)
                loss.backward()
                optimizer.step()
                model.gpu_update_teacher()
                torch.save(logits, f"{dir_path}/logits_{i}.pth")
                torch.save(model.state_dict(), f"{dir_path}/model{i}.pth")
                if i == 0:
                    result = model.state_dict()
    return result


def grad_compare(model, opts, optimizer, center, teacher_temp_factor, img1, img2, ema_factor, path, dir_path, steps):
    name_list = torch.load(path)
    grad_list = []
    result = None
    for i, name in enumerate(name_list):
        if "Gradient___model." in name or "UpdatedVar___model." in name:
            print(name)
            opts.anchorTensor(name, name)
            grad_list.append(name)
    ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
    for i in range(steps):
        _, loss = ipu_model(img1, img2, ema_factor, center, teacher_temp_factor)
        grad_dict = {}
        for name in grad_list:
            grad_ipu = ipu_model.getAnchoredTensor(name)
            grad_dict[name] = grad_ipu
        torch.save(grad_dict, f"{dir_path}/ipu_grad{i}.pt")
        torch.save(ipu_model.state_dict(), f"{dir_path}/ipu{i}.pt")
        if i == 0:
            result = copy.deepcopy(grad_dict)
    return result


def pipeline_compare(config, model, optimizer, center):
    img_name = os.path.join(config["output"], "image_pipeline.pth")
    ga = config["ga"]
    if os.path.exists(img_name):
        img1, img2 = torch.load(img_name)
        print("load image")
    else:
        bs = config["batch_size"]
        img1 = torch.randint(0, 255, (bs, 2, 3, 224, 224), dtype=torch.uint8)
        img2 = torch.randint(0, 255, (bs, config["local_crops_number"], 3, 96, 96), dtype=torch.uint8)
        torch.save([img1, img2], img_name)

    dir_path = os.path.join(config["output"], config["device"])
    os.makedirs(dir_path, exist_ok=True)

    count = config["count"]
    ema_factor_base = torch.ones((1))
    ema_factor = ema_factor_base * config["momentum_teacher"]
    global_center = center.repeat(config["ga"], 1)
    teacher_temp_factor = 0.04 * torch.ones((config["ga"]))
    result = None
    if config["device"] == "ipu":
        img1 = torch.cat([img1 for _ in range(ga)])
        img2 = torch.cat([img2 for _ in range(ga)])
        ema_factor = torch.cat([ema_factor for _ in range(ga)])
        assert os.path.exists(config["ema_so"]), "please compile custom op ema"
        libc = ctypes.cdll.LoadLibrary(config["ema_so"])
        opts = get_options(ga)

        ipu_model = poptorch.trainingModel(model, opts, optimizer=optimizer)
        for i in range(count):
            logits, loss = ipu_model(img1, img2, ema_factor, global_center, teacher_temp_factor)
            print(loss)
            torch.save(ipu_model.state_dict(), f"{dir_path}/pipeline_model{i}.pth")
            if i == 0:
                result = ipu_model.state_dict()

    else:
        for i in range(count):
            optimizer.zero_grad()
            losses = []
            for j in range(ga):
                logits, loss = model(img1, img2, ema_factor, center, teacher_temp_factor)
                losses.append(loss.item())
                loss.backward()
            optimizer.step()
            model.gpu_update_teacher()
            print(losses)
            torch.save(model.state_dict(), f"{dir_path}/pipeline_model{i}.pth")
            if i == 0:
                result = model.state_dict()
    return result


def process(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if config["arch"] in vits.__dict__.keys():
        student = vits.__dict__[config["arch"]](drop_path_rate=config["drop_path_rate"], act_layer=ERF_GELU)
        teacher = vits.__dict__[config["arch"]](act_layer=ERF_GELU)
        embed_dim = student.embed_dim
    else:
        print(f"Unknown architecture: {config['arch']}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    model = MultiCropWrapper(
        student,
        teacher,
        DINOHead(embed_dim, config["out_dim"], norm_last_layer=config["norm_last_layer"], act_layer=ERF_GELU),
        DINOHead(embed_dim, config["out_dim"], act_layer=ERF_GELU),
        DINOLoss(
            # total number of crops = 2 global crops + local_crops_number
            config["local_crops_number"]
            + 2
        ),
        config["momentum_teacher"],
        device=config["device"],
        pipeline=config["pipeline"],
        alignment=config["alignment"],
    )

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)

    # ============ building student and teacher networks ... ============
    optimizer = AdamW(params_groups, lr=1.0, eps=1e-5)

    optimizer.param_groups[2]["lr"] = 0.0
    optimizer.param_groups[3]["lr"] = 0.0
    state_name = os.path.join(config["output"], "model.pth")
    if os.path.exists(state_name):
        load_weight(model, state_name)
        print("load_weights")
    else:
        torch.save(model.state_dict(), state_name)

    model.train()
    source_weights = copy.deepcopy(model.state_dict())
    center = torch.zeros(1, config["out_dim"])
    if config["extract_name"]:
        return extract_name(config, model, optimizer, center)

    if config["alignment_pipeline"]:
        print("alignment pipeline")
        weights = pipeline_compare(config, model, optimizer, center)
        return source_weights, weights
    else:
        print("alignment shard")
        weights = shard_alignment(config, model, optimizer, center)
        return source_weights, weights


def main():
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()
    config = default_config(args)
    Path(config["output"]).mkdir(parents=True, exist_ok=True)
    process(config)


if __name__ == "__main__":
    main()
