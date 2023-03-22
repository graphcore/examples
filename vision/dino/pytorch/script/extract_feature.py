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
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import poptorch

sys.path.append("..")
from core import vision_transformer as vits


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Model parameters
    parser.add_argument("--arch", default="vit_base", type=str, choices=["vit_tiny", "vit_small", "vit_base"])
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    # Misc
    parser.add_argument("--data_path", default="", type=str, help="Please specify path to the ImageNet training data.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--num_workers", default=32, type=int, help="Number of data loading workers.")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--output", type=str)

    # IPU
    parser.add_argument("--half", action="store_true", help="use half")
    parser.add_argument("--di", type=int, default=256, help="device iterations number")
    parser.add_argument("--replic", type=int, default=4, help="device iterations number")
    return parser


def get_dataset(args):
    if args.half:
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Lambda(lambda img: img.half()),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    dataset = datasets.ImageFolder(os.path.join(args.data_path), transform=transform)
    return dataset


def extract_features(model, data_loader):
    features = []
    labels = []
    end = time.time()
    for it, (samples, label) in enumerate(data_loader):
        feats = model(samples)
        use_time = time.time() - end
        end = time.time()
        print(f"[{it}/{len(data_loader)}]\ttime:{use_time:.2f}\t{feats.shape}\ttput:{(feats.shape[0]/use_time):.1f}")
        features.append(feats)
        labels.append(label)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def main(args):
    opts = poptorch.Options()
    opts.deviceIterations(args.di)
    opts.replicationFactor(args.replic)
    opts.enableExecutableCaching("./cachedir")

    # ============ preparing data ... ============
    dataset = get_dataset(args)
    data_loader = poptorch.DataLoader(
        options=opts,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"Data loaded: {len(dataset)} images. steps is {len(data_loader)}")

    # ============ building student and teacher networks ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size)

    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)
    print(f"load {args.weights} success.")
    if args.half:
        print("use half for extract features")
        model.half()

    ipu_model = poptorch.inferenceModel(model.eval(), options=opts)

    features, labels = extract_features(ipu_model, data_loader)
    features = nn.functional.normalize(features, dim=1, p=2)
    torch.save({"features": features, "labels": labels}, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
