# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2022 Facebook, Inc. and its affiliates.
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
#
# This file has been modified by Graphcore Ltd.
import os
import argparse
import json
import time
from pathlib import Path
from collections import OrderedDict
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms as pth_transforms
import poptorch
from poptorch.optim import SGD
import core.utils as utils
from core.utils import AverageMeter
import core.vision_transformer as vits


def load_pretrain(model, path, backbone_key):
    assert os.path.exists(path), f'{path} not exists!'
    state_dict = torch.load(path)
    epoch = state_dict['epoch'] + 1
    model_state = state_dict['model']
    backbone_dict = OrderedDict()
    for k, v in model_state.items():
        if backbone_key in k and 'head' not in k:
            name = k[len(backbone_key) + 1:]
            backbone_dict[name] = v
    model.backbone.load_state_dict(backbone_dict)


def save_checkpoint(epoch, model, optimizer, path):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
    torch.save(save_state, f'{path}/linear_{epoch}.pth')
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


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)


class VitLinear(nn.Module):
    def __init__(self, arch, patch_size, num_labels, n_last_blocks):
        super().__init__()
        self.n = n_last_blocks
        self.backbone = vits.__dict__[args.arch](
            patch_size=patch_size, num_classes=0)
        embed_dim = self.backbone.embed_dim * n_last_blocks
        self.linear_classifier = LinearClassifier(
            embed_dim, num_labels=num_labels)
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, inp, target=None):
        with torch.no_grad():
            if self.n == 1:
                output = self.backbone(inp)
            else:
                intermediate_output = self.backbone.get_intermediate_layers(
                    inp, self.n)
                output = torch.cat([x[:, 0]
                                   for x in intermediate_output], dim=-1)
        output = self.linear_classifier(output.detach())
        if target is None:
            return output
        else:
            loss = self.celoss(output, target)
            return output, loss


def get_options(args, opt_type='train'):
    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.replicationFactor(args.replica)
    if opt_type == 'train':
        opts.deviceIterations(args.di)
        opts.Training.gradientAccumulation(args.ga)
        opts.Training.accumulationAndReplicationReductionType(
            poptorch.ReductionType.Mean)
    else:
        opts.deviceIterations(8)
    return opts


def eval_linear(args):
    opts = get_options(args)
    infer_opts = get_options(args, opt_type='eval')

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_val = datasets.ImageFolder(
        os.path.join(
            args.data_path,
            "validation"),
        transform=val_transform)

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(
        os.path.join(
            args.data_path,
            "train"),
        transform=train_transform)
    train_loader = poptorch.DataLoader(
        options=opts,
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    model = VitLinear(
        args.arch,
        args.patch_size,
        args.num_labels,
        args.n_last_blocks)
    optimizer = SGD(model.linear_classifier.parameters(), lr=args.lr *
                    (args.batch_size *
                     args.ga *
                     args.replica *
                     args.di) /
                    256., momentum=0.9, weight_decay=0,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0)

    start_epoch = 0
    if args.resume:
        weight_path = os.path.join(args.output, 'checkpoint.pth')
        start_epoch = load_checkpoint(model, optimizer, weight_path)
        start_epoch += 1
        print(f'load {weight_path} success, train start at epoch:{start_epoch}')
    else:
        load_pretrain(model, args.pretrained_weights, args.checkpoint_key)
        print(f'load {args.pretrained_weights} success.')

    ipu_model = poptorch.trainingModel(
        model.train(), opts, optimizer=optimizer)
    infer_model = poptorch.inferenceModel(model, infer_opts)
    for epoch in range(start_epoch, args.epochs):
        print(optimizer)
        lr = optimizer.param_groups[0]["lr"]
        train(ipu_model, lr, train_loader, epoch, args)
        scheduler.step()
        ipu_model.setOptimizer(optimizer)

        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(epoch, model, optimizer, args.output)
        if epoch == args.epochs - 1:
            ipu_model.detachFromDevice()
            model.eval()
            acc1 = validate_network(dataset_val, infer_model, infer_opts, args)


def train(model, lr, loader, epoch, args):
    losses = AverageMeter('loss', ':.2f')
    batch_time = AverageMeter('batch', ':.2f')
    data_time = AverageMeter('data', ':.2f')
    throughput = AverageMeter('throughput', ':.0f')
    steps_per_epoch = len(loader)
    log_path = os.path.join(args.output, args.log)
    end = time.time()
    for it, (inp, target) in enumerate(loader):
        current_step = it + epoch * steps_per_epoch
        if args.compile_only:
            model.compile(inp, target)
            sys.exit(0)
        _, loss = model(inp, target)

        loss = torch.mean(loss)
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), inp.size(0))
        tput = inp.size(0) / batch_time.val
        throughput.update(tput)
        info = (f'[{epoch}/{args.epochs}|{it}/{steps_per_epoch}]\t'
                f'train\t'
                f'lr:{lr:.3e}\t'
                f'{losses}\t'
                f'{batch_time}\t'
                f'{data_time}\t'
                f'{throughput}\n')
        with open(log_path, 'a') as fw:
            fw.write(info)
            if it % args.print_freq == 0:
                print(info)


@torch.no_grad()
def validate_network(dataset_val, model, opts, args):
    log_path = os.path.join(args.output, args.log)
    val_loader = poptorch.DataLoader(
        options=opts,
        dataset=dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    losses = AverageMeter('loss', ':.3f')
    acc1_log = AverageMeter('acc1', ':.3f')
    acc5_log = AverageMeter('acc5', ':.3f')
    if model._executable:
        model.attachToDevice()
    for it, (inp, target) in enumerate(val_loader):
        output, loss = model(inp, target)
        loss = torch.mean(loss)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        acc1 = torch.mean(acc1)
        acc5 = torch.mean(acc5)
        batch_size = inp.shape[0]
        losses.update(loss.item(), batch_size)
        acc1_log.update(acc1.item(), batch_size)
        acc5_log.update(acc5.item(), batch_size)

        info = (f'[{it}/{len(val_loader)}]\t'
                f'evaluate\t'
                f'{losses}\t'
                f'{acc1_log}\t'
                f'{acc5_log}\n')
        with open(log_path, 'a') as fw:
            fw.write(info)
            if it % args.print_freq == 0:
                print(info)

    model.detachFromDevice()
    return acc1_log.val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Evaluation with linear classification on ImageNet')
    parser.add_argument(
        '--n_last_blocks',
        default=4,
        type=int,
        help=("Concatenate [CLS] tokens"
              "for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base."))
    parser.add_argument(
        '--arch',
        default='vit_small',
        type=str,
        help='Architecture')
    parser.add_argument(
        '--patch_size',
        default=16,
        type=int,
        help='Patch resolution of the model.')
    parser.add_argument(
        '--pretrained_weights',
        default='',
        type=str,
        help="Path to pretrained weights to evaluate.")
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs of training.')
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help=("Learning rate at the beginning of training (highest LR used during training). The learning"
              "rate is linearly scaled with the batch size, and specified here for a reference batch size"
              "of 256. We recommend tweaking the LR depending on the checkpoint evaluated."))
    parser.add_argument('--batch_size', default=4, type=int, help='batch-size')
    parser.add_argument('--ga', default=8, type=int, help='accumulate number')
    parser.add_argument(
        '--di',
        default=1,
        type=int,
        help='device iteration number')
    parser.add_argument(
        '--replica',
        default=1,
        type=int,
        help='replica number')
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--log', default='linear.log', type=str)
    parser.add_argument(
        '--num_workers',
        default=32,
        type=int,
        help='Number of data loading workers per GPU.')
    parser.add_argument(
        '--save_freq',
        default=10,
        type=int,
        help="Epoch frequency for save.")
    parser.add_argument(
        '--print_freq',
        default=10,
        type=int,
        help='Save log every x steps.')
    parser.add_argument(
        '--output',
        default="linear",
        help='Path to save logs and checkpoints')
    parser.add_argument(
        '--num_labels',
        default=1000,
        type=int,
        help='Number of labels for linear classifier')
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
