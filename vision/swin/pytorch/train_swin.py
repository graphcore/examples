# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from config import get_config
from models.build import build_pipeline as build_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optimizer import build_optimizer
from utils import AverageMeter, get_lr_scheduler, load_pretrained
from options import get_options
import poptorch
import pdb
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from dataset.build_ipu import build_loader
from lr_scheduler import build_scheduler
import sys
from timm.models import resume_checkpoint
import wandb


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script',
        add_help=False)
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        metavar="FILE",
        help='path to config file',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        metavar="FILE",
        help='path to dataset')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        metavar="FILE",
        help='path to save output files')
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default=None,
        help='path to init checkpoint when fine tune models')
    parser.add_argument(
        '--batch-size',
        type=int,
        help="batch size for single GPU")
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help="batch size for single GPU")
    parser.add_argument('--weights', type=str, help='weights for model')
    parser.add_argument(
        '--device',
        type=str,
        default='',
        choices=[
            'cpu',
            'ipu',
            'gpu'])
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--wandb', action='store_true', help="Add Weights & Biases logging")
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs")
    parser.add_argument('--training-steps', type=int, help="Number of training steps")
    args, unparsed = parser.parse_known_args()
    if args.training_steps is not None:
        args.epochs = 1

    config = get_config(args)
    return args, config


def load_weigths(path, model):
    new_state_dict = OrderedDict()
    weights = torch.load(path)
    encoder_keys = []
    for k, v in weights.items():
        if k.startswith('siamese_encoder') and 'encoder_k' not in k:
            encoder_keys.append(k)

    for key_swin, key_moby_encoder in zip(
            model.state_dict().keys(), encoder_keys):
        value = weights[key_moby_encoder]
        new_state_dict[key_swin] = value
    model.load_state_dict(new_state_dict)


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, loss):
    save_state = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    if not os.path.exists(os.path.join(config.OUTPUT)):
        os.makedirs(os.path.join(config.OUTPUT))
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}_loss_{str(loss)}.pth')
    torch.save(save_state, save_path)


def get_random_datum(config):
    result = []
    batch_size = config.DATA.BATCH_SIZE * config.IPU.NUM_REPLICAS * config.IPU.GRADIENT_ACCUMULATION_STEPS
    if config.PRECISION[0] == 'half':
        use_half = True
    else:
        use_half = False

    dataset = GeneratedDataset(shape=[3, config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[0]],  # 1024 for global batch
                               half_precision=use_half)  # use_half)
    data = (dataset[i] for i in range(batch_size))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """

    def __init__(self, shape, size=60000, half_precision=True):  # use_half
        self.size = size
        self.half_precision = half_precision
        self.data_shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        synthetic_data = torch.rand(self.data_shape)
        synthetic_label = torch.randint(0, 2, [1], dtype=torch.long)
        if self.half_precision:
            synthetic_data = synthetic_data.half()
        return synthetic_data, synthetic_label


def compile_model(poptorch_model, mixup_fn, log_path, config):
    datum = get_random_datum(config)
    (pre_input, pre_label) = datum
    if mixup_fn is not None:
        pre_input, pre_label = mixup_fn(pre_input, pre_label)
    poptorch_model.compile(pre_input, pre_label)
    info = (f'Compiled model Finish\n'
            f'---------------------------------------\n')
    with open(log_path, 'a') as fw:
        fw.write(info)


def train(args, opts, config):
    if not os.path.exists(os.path.join(config.OUTPUT)):
        os.makedirs(os.path.join(config.OUTPUT))
    log_path = os.path.join(config.OUTPUT, 'logs.log')
    if args.wandb:
        wandb.init(
            project='swin_pretrain',
            settings=wandb.Settings(
                console='off'))
    dataset_train, data_loader_train, mixup_fn = build_loader(
        config, opts)  # dataset_val,data_loader_val,
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    model = build_model(config=config, train_loss_fn=criterion)
    info = (
        f'use data type:{config.PRECISION[0]},use model type:{config.PRECISION[1]}')
    with open(log_path, 'a') as fw:
        fw.write(info)
    if config.PRECISION[1] == 'half':
        model.half()
    optimizer = build_optimizer(config, model)
    model = poptorch.trainingModel(model.train(), opts, optimizer=optimizer)
    compile_model(model, mixup_fn, log_path, config)
    resume_epoch = None
    resume_opt = True
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=optimizer if resume_opt else None
        )
    else:
        if config.PRETRAINED:
            load_pretrained(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    losses = AverageMeter('loss', ':.2f')
    num_steps = len(data_loader_train)
    num_epochs = min(args.epochs, config.TRAIN.EPOCHS)
    for epoch in range(start_epoch, num_epochs):
        record_loss = 0
        tput = 0
        for step, (data, targets) in enumerate(data_loader_train):
            if mixup_fn is not None:
                data, targets = mixup_fn(data, targets)
            if config.PRECISION[0] == 'half':
                data = data.half()
            s0 = time.time()
            model.setOptimizer(lr_scheduler.optimizer)
            _, loss = model(data, targets)
            s1 = time.time()
            mean_loss = loss.mean().item()
            losses.update(mean_loss, data.size(0))
            tput = tput + data.size(0) / (s1 - s0)

            record_loss += mean_loss

            lr_scheduler.step_update(epoch * num_steps + step)
            if step % 50 == 0:
                lrl = [param_group['lr']
                       for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                throughput = tput/(step+1)
                train_info = (
                    f'------------------epoch {epoch}: {step} / {num_steps}-----------------\n'
                    f'lr: {lr}\n'
                    f'loss: {losses.avg:0.4f}\n'
                    f'throughput: {throughput:0.4f}\n'
                )
                print(train_info)
                sys.stdout.flush()
                with open(log_path, 'a') as fw:
                    fw.write(train_info)

                if args.wandb:
                    wandb.log({
                        "Throughput": tput / (step + 1),
                        "Loss": record_loss / (step + 1),
                        "LR": lr})

            if args.training_steps is not None and step >= args.training_steps:
                break

        save_checkpoint(
            config,
            epoch,
            model,
            optimizer,
            lr_scheduler,
            record_loss /
            len(data_loader_train))


def main():
    args, config = parse_option()
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    opts = get_options(config)
    train(args, opts, config)


if __name__ == '__main__':
    main()
