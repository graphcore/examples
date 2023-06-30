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
import popdist
import horovod.torch as hvd
import os
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from config import get_config
from models.build import build_pipeline as build_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from optimizer import build_optimizer
from utils import AverageMeter, get_lr_scheduler, load_pretrained, CustomOpsNotFoundException
from options import get_options
import poptorch
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from dataset.build_ipu import build_loader
from lr_scheduler import build_scheduler
from timm.models import resume_checkpoint
import wandb
import ctypes
import logging

threads = 4
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
torch.set_num_threads(threads)


def get_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab


def parse_option():
    parser = argparse.ArgumentParser("Swin Transformer training and evaluation script", add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--data-path", type=str, required=True, metavar="FILE", help="path to dataset")
    parser.add_argument(
        "--checkpoint-output-dir", type=str, required=True, metavar="FILE", help="path to save output files"
    )
    parser.add_argument("--pretrained-model", type=str, help="path to init checkpoint when fine tune models")
    parser.add_argument("--batch-size", type=int, help="batch size for single replica")
    parser.add_argument("--num-workers", type=int, help="worker size for single instance")
    parser.add_argument("--weights", type=str, help="weights for model")
    parser.add_argument("--device", type=str, choices=["cpu", "ipu", "gpu"])
    parser.add_argument("--alignment", action="store_true", help="if alignment fwd or bwd")
    parser.add_argument("--half", action="store_true", help="use half")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Resume full model and optimizer state from checkpoint (default: none)",
    )
    parser.add_argument("--compile-only", action="store_true", help="Compile only")
    parser.add_argument("--ga", type=int, help="Gradient Accumulations Steps")
    parser.add_argument("--amp", type=float, help="Available memory proportion")
    parser.add_argument("--rts", action="store_true", help="Replicated tensor sharding")
    parser.add_argument("--precision", type=str, choices=["half", "float"])
    parser.add_argument("--wandb", action="store_true", help="Add Weights & Biases logging")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--training-steps", type=int, help="Number of training steps")
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
        if k.startswith("siamese_encoder") and "encoder_k" not in k:
            encoder_keys.append(k)

    for key_swin, key_moby_encoder in zip(model.state_dict().keys(), encoder_keys):
        value = weights[key_moby_encoder]
        new_state_dict[key_swin] = value
    model.load_state_dict(new_state_dict)


def save_checkpoint(config, epoch, model, optimizer, lr_scheduler, loss):
    save_state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "config": config,
    }
    if not os.path.exists(os.path.join(config.CHECKPOINT_OUTPUT_DIR)):
        os.makedirs(os.path.join(config.CHECKPOINT_OUTPUT_DIR))
    save_path = os.path.join(config.CHECKPOINT_OUTPUT_DIR, f"ckpt_epoch_{epoch}_loss_{str(loss)}.pth")
    torch.save(save_state, save_path)


def get_random_datum(config):
    result = []
    batch_size = config.DATA.BATCH_SIZE * config.IPU.NUM_LOCALREPLICA * config.IPU.GRADIENT_ACCUMULATION_STEPS
    if config.PRECISION[0] == "half":
        use_half = True
    else:
        use_half = False
    dataset = GeneratedDataset(
        shape=[3, config.DATA.IMG_SIZE[0], config.DATA.IMG_SIZE[0]], size=batch_size, half_precision=use_half
    )
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


def compile_model(poptorch_model, config):
    from dataset.ipu_mixup import Mixup

    datum = get_random_datum(config)
    (pre_input, pre_label) = datum
    mixup_fn = Mixup(
        mixup_alpha=config.AUG.MIXUP,
        cutmix_alpha=config.AUG.CUTMIX,
        cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        prob=config.AUG.MIXUP_PROB,
        switch_prob=config.AUG.MIXUP_SWITCH_PROB,
        mode=config.AUG.MIXUP_MODE,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES,
    )
    pre_input, pre_label = mixup_fn(pre_input, pre_label)
    poptorch_model.compile(pre_input, pre_label)


def train(args, opts, config):
    log_path = os.path.join(config.CHECKPOINT_OUTPUT_DIR, "logs.log")
    try:
        if not os.path.exists(os.path.join(config.CHECKPOINT_OUTPUT_DIR)):
            os.makedirs(os.path.join(config.CHECKPOINT_OUTPUT_DIR))
    except BaseException:
        logging.info("CHECKPOINT_OUTPUT_DIR dir already exists")
    logger = get_logger(log_path)
    train_start_time = datetime.datetime.now()

    if args.wandb:
        wandb.init(project="torch-swin", settings=wandb.Settings(console="off"))
    if config.AUG.MIXUP > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    model = build_model(config=config, train_loss_fn=criterion)
    logger.info(f"use data type:{config.PRECISION[0]}, use model type:{config.PRECISION[1]}\n")
    if config.PRECISION[1] == "half":
        model.half()
    optimizer = build_optimizer(config, model)
    model = poptorch.trainingModel(model.train(), opts, optimizer=optimizer)
    compile_model(model, config)
    if args.compile_only:
        logger.info("Compilation done!")
        exit()
    resume_epoch = None
    resume_opt = True
    if args.resume:
        resume_epoch = resume_checkpoint(model, args.resume, optimizer=optimizer if resume_opt else None)
    else:
        if config.PRETRAINED:
            load_pretrained(config, model)
    dataset_train, data_loader_train, mixup_fn = build_loader(config, opts)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    losses = AverageMeter("loss", ":.2f")
    num_steps = len(data_loader_train)
    epoch_start_time = time.time()
    num_epochs = min(args.epochs, config.TRAIN.EPOCHS)
    for epoch in range(start_epoch, num_epochs):
        record_loss = 0
        loop_start_time = time.time()
        for step, (data, targets) in enumerate(data_loader_train):
            optimizer_start_time = time.time()
            model.setOptimizer(lr_scheduler.optimizer)
            _, loss = model(data, targets)
            model_end_time = time.time()

            mean_loss = loss.mean().item()
            losses.update(mean_loss, data.size(0))
            record_loss += mean_loss
            model_execution_time = model_end_time - optimizer_start_time
            model_tput = data.size(0) / model_execution_time

            lr_scheduler.step_update(epoch * num_steps + step)
            if step % 1 == 0:
                lrl = [param_group["lr"] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                if args.wandb:
                    wandb.log({"Throughput": model_tput / (step + 1), "Loss": record_loss / (step + 1), "LR": lr})

            if step == 0:
                batch_time = time.time() - loop_start_time
            else:
                batch_time = time.time() - batch_end_time
            current_batch_tput = data.size(0) / batch_time

            if not popdist.isPopdistEnvSet():
                logger.info(
                    f"TRAIN: epoch[{epoch}/{config.TRAIN.EPOCHS}] step[{step}/{len(data_loader_train)}]    model_execution_time:{model_execution_time}    throughput:{model_tput:.4f} samples/sec    batch_time:{batch_time}    batch_tput:{current_batch_tput}   loss:{losses.avg:.6f}\n"
                )
            elif popdist.isPopdistEnvSet() and (config.popdist_rank == 0):
                logger.info(
                    f"RANK_0 TRAIN: epoch[{epoch}/{config.TRAIN.EPOCHS}] step[{step}/{len(data_loader_train)}]     model_execution_time:{model_execution_time}    throughput:{config.popdist_size*model_tput:.4f} samples/sec    batch_time:{batch_time}    batch_tput:{config.popdist_size*current_batch_tput:.4f}   loss:{losses.avg:.6f}\n"
                )
            batch_end_time = time.time()

            if args.training_steps is not None and step >= args.training_steps:
                break

        epoch_end_time = time.time()
        if not popdist.isPopdistEnvSet():
            logger.info(f"EPOCH {epoch} TIME: {epoch_end_time-epoch_start_time}\n")
        elif popdist.isPopdistEnvSet() and (config.popdist_rank == 0):
            logger.info(f"RANK_0 EPOCH {epoch} TIME: {epoch_end_time-epoch_start_time}\n")

        if epoch == config.TRAIN.EPOCHS - 1:
            save_start_time = time.time()
            save_checkpoint(config, epoch, model, optimizer, lr_scheduler, record_loss / len(data_loader_train))
            save_end_time = time.time()
            if not popdist.isPopdistEnvSet():
                logger.info(f"SAVE_MODEL TIME: {save_end_time-save_start_time}\n")
            elif popdist.isPopdistEnvSet() and (config.popdist_rank == 0):
                logger.info(f"RANK_0 SAVE_MODEL TIME: {save_end_time-save_start_time}\n")

    train_end_time = datetime.datetime.now()
    if not popdist.isPopdistEnvSet():
        epoch_time_info = f"TOTAL TRAIN TIME: {str(train_end_time-train_start_time)}\n"
        logger.info(epoch_time_info)
    elif popdist.isPopdistEnvSet() and (config.popdist_rank == 0):
        epoch_time_info = f"RANK_0 TOTAL TRAIN TIME: {str(train_end_time-train_start_time)}\n"
        logger.info(epoch_time_info)


def init_popdist(config):
    hvd.init()
    if popdist.getNumTotalReplicas() != config.IPU.NUM_LOCALREPLICA:
        print(f"The number of replicas is overridden by PopRun. " f"The new value is {popdist.getNumTotalReplicas()}.")
    config.IPU.NUM_LOCALREPLICA = popdist.getNumLocalReplicas()
    config.popdist_rank = popdist.getInstanceIndex()
    config.popdist_size = popdist.getNumInstances()
    hvd.broadcast(torch.Tensor([config.SEED]), root_rank=0)


def main():
    args, config = parse_option()
    config.defrost()
    if popdist.isPopdistEnvSet():
        init_popdist(config)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    opts = get_options(config)

    global_batch_size = config.DATA.BATCH_SIZE * config.IPU.NUM_LOCALREPLICA * config.IPU.GRADIENT_ACCUMULATION_STEPS
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR * (global_batch_size / 1024)
    config.TRAIN.WARMUP_LR = config.TRAIN.WARMUP_LR * (global_batch_size / 1024)
    config.TRAIN.MIN_LR = config.TRAIN.MIN_LR * (global_batch_size / 1024)
    config.freeze()
    train(args, opts, config)


if __name__ == "__main__":
    try:
        ctypes.cdll.LoadLibrary("./custom_ops.so")
    except:
        raise CustomOpsNotFoundException(
            "`./custom_ops.so` not found. Please run `make` in this application's root directory first."
        )

    main()
