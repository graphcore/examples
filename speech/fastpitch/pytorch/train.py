# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# This file has been modified by Graphcore Ltd.
# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import json
import glob
import os
import re
import time
from collections import defaultdict, OrderedDict
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import common.tb_dllogger as logger
import data_functions
import loss_functions
import models
import time
import wandb
import poptorch
from pipeline_base_model import BasePipelineModel
import sys
import popdist
import popdist.poptorch
import horovod.torch as hvd


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--pyprof', action='store_true', help='Enable pyprof profiling')
    parser.add_argument('--wandb', type=bool, default=False, help='Enable wandb')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, required=True,
                          help='Number of total epochs to run')
    training.add_argument('--epochs-per-ckpt', type=int, default=50,
                          help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path', type=str, default=None,
                          help='Checkpoint path to resume training')
    training.add_argument('--resume', action='store_true',
                          help='Resume training from the last available checkpoint')
    training.add_argument('--seed', type=int, default=1234,
                          help='Seed for PyTorch random number generators')
    training.add_argument('--amp', action='store_true',
                          help='Enable AMP')
    training.add_argument('--cuda', action='store_true',
                          help='Run on GPU using CUDA')
    training.add_argument('--cudnn-benchmark', action='store_true',
                          help='Enable cudnn benchmark mode')
    training.add_argument('--ema-decay', type=float, default=0,
                          help='Discounting factor for training weights EMA')
    training.add_argument('--gradient-accumulation', type=int, default=1,
                          help='Training steps to accumulate gradients for')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--optimizer', type=str, default='lamb',
                              help='Optimization algorithm')
    optimization.add_argument('-lr', '--learning-rate', type=float, required=True,
                              help='Learing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float,
                              help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                              help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size', type=int, required=True,
                              help='Batch size per GPU')
    optimization.add_argument('--warmup-steps', type=int, default=1000,
                              help='Number of steps for lr warmup')
    optimization.add_argument('--dur-predictor-loss-scale', type=float,
                              default=1.0, help='Rescale duration predictor loss')
    optimization.add_argument('--pitch-predictor-loss-scale', type=float,
                              default=1.0, help='Rescale pitch predictor loss')

    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--validation-files', type=str,
                         help='Path to validation filelist. Separate multiple paths with commas.')
    dataset.add_argument('--pitch-mean-std-file', type=str, default=None,
                         help='Path to pitch stats to be stored in the model')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['english_cleaners'], type=str,
                         help='Type of text cleaners for input text')
    dataset.add_argument('--symbol-set', type=str, default='english_basic',
                         help='Define symbol set for input text')

    data_type = parser.add_mutually_exclusive_group(required=True)
    data_type.add_argument('--training-files', type=str,
                           help='Path to training filelist. Separate multiple paths with commas.')
    data_type.add_argument('--generated-data', action="store_true",
                           help="Whether or not to use generated data instead of real data.")

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Condition on speaker, value > 1 enables trainable speaker embeddings.')

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                             help='Rank of the process for multiproc. Do not set manually.')
    distributed.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                             help='Number of processes for multiproc. Do not set manually.')
    distributed.add_argument('--replication-factor', type=int, default=1,
                             help='Number of replicas to create for training.')
    distributed.add_argument('--num-dataloader-workers', type=int, default=8,
                             help='Number of workers to use for the dataloader.')
    distributed.add_argument('--optimizer-state-offchip', type=bool, default=False)
    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(backend=('nccl' if args.cuda else 'gloo'),
                            init_method='env://')
    print("Done initializing distributed training")


def last_checkpoint(output):

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            print(f'WARNING: Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/FastPitch_checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None


def save_checkpoint(local_rank, model, ema_model, optimizer, epoch, total_iter,
                    config, amp_run, filepath):
    if local_rank != 0:
        return

    print(f"Saving model and optimizer state at epoch {epoch} to {filepath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    if amp_run:
        checkpoint['amp'] = amp.state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(local_rank, model, ema_model, optimizer, epoch, total_iter,
                    config, amp_run, filepath, world_size):
    if local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']
    config = checkpoint['config']

    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if amp_run:
        amp.load_state_dict(checkpoint['amp'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])


def validate(model, epoch, total_iter, criterion, valset, batch_size,
             collate_fn, distributed_run, batch_to_gpu, use_gt_durations=False,
             ema=False):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=8, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x, use_gt_durations=use_gt_durations)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik

    logger.log((epoch,) if epoch is not None else (),
               tb_total_steps=total_iter,
               subset='val_ema' if ema else 'val',
               data=OrderedDict([
                   ('loss', val_meta['loss'].item()),
                   ('mel_loss', val_meta['mel_loss'].item()),
                   ('frames/s', num_frames.item() / val_meta['took']),
                   ('took', val_meta['took'])]),)

    if was_training:
        model.train()
    return val_meta


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def apply_ema_decay(model, ema_model, decay):
    if not decay:
        return
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k, v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])


def apply_multi_tensor_ema(model_weight_list, ema_model_weight_list, decay, overflow_buf):
    if not decay:
        return
    amp_C.multi_tensor_axpby(65536, overflow_buf, [ema_model_weight_list, model_weight_list, ema_model_weight_list], decay, 1-decay, -1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    if args.training_files and not args.validation_files:
        print("Both '--training-files' and '--validation-files' must be "
              "provided if either one of them is provided. Exiting.")
        sys.exit(1)

    if args.local_rank == 0:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

    log_fpath = args.log_file or os.path.join(args.checkpoint_dir, 'nvlog.json')
    tb_subsets = ['train', 'val']
    if args.ema_decay > 0.0:
        tb_subsets.append('val_ema')

    logger.init(log_fpath, args.checkpoint_dir, enabled=(args.local_rank == 0),
                tb_subsets=tb_subsets)
    logger.parameters(vars(args), tb_subset='train')

    parser = models.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    device = torch.device('cpu')

    if args.generated_data:
        pitch_mean = 0
        pitch_std = 1
    else:
        # Store pitch mean/std as params to translate from Hz during inference
        with open(args.pitch_mean_std_file, 'r') as f:
            stats = json.load(f)
            pitch_mean = stats['mean']
            pitch_std = stats['std']

    model_config = models.get_model_config('FastPitch', args)
    model = models.get_model('FastPitch', model_config, device)
    # ====== IPU related start====== #

    class Model(BasePipelineModel):
        def __init__(self):
            super().__init__()
            self.model = model
            self.model.pitch_mean[0] = pitch_mean
            self.model.pitch_std[0] = pitch_std
            self.criterion = loss_functions.get_loss_function(
                'FastPitch', dur_predictor_loss_scale=args.dur_predictor_loss_scale,
                pitch_predictor_loss_scale=args.pitch_predictor_loss_scale)

        def forward(self, text_padded, mel_padded, dur_padded, pitch_padded, dur_lens):

            y_pred = self.model([text_padded, mel_padded, dur_padded, pitch_padded], use_gt_durations=True)
            loss = self.criterion(y_pred, [mel_padded, dur_padded, dur_lens, pitch_padded])
            return loss

    model = Model()
    print(model)
    model.set_start_point_list([('model.decoder.layers__0', 1), ])
    # ====== IPU related end====== #

    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = poptorch.optim.Adam(model.parameters(), **kw)
    elif args.optimizer == 'lamb':
        optimizer = poptorch.optim.LAMB(model.parameters(), accum_type=torch.float32, max_weight_norm=1000, bias_correction=True, **kw)
    else:
        raise ValueError

    start_epoch = [1]
    start_iter = [0]
    start_epoch = start_epoch[0]
    total_iter = start_iter[0]

    # cached data in the samplest way
    from tqdm import tqdm
    from dataset import CachedDataset, GenCachedDataset

    collate_fn = data_functions.get_collate_function('FastPitch')
    if args.generated_data:
        trainset = GenCachedDataset()
    else:
        trainset = data_functions.get_data_loader(
            'FastPitch',
            audiopaths_and_text=args.training_files,
            **vars(args))

        cached_data_folder = 'cached_data'
        if not os.path.exists(cached_data_folder):
            os.mkdir(cached_data_folder)
            for i, v in tqdm(enumerate(trainset)):
                path = os.path.join(cached_data_folder, str(i))
                sample_numpy = []
                for j in v:
                    try: tensor = j.numpy()
                    except: tensor = j
                    sample_numpy.append(tensor)
                np.save(path, sample_numpy)

        trainset = CachedDataset()

    # done cache data
    # ====== IPU related start====== #
    from ipu_config import build_ipu_config
    ipu_config = build_ipu_config(args, seed=args.seed, gradient_accmulation=args.gradient_accumulation)

    # Adjust batch size according to IPU configs
    batch_size = args.batch_size * args.gradient_accumulation * args.replication_factor

    train_loader = poptorch.DataLoader(
        options=ipu_config, dataset=trainset, num_workers=args.num_dataloader_workers, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print('len data loader', len(train_loader))
    model = poptorch.trainingModel(model=model, options=ipu_config, optimizer=optimizer)
    # ====== IPU related end====== #
    model.train()
    if args.wandb:
        wandb.init('fastpitch')
    num_iters = len(train_loader)
    samples_per_step = train_loader._combined_batch_size

    def cycle(iterator):
        """
        Loop `iterator` forever
        """
        while True:
            for item in iterator:
                yield item

    for epoch in range(start_epoch, args.epochs + 1):
        c_train_loader = cycle(train_loader)
        print('samples_per_step', samples_per_step)
        for i in range(num_iters):
            adjust_learning_rate(total_iter, optimizer, args.learning_rate, args.warmup_steps)
            model.setOptimizer(optimizer)
            start = time.time()
            text_padded, mel_padded, dur_padded, pitch_padded, dur_lens, output_lengths = next(c_train_loader)
            data_end = time.time()
            num_frames = torch.sum(output_lengths).item()
            loss = model(text_padded, mel_padded, dur_padded, pitch_padded, dur_lens)
            end = time.time()
            total_iter += 1
            lr = optimizer.param_groups[0]['lr']
            if i % 5 == 0:
                if args.wandb:
                    wandb.log(dict(
                        loss=loss.mean().item(), epoch=epoch, global_step=total_iter, lr=lr,
                        num_frame_per_sec=num_frames/(end-data_end), all_nf_per_sec=870 * samples_per_step/(end-data_end)))
                print(
                    'lr', lr, 'epoch', epoch, 'step', total_iter, 'num_frame', num_frames,
                    'data time', data_end - start, 'num_frame/sec', num_frames/(end-data_end),
                    'all nf/s', 870 * samples_per_step/(end-data_end))
                # the num frame / sec is the validate frames processed
                # the all num frame /sec is the frames process that contains padding positions

                if args.generated_data:
                    loss = torch.zeros(1)
                # Standardised metrics outputs
                print(f"loss: {loss.mean().item()},")
                print(f"throughput: {samples_per_step/(end-start)} samples/sec,")
            else:
                print('step: ', i)


if __name__ == '__main__':
    main()
