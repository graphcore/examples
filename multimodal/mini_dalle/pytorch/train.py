# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 lucidrains

# This file has been modified by Graphcore


import sys
from pathlib import Path
import datetime
import time
from glob import glob
import os
from functools import partial
from log import Logger
import torch
import poptorch
import popart
import wandb  # Quit early if user doesn't have wandb installed.
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from models.dalle import default
from models import VQGanVAE, WrappedDALLE
from models.loader import get_data
from models.tokenizer import SimpleTokenizer, YttmTokenizer
from models.optimization import get_optimizer
from models.optimization import get_lr_sched
from args import parse_args, sync_metrics
from ipu_options import get_options


# helpers


def exists(val):
    return val is not None


def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


def main(args):
    if not args.generated_data and not args.synthetic_data:
        assert Path(args.input_folder).exists(), f'The path {args.input_folder} was not found.'

    abs_pathd = os.path.abspath(args.checkpoint_output_dir)
    os.makedirs(abs_pathd, exist_ok=True)
    log = Logger(abs_pathd+"/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log',
                 level='INFO')

    # vocab size

    if exists(args.bpe_path):
        klass = YttmTokenizer
        tokenizer = klass(args.bpe_path)
    else:
        tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    del tokenizer

    # reconstitute vae
    if exists(args.pretrained_checkpoint):
        dalle_path = Path(args.pretrained_checkpoint)

        assert dalle_path.exists(), 'DALL-E model file does not exist'
        loaded_obj = torch.load(str(dalle_path), map_location='cpu')

        dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
        opt_state = loaded_obj.get('opt_state')
        scheduler_state = loaded_obj.get('scheduler_state')

        vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path)

        dalle_params = dict(
            **dalle_params
        )
        resume_epoch = loaded_obj.get('epoch', 0)
    else:
        print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path)

        dalle_params = dict(
            num_text_tokens=vocab_size,
            text_seq_len=args.text_seq_len,
            dim=args.hidden_size,
            depth=args.num_hidden_layers,
            heads=args.num_attention_heads,
            dim_head=args.dim_head,
            loss_img_weight=args.loss_img_weight,
            attn_types=tuple(args.attn_types.split(',')),
            ff_dropout=args.ff_dropout,
            attn_dropout=args.attn_dropout,
            sandwich_norm=args.sandwich_norm,
            embedding_ipu_id=args.embedding_ipu_id,
            embedding_serialization_factor=args.embedding_serialization_factor,
            layers_per_ipu=args.layers_per_ipu,
            cls_ipu_id=args.cls_ipu_id,
            fp16=args.fp16,
            byteio=args.byteio
        )
        resume_epoch = 0

    # Execution parameters
    opts = get_options(args)

    # Dataloader
    dl = get_data(args, opts, vae.image_size, train=True, async_dataloader=args.async_dataloader)
    steps_per_epoch = len(dl)

    # initialize DALL-E

    dalle = WrappedDALLE(vae=vae, **dalle_params)

    # if using fp16:
    if args.fp16:
        dalle = dalle.half()

    if exists(args.pretrained_checkpoint):
        dalle.load_state_dict(weights)

    # optimizer
    opt = get_optimizer(args, dalle)
    if exists(args.pretrained_checkpoint) and opt_state:
        opt.load_state_dict(opt_state)
    poptorch_dalle = poptorch.trainingModel(dalle,
                                            options=opts,
                                            optimizer=opt)
    if args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
    else:
        lr_lambda = partial(get_lr_sched, scheduler=args.lr_scheduler,
                            num_train_steps=args.epochs,
                            warmup_ratio=0.2)
        scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    if exists(args.pretrained_checkpoint) and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    # experiment tracker

    model_config = dict(
        depth=args.num_hidden_layers,
        heads=args.num_attention_heads,
        dim_head=args.dim_head
    )

    if args.wandb and (not args.use_popdist or args.popdist_rank == 0):
        run = wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            entity=None,
            resume=False,
            config=model_config,
            settings=wandb.Settings(console='off')
        )


    def save_model(path, epoch=0):
        if not path:
            return

        save_obj = {
            'hparams': dalle_params,
            'vae_params': vae_params,
            'epoch': epoch,
        }

        save_obj = {
            **save_obj,
            'weights': dalle.state_dict(),
            'opt_state': opt.state_dict(),
        }
        save_obj['scheduler_state'] = (scheduler.state_dict() if scheduler else None)
        filename = f"dalle_{epoch}.pt"
        save_path = os.path.join(path, filename)
        torch.save(save_obj, save_path)

    # Compile model
    log.logger.info("---------- Compilation Started ---------")
    start_compile = time.perf_counter()
    text, images = next(iter(dl))
    poptorch_dalle.compile(text, images)
    duration_compilation = time.perf_counter() - start_compile
    log.logger.info(f"Compiled model in {duration_compilation} secs")
    log.logger.info("---------------------------------------")

    # Exit here if compile only mode is enabled
    if args.compile_only:
        log.logger.info("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
        sys.exit(0)

    # Training loop
    log.logger.info("---------- Training Started -----------")

    save_model(args.checkpoint_output_dir, epoch=resume_epoch)
    global_batch_size = args.batch_size * args.gradient_accumulation * args.replication_factor
    samples_per_step = global_batch_size * args.device_iterations
    training_steps = args.epochs * steps_per_epoch

    # Track approx. IPU compute time
    total_compute_time = 0
    # Track total train time
    start_train = time.perf_counter()
    for epoch in range(resume_epoch, args.epochs):
        for i, (text, images) in enumerate(dl):
            current_step = i + epoch * steps_per_epoch

            start_step = time.perf_counter()
            loss = poptorch_dalle(text, images)
            step_length = sync_metrics(time.perf_counter() - start_step)
            mean_loss = sync_metrics(loss.mean().item())
            if epoch > 0 or i > 0:  # The throughput of the first step is unstable
                total_compute_time += step_length

            if not args.use_popdist or args.popdist_rank == 0:
                num_instances = args.popdist_size if args.use_popdist else 1
                step_throughput = samples_per_step * num_instances / step_length
                msg = ("Epoch: {:.2f}/{} "
                       "Step: {}/{} "
                       "Lr: {:.6f} "
                       "loss: {:.3f} "
                       "throughput: {:.2f} samples/sec"
                       ).format(epoch, args.epochs,
                                current_step, training_steps,
                                opt.param_groups[0]['lr'],
                                mean_loss,
                                step_throughput)
                log.logger.info(msg)
                if args.wandb and (not args.use_popdist or args.popdist_rank == 0):
                    wandb.log({"LR": opt.param_groups[0]['lr'],
                               "Throughput": step_throughput,
                               "Loss": mean_loss})

            start_step = time.perf_counter()
            if i != 0 and i % args.checkpoint_save_steps == 0:
                save_model(args.checkpoint_output_dir, epoch=epoch+1)

        if args.lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(mean_loss)
        else:
            scheduler.step()
        poptorch_dalle.setOptimizer(opt)

        save_model(args.checkpoint_output_dir, epoch=epoch+1)

    if args.wandb and (not args.use_popdist or args.popdist_rank == 0):
        wandb.finish()

    stop_train = time.perf_counter()
    if not args.use_popdist or args.popdist_rank == 0:
        log.logger.info("---------------------------------------")

        log.logger.info("---------- Training Metrics -----------")
        log.logger.info(f"global_batch_size: {global_batch_size}")
        log.logger.info(f"device_iterations: {args.device_iterations}")
        log.logger.info(f"training_steps: {training_steps}")
        duration_run = stop_train - start_train
        num_samples = samples_per_step * num_instances * (training_steps-1)
        overall_throughput = num_samples / total_compute_time
        log.logger.info(f"Training time: {duration_run:.3f} secs")
        log.logger.info("throughput: {:5f} samples/sec.".format(overall_throughput))
        log.logger.info("---------------------------------------")

if __name__ == "__main__":
    # argument parsing
    args = parse_args()

    torch.manual_seed(args.random_seed)
    main(args)
