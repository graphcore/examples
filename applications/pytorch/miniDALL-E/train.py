# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2021 lucidrains

# This file has been modified by Graphcore


import argparse
from pathlib import Path
import datetime
import time
from glob import glob
import os
import shutil
from log import Logger
import torch
import poptorch
import popart
import wandb  # Quit early if user doesn't have wandb installed.
from poptorch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.dalle import default
from models import VQGanVAE, WrappedDALLE
from models.loader import TextImageDataset
from models.tokenizer import SimpleTokenizer, YttmTokenizer
from args import parse_args


# helpers


def exists(val):
    return val is not None


def get_trainable_params(model, weight_decay=0):
    # Do not apply weight_decay for one-dimensional parameters
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": weight_decay},
        {"params": non_regularized_params, "weight_decay": 0}
    ]

    return params


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
    if not args.synthetic_data:
        assert Path(args.input_folder).exists(), f'The path {args.input_folder} was not found.'

    abs_pathd = os.path.abspath(args.checkpoint_output_dir)
    os.makedirs(abs_pathd, exist_ok=True)
    log = Logger(abs_pathd+"/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log',
                 level='INFO')

    # tokenizer

    if exists(args.bpe_path):
        klass = YttmTokenizer
        tokenizer = klass(args.bpe_path)
    else:
        tokenizer = SimpleTokenizer()

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
            num_text_tokens=tokenizer.vocab_size,
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
            fp16=args.fp16
        )
        resume_epoch = 0


    # create dataset and dataloader

    ds = TextImageDataset(
        args.input_folder,
        text_len=args.text_seq_len,
        image_size=vae.image_size,
        resize_ratio=1.0,
        truncate_captions=args.truncate_captions,
        tokenizer=tokenizer,
        shuffle=True,
        synthetic=args.synthetic_data,
        fp16=args.fp16
    )

    assert len(ds) > 0, 'dataset is empty'
    print(f'{len(ds)} image-text pairs found for training')


    opts = poptorch.Options()
    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(args.batches_per_step)
    opts.replicationFactor(args.replication_factor)
    opts.Training.gradientAccumulation(args.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    opts.Precision.enableStochasticRounding(args.stochastic_rounding)
    opts.anchorMode(poptorch.AnchorMode.Final)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(True))

    if args.enable_rts:
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useReplicatedTensorSharding(True).minElementsForReplicatedTensorSharding(args.replication_factor))

    opts.randomSeed(args.random_seed)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    mem_prop = {
        f'IPU{i}': args.matmul_proportion[i]
        for i in range(args.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)

    # PopART options
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("outlineThreshold", 10.0)

    if args.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)
    else:
        opts.Precision.setPartialsType(torch.float32)

    dl = poptorch.DataLoader(options=opts, dataset=ds, batch_size=args.batch_size, num_workers=args.dataloader_workers,
                             persistent_workers=True, shuffle=True, drop_last=True, sampler=None)
    steps_per_epoch = len(dl)

    # initialize DALL-E

    dalle = WrappedDALLE(vae=vae, **dalle_params)

    # if using fp16:
    if args.fp16:
        dalle = dalle.half()

    if exists(args.pretrained_checkpoint):
        dalle.load_state_dict(weights)

    # optimizer
    first_order_type = torch.float16 if args.enable_half_first_order_momentum else torch.float32
    accum_type = torch.float16 if args.fp16 else torch.float32
    if args.optimizer == "Adam":
        opt = Adam(get_trainable_params(dalle, args.weight_decay), lr=args.learning_rate, eps=1e-6, loss_scaling=args.loss_scaling,
                   accum_type=accum_type, first_order_momentum_accum_type=first_order_type, second_order_momentum_accum_type=torch.float32)
    elif args.optimizer == "AdamW":
        opt = AdamW(get_trainable_params(dalle, args.weight_decay), lr=args.learning_rate, eps=1e-6, loss_scaling=args.loss_scaling,
                    accum_type=accum_type, first_order_momentum_accum_type=first_order_type, second_order_momentum_accum_type=torch.float32)
    else:
        raise ValueError("Unknown Optimizer:", args.optimizer)
    if exists(args.pretrained_checkpoint) and opt_state:
        opt.load_state_dict(opt_state)
    poptorch_dalle = poptorch.trainingModel(dalle,
                                            options=opts,
                                            optimizer=opt)
    if args.lr_decay:
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
        if exists(args.pretrained_checkpoint) and scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    else:
        scheduler = None

    # experiment tracker

    model_config = dict(
        depth=args.num_hidden_layers,
        heads=args.num_attention_heads,
        dim_head=args.dim_head
    )

    if args.wandb:
        run = wandb.init(
            project=args.wandb_project_name,
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

    # Training loop
    log.logger.info("---------- Training Started -----------")

    save_model(args.checkpoint_output_dir, epoch=resume_epoch)
    global_batch_size = args.batch_size * args.gradient_accumulation * args.replication_factor
    samples_per_step = global_batch_size * args.batches_per_step
    training_steps = args.epochs * steps_per_epoch
    start_train = time.perf_counter()
    start_step = time.perf_counter()
    for epoch in range(resume_epoch, args.epochs):
        for i, (text, images) in enumerate(dl):
            current_step = i + epoch * steps_per_epoch
            loss = poptorch_dalle(text, images)
            # Average loss across replicas
            if args.replication_factor == 1:
                mean_loss = loss
            else:
                mean_loss = loss.mean()
            step_length = time.perf_counter() - start_step
            step_throughput = samples_per_step / step_length
            msg = ("Epoch: {:.2f}/{} "
                   "Step: {}/{} "
                   "Lr: {:.6f} "
                   "Loss: {:.3f} "
                   "Throughput: {:.2f} samples/sec"
                   ).format(epoch, args.epochs,
                            current_step, training_steps,
                            opt.param_groups[0]['lr'],
                            mean_loss.item(),
                            step_throughput)
            log.logger.info(msg)
            if args.wandb:
                wandb.log({"LR": opt.param_groups[0]['lr'],
                           "Throughput": step_throughput,
                           "Loss": mean_loss.item()})

            start_step = time.perf_counter()
            if i % args.checkpoint_save_steps == 0:
                save_model(args.checkpoint_output_dir, epoch=epoch)

        if args.lr_decay:
            scheduler.step(mean_loss)

        save_model(args.checkpoint_output_dir, epoch=epoch)

    if args.wandb:
        wandb.finish()

    stop_train = time.perf_counter()
    log.logger.info("---------------------------------------")

    log.logger.info("---------- Training Metrics -----------")
    log.logger.info(f"global_batch_size: {global_batch_size}")
    log.logger.info(f"batches_per_step: {args.batches_per_step}")
    log.logger.info(f"training_steps: {training_steps}")
    duration_run = stop_train - start_train
    num_samples = samples_per_step * training_steps
    log.logger.info(f"Training time: {duration_run:.3f} secs")
    log.logger.info("Throughput: {:5f} samples/sec.".format(num_samples / duration_run))
    log.logger.info("---------------------------------------")

if __name__ == "__main__":
    # argument parsing
    args = parse_args()

    torch.manual_seed(args.random_seed)
    main(args)
