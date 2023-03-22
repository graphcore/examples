# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import glob
import os

import torch


def prepare_checkpoint_metrics(outputs, factor):
    return {"Loss": outputs.div(factor).mean().item()}


def save_model(config, model, optimizer, epoch, metrics=None, scheduler=None):
    if config.checkpoint_dir:
        abs_pathd = os.path.abspath(config.checkpoint_dir)
        os.makedirs(abs_pathd, exist_ok=True)
        filename = f"CLIP_epoch_{epoch}.pt"
        save_path = os.path.join(abs_pathd, filename)
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "metrics": metrics,
                "config": config,
                "scheduler_state_dict": scheduler_state,
            },
            save_path,
        )
        return save_path


def checkpoints_exist(path, config=None, inverse=None):
    if os.path.exists(path):
        if config is not None:
            prefix = ""
            files = glob.glob(f"{os.path.join(path, prefix + '*')}")
            if inverse is not None:
                files_all = glob.glob(f"{os.path.join(path, '*.pt')}")
                files = list(set(files_all) - set(files))
        else:
            # All checkpoint files
            files = glob.glob(f"{os.path.join(path, '*.pt')}")
        if len(files) > 0:
            return True
    return False


def get_latest_filepath(path, config):
    prefix = ""
    files = glob.iglob(f"{os.path.join(path, prefix + '*')}")
    latest_file = max(files, key=os.path.getctime)
    return os.path.join(path, latest_file)


def load_from_file_passing_constraints(config):
    if config.checkpoint_file:
        abs_path_ckpt = os.path.abspath(config.checkpoint_file)
        # Check save constraints for preventing overwrite
        if config.checkpoint_dir and checkpoints_exist(os.path.abspath(config.checkpoint_dir)):
            raise RuntimeError(
                "Found previously saved checkpoint(s) at checkpoint-dir. "
                "Overwriting them with checkpoints building on checkpoint-file "
                "is not supported. Please specify a different checkpoint-dir to "
                "save checkpoints from this run."
            )
        # Return checkpoint if valid
        if os.path.isfile(abs_path_ckpt):
            try:
                checkpoint = torch.load(abs_path_ckpt)
                return checkpoint
            except Exception as e:
                print(f"Failed with exception {e}.")
        else:
            raise RuntimeError("Please specify a PyTorch checkpoint file.")
    return None


def load_from_dir_passing_constraints(config):
    # Latest checkpoint at checkpoint_dir
    if config.checkpoint_dir:
        abs_pathd = os.path.abspath(config.checkpoint_dir)
        print("abs_pathd: ", abs_pathd)
        # Check save constraints for resuming run without overwrite
        if checkpoints_exist(abs_pathd, config, inverse=True):
            raise RuntimeError(
                "Please specify a different checkpoint-dir. " "This one has checkpoints from another incompatible run."
            )
        if checkpoints_exist(abs_pathd, config):
            if config.restore_epochs_and_optimizer:
                abs_path_ckpt = get_latest_filepath(abs_pathd, config)
                try:
                    checkpoint = torch.load(abs_path_ckpt)
                    return checkpoint
                except Exception as e:
                    print(f"Failed with exception {e}.")
            else:
                # Overwrite prevention
                raise RuntimeError(
                    "Please restore full state to continue training. "
                    "Alternatively, specify a different checkpoint-dir "
                    "and a checkpoint-file from which to restore "
                    "(only model) state and retrain."
                )
    return None


def load_checkpoint_passing_constraints(config):

    if config.checkpoint_file:
        checkpoint = load_from_file_passing_constraints(config)
    else:
        checkpoint = load_from_dir_passing_constraints(config)

    return checkpoint
