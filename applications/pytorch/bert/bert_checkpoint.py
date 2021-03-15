# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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
import glob
import torch


def prepare_checkpoint_metrics(outputs, factor):
    return {"Loss": outputs[0].div(factor).mean().item(),
            "Acc/MLM": outputs[3].div(factor).mean().item(),
            "Acc/NSP": outputs[4].div(factor).mean().item()}


def check_sanity_configs_compatible(saved_config, config):
    if (saved_config.num_hidden_layers != config.num_hidden_layers or
            saved_config.hidden_size != config.hidden_size or
            saved_config.num_attention_heads != config.num_attention_heads):
        raise RuntimeError("Checkpoint being loaded does not match model definition.")


def get_filename_or_prefix(config, epoch=None):
    phase = config.dataset
    model_type = config.model_type
    layers = config.num_hidden_layers
    hidden = config.hidden_size
    heads = config.num_attention_heads
    seqlen = config.sequence_length
    prefix = f"{phase}_{model_type}_L_{layers}_H_{hidden}_A_{heads}_seqlen_{seqlen}"
    if epoch is not None:
        filename = f"{prefix}_epoch_{epoch}.pt"
        return filename
    return prefix


def checkpoints_exist(path, config=None, mismatch_allowed=False):

    if os.path.exists(path):
        if config is not None:
            # Checkpoint files given config
            prefix = get_filename_or_prefix(config)
            files = glob.glob(f"{os.path.join(path, prefix + '*')}")
            if mismatch_allowed is True:
                # Look for checkpoints that do not match config
                files_all = glob.glob(f"{os.path.join(path, '*.pt')}")
                files = list(set(files_all)-set(files))
        else:
            # All checkpoint files
            files = glob.glob(f"{os.path.join(path, '*.pt')}")
        if len(files) > 0:
            return True
    return False


def get_latest_filepath(path, config):
    prefix = get_filename_or_prefix(config)
    files = glob.iglob(f"{os.path.join(path, prefix + '*')}")
    latest_file = max(files, key=os.path.getctime)
    return os.path.join(path, latest_file)


def maybe_load_from_file_passing_constraints(config):
    if config.checkpoint_file:
        abs_path_ckpt = os.path.abspath(config.checkpoint_file)
        # Check save constraints for preventing overwrite
        if (config.checkpoint_dir and
                checkpoints_exist(os.path.abspath(config.checkpoint_dir))):
            raise RuntimeError("Found previously saved checkpoint(s) at checkpoint-dir. "
                               "Overwriting them with checkpoints building on checkpoint-file "
                               "is not supported. Please specify a different checkpoint-dir to "
                               "save checkpoints from this run.")
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


def maybe_load_from_dir_passing_constraints(config):
    # Latest checkpoint at checkpoint_dir
    if config.checkpoint_dir:
        abs_pathd = os.path.abspath(config.checkpoint_dir)
        # Check save constraints for resuming run without overwrite
        if checkpoints_exist(abs_pathd, config, mismatch_allowed=True):
            raise RuntimeError("Please specify a different checkpoint-dir. "
                               "This one has checkpoints from another incompatible run.")
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
                raise RuntimeError("Please restore full state to continue training. "
                                   "Alternatively, specify a different checkpoint-dir "
                                   "and a checkpoint-file from which to restore "
                                   "(only model) state and retrain.")
    return None


def maybe_load_checkpoint_passing_constraints(config):
    # Loading checkpoint_file prioritised
    checkpoint = maybe_load_from_file_passing_constraints(config)
    if not checkpoint:
        checkpoint = maybe_load_from_dir_passing_constraints(config)

    if checkpoint:
        check_sanity_configs_compatible(checkpoint["config"], config)

    return checkpoint


def save_model(config, model, optimizer, epoch, metrics=None):
    if config.checkpoint_dir:
        abs_pathd = os.path.abspath(config.checkpoint_dir)
        os.makedirs(abs_pathd, exist_ok=True)
        filename = get_filename_or_prefix(config, epoch)
        save_path = os.path.join(abs_pathd, filename)
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
            "config": config
        }, save_path)
