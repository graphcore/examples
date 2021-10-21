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

import glob
import os
import torch
from log import Logger


def prepare_checkpoint_metrics(outputs, factor):
    return {"Loss": outputs.div(factor).mean().item()}


def _check_config_is_compatible(saved_config, config):
    if (saved_config.num_hidden_layers != config.num_hidden_layers or
            saved_config.hidden_size != config.hidden_size or
            saved_config.num_attention_heads != config.num_attention_heads):
        raise RuntimeError("Checkpoint being loaded does not match model definition.")


def _get_checkpoint_filename(config, step):
    phase = config.dataset
    layers = config.num_hidden_layers
    hidden = config.hidden_size
    heads = config.num_attention_heads
    filename = f"{phase}_L_{layers}_H_{hidden}_A_{heads}_epoch_{step}.pt"
    return filename


def checkpoints_exist(config):
    path = os.path.abspath(config.checkpoint_dir)
    if os.path.exists(path):
        # All checkpoint files
        files = glob.glob(f"{os.path.join(path, '*.pt')}")
        if len(files) > 0:
            return True
    return False


def _load_checkpoint_from_file(config):
    abs_path_ckpt = os.path.abspath(config.checkpoint_file)

    # Return checkpoint if valid
    if os.path.isfile(abs_path_ckpt):
        try:
            checkpoint = torch.load(abs_path_ckpt)
            return checkpoint
        except Exception as e:
            log = Logger()
            log.logger.error(f"Failed with exception {e}.")
    else:
        raise RuntimeError("Please specify a PyTorch checkpoint file.")


def restore_checkpoint(config):
    checkpoint = _load_checkpoint_from_file(config)
    _check_config_is_compatible(checkpoint["config"], config)
    return checkpoint


def save_checkpoint(config, model, optimizer, epoch, metrics=None):
    if config.checkpoint_dir:
        abs_pathd = os.path.abspath(config.checkpoint_dir)
        os.makedirs(abs_pathd, exist_ok=True)
        filename = _get_checkpoint_filename(config, epoch)
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
    return save_path
