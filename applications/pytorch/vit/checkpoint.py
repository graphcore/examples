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

import torch
from pathlib import Path
from log import logger


def _check_config_is_compatible(saved_config, config):
    depth_mismatch = saved_config.num_hidden_layers != config.num_hidden_layers
    breadth_mismatch = saved_config.hidden_size != config.hidden_size
    attention_size_mismatch = saved_config.num_attention_heads != config.num_attention_heads
    if(depth_mismatch or breadth_mismatch or attention_size_mismatch):
        raise RuntimeError("Checkpoint being loaded does not match model definition.\n"
                           f"Hidden layers: {'match' * int(depth_mismatch) + 'not match' * int(not depth_mismatch)}\n"
                           f"Hidden size: {'match' * int(breadth_mismatch) + 'not match' * int(not breadth_mismatch)}\n"
                           f"Attention layer size: {'match' * int(attention_size_mismatch) + 'not match' * int(not attention_size_mismatch)}\n")


def _load_checkpoint_from_file(file_path):
    abs_path_ckpt = os.path.abspath(file_path)

    # Return checkpoint if valid
    if os.path.isfile(abs_path_ckpt):
        try:
            checkpoint = torch.load(abs_path_ckpt)
            return checkpoint
        except Exception as e:
            logger.error(f"Failed with exception {e}.")
    else:
        raise RuntimeError("Please specify a PyTorch checkpoint file.")


def restore_checkpoint(config, val=False):
    model_path = Path(config.pretrained_checkpoint) / f'pytorch_model.bin'
    model_state_dict = _load_checkpoint_from_file(model_path)
    if val:
        return model_state_dict

    training_state_path = Path(config.pretrained_checkpoint) / f'training_state.pt'
    training_state = _load_checkpoint_from_file(training_state_path)
    _check_config_is_compatible(training_state["config"], config)
    return model_state_dict, training_state


def save_checkpoint(config, model, optimizer, step, metrics=None):
    if config.checkpoint_output_dir:
        path = Path(config.checkpoint_output_dir) / f"step_{step}"
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving checkpoint for step {step} to: {path}\n")
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(path)
        else:
            torch.save(model.state_dict(), path / f"pytorch_model.bin")
        optimizer_state = optimizer.state_dict()
        torch.save({
            "step": step,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
            "config": config
        }, os.path.join(path, "training_state.pt"))
