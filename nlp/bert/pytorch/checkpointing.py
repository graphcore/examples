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
from utils import logger


def checkpoints_exist(path):
    if os.path.exists(path):
        # All checkpoint files
        files = glob.glob(f"{os.path.join(path, 'step_*/*.pt')}")
        if len(files) > 0:
            return True
    return False


def save_checkpoint(config, model, step, optimizer=None, metrics=None):
    if config.checkpoint_output_dir:
        path = os.path.join(os.path.abspath(config.checkpoint_output_dir), f"step_{step}")
        os.makedirs(path, exist_ok=True)

        logger(f"Saving checkpoint for step {step} to: {path}\n")
        model.save_pretrained(path)
        if optimizer is None:
            torch.save({
                "step": step,
                "metrics": metrics,
                "config": config
            }, os.path.join(path, "training_state.pt"))
        else:
            torch.save({
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config
            }, os.path.join(path, "training_state.pt"))
