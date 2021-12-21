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


def save_checkpoint(config, model, optimizer, step, metrics=None):
    if config.checkpoint_output_dir:
        path = Path(config.checkpoint_output_dir) / f"step_{step}"
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving checkpoint for step {step} to: {path}\n")
        model.save_pretrained(path)
        optimizer_state = optimizer.state_dict()
        torch.save({
            "step": step,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
            "config": config
        }, os.path.join(path, "training_state.pt"))
