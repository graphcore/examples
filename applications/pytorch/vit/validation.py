# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2020 jeonsworld
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

# This file has been modified by Graphcore


import datetime
import os

import numpy as np
import poptorch
import torch
import transformers

from args import parse_args
from datasets import dataset
from ipu_options import get_options
from log import Logger
from metrics import accuracy
from model import PipelinedViTForImageClassification


if __name__ == "__main__":
    # Validation loop
    # Build config from args
    config = transformers.ViTConfig(**vars(parse_args()))

    # Execution parameters
    opts = get_options(config)

    test_loader = dataset.get_data(config, opts, train=False, async_dataloader=True)

    # Check output dir
    abs_pathd = os.path.abspath(config.checkpoint_dir)
    os.makedirs(abs_pathd, exist_ok=True)

    log = Logger(abs_pathd+"/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log',
                 level='INFO')

    model = PipelinedViTForImageClassification(config).eval()
    if config.precision.startswith("16."):
        model.half()

    # Init from a checkpoint
    try:
        state_dict = torch.load(config.checkpoint_file)["model_state_dict"]
    except KeyError as e:
        state_dict = torch.load(config.checkpoint_file)
    except FileNotFoundError as e:
        raise Exception('checkpoint not found: %s' % config.checkpoint_file)

    model.load_state_dict(state_dict)

    # Execution parameters
    valid_opts = poptorch.Options()
    valid_opts.deviceIterations(4)
    valid_opts.anchorMode(poptorch.AnchorMode.All)
    valid_opts.Precision.enableStochasticRounding(False)

    # Wrap in the PopTorch inference wrapper
    inference_model = poptorch.inferenceModel(model, options=valid_opts)
    all_acc = []
    all_preds, all_labels = [], []
    for step, (input_data, labels) in enumerate(test_loader):
        losses, logits = inference_model(input_data, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = accuracy(preds, labels)
        all_acc.append(acc)
        all_preds.append(preds.detach().clone())
        all_labels.append(labels.detach().clone())
        log.logger.info("Valid Loss: {:.3f} Acc: {:.3f}".format(torch.mean(losses).item(), acc))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    val_accuracy = accuracy(all_preds, all_labels)

    log.logger.info("\n")
    log.logger.info("Validation Results")
    log.logger.info("Valid Loss: %2.5f" % torch.mean(losses).item())
    log.logger.info("Valid Aver Batch Accuracy: %2.5f" % np.mean(all_acc))
    log.logger.info("Valid Accuracy: %2.5f" % val_accuracy)
