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
import sys
import time
import logging
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import torch.onnx
import poptorch
import popdist
import popdist.poptorch

from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
from transformers import GPT2Config, GPT2LMHeadModel

from arguments import set_args
from ipu_options import get_options
from model.optimized_gpt2_attn import OptimizedGPT2Attention
from tools import (SerializedLinear, _get_layer_ipu, _WorkerInit,
                   collate_fn, get_generated_datum, load_dataset,
                   outline_attribute, sync_metrics)

MODEL_CONFIG = {'gpt2-test': 'config/config_test.json', 'gpt2': 'config/config.json',
                'gpt2-medium': 'config/config_medium.json', 'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}
file_dir = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(level=logging.INFO, format="%(message)s")


def logger(msg):
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        logging.info(msg)


class GPT2Wrapper(nn.Module):
    def __init__(self, args, model_config):
        super().__init__()
        self.args = args
        if args.pretrained_checkpoint:  # load pretrained model checkpoint
            self.model = GPT2LMHeadModel.from_pretrained(
                args.pretrained_checkpoint)
        else:  # init model
            self.config = model_config
            self.model = GPT2LMHeadModel(config=self.config)

        for layer in self.model.transformer.h:
            gpt2_attn = OptimizedGPT2Attention(
                self.model.config, layer_idx=layer.attn.layer_idx)
            gpt2_attn.load_state_dict(layer.attn.state_dict())
            layer.attn = gpt2_attn

        if args.embedding_serialization_factor > 1:
            serialized_lmhead = SerializedLinear(self.model.config.n_embd, self.model.config.vocab_size,
                                                 args.embedding_serialization_factor,
                                                 bias=False,
                                                 mode=poptorch.MatMulSerializationMode.OutputChannels)
            serialized_lmhead.load_state_dict(self.model.lm_head.state_dict())
            self.model.lm_head = serialized_lmhead
            self.model.tie_weights()

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.model.transformer.wte = poptorch.BeginBlock(
            self.model.transformer.wte, "wte", ipu_id=0)
        self.model.transformer.wpe = poptorch.BeginBlock(
            self.model.transformer.wpe, "wpe", ipu_id=1)
        outline_attribute(self.model.transformer.ln_f, "LayerNorm")

        layer_ipu = _get_layer_ipu(args.layers_per_ipu)
        for index, layer in enumerate(self.model.transformer.h):
            ipu = layer_ipu[index]
            self.model.transformer.h[index] = poptorch.BeginBlock(
                layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Layer {index:<2} --> IPU {ipu}")

        logger(f'LM_head --> IPU 0')
        self.model.lm_head = poptorch.BeginBlock(self.model.lm_head, ipu_id=0)

    def forward(self, input_ids):
        transformer_outputs = self.model.transformer(input_ids=input_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        outputs = torch.argmax(lm_logits, -1)
        return outputs


if __name__ == "__main__":
    args = set_args()
    opts = get_options(args)
    opts.setExecutionStrategy(
        poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))
    logger("Model initializing")
    model_config = GPT2Config.from_json_file(
        os.path.join(file_dir, MODEL_CONFIG[args.model]))
    model_config.n_positions = args.max_len
    model = GPT2Wrapper(args, model_config).half().eval()

    logger("Arguments: {}".format(args))
    logger("Model config: {}".format(model_config))
    poptorch_model = poptorch.inferenceModel(model, opts)

    if args.compile_only:
        # Compile model
        logger("---------- Compilation/Loading from Cache Started ---------")
        start_compile = time.perf_counter()
        datum = get_generated_datum(args, model_config.vocab_size)
        poptorch_model.compile(*datum)
        duration_compilation = time.perf_counter() - start_compile
        logger(f"Compiled/Loaded model in {duration_compilation} secs")
        logger("-----------------------------------------------------------")
        logger(
            "Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
        sys.exit(0)

    # Dataloader
    logger("------------------- Data Loading Started ------------------")
    start_loading = time.perf_counter()
    train_dataset, validate_dataset = load_dataset(
        logger, args, model_config.vocab_size)
    loader = DataLoader(opts,
                        train_dataset,
                        shuffle=(args.dataset=="pickle"),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        worker_init_fn=_WorkerInit(args.seed),
                        collate_fn=collate_fn if not (args.dataset=="mmap") else None,
                        drop_last=True,
                        auto_distributed_partitioning=not isinstance(
                            train_dataset, torch.utils.data.IterableDataset),
                        mode=DataLoaderMode.AsyncRebatched if args.async_dataloader else DataLoaderMode.Sync)
    samples_per_epoch = int(len(
        train_dataset) / args.epochs) if (args.dataset=="mmap") else len(train_dataset)
    steps_per_epoch = int(
        len(loader) / args.epochs) if (args.dataset=="mmap") else len(loader)
    logger(f"Samples per epoch: {samples_per_epoch}")
    logger(f"Steps per epoch: {steps_per_epoch}")
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration, "
                           "try reducing deviceIterations or gradientAccumulation.")
    duration_loader = time.perf_counter() - start_loading
    logger(f"Data loaded in {duration_loader} secs")
    logger("-----------------------------------------------------------")

    if args.resume_training_from_checkpoint:
        training_state = torch.load(
            Path(args.pretrained_checkpoint) / "training_state.pt")

    # Inference loop
    logger("--------------------- Inference Started --------------------")
    factor = args.gradient_accumulation * args.device_iterations
    start_train = time.perf_counter()

    epoch = 0
    total_step = 0
    while epoch < args.epochs and total_step < steps_per_epoch * args.epochs:
        for batch_idx, batch in enumerate(loader):
            if args.dataset=="mmap":
                input_ids = batch[:, :-1]
            else:
                _input_ids, _labels = batch
                input_ids = _input_ids[:, :-1]

            start_step = time.perf_counter()
            outputs = poptorch_model(input_ids=input_ids)
            step_length = sync_metrics(time.perf_counter() - start_step)
            num_instances = args.popdist_size if args.use_popdist else 1
            step_throughput = num_instances * args.replication_factor * args.batch_size * \
                args.gradient_accumulation * args.device_iterations / step_length
            if (batch_idx + 1) % args.log_steps == 0:
                logger("step {} of epoch {}, throughput: {} samples/sec, latency avg: {} ms".format(
                    batch_idx, epoch, step_throughput, step_length*1000))
            total_step += 1
            if total_step % steps_per_epoch == 0:
                epoch += 1
