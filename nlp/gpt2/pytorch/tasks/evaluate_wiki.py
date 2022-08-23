# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
# This file has been modified by Graphcore Ltd.

"""GPT2 zero-shot evaluation."""

import math
import argparse

import torch
import torch.nn as nn
import poptorch

from torch.nn import CrossEntropyLoss
from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tasks.detokenizer import get_detokenizer
from tasks.evaluate_utils import process_batch, _LMDataset
from tools import _get_layer_ipu, _WorkerInit, str_to_bool
from model.optimized_gpt2_attn import OptimizedGPT2Attention


def get_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--task', type=str, default='WIKITEXT103', choices=('WIKITEXT103', 'LAMBADA'),
                        help='task to evaluate')
    parser.add_argument('--valid-data', nargs='*', default=None,
                        help='path(s) to the validation data.')
    parser.add_argument('--tokenizer-type', type=int, default=0,
                        help='0: transformers.tokenizer, 1: Megatron.tokenizer')
    parser.add_argument('--overlapping-eval', type=int, default=32,
                        help='Sliding window for overlapping evaluation.')
    parser.add_argument('--pretrained-checkpoint', default='',
                        type=str, required=True, help='pretrained checkpoint to load')
    parser.add_argument('--seq-length', default=128, type=int,
                        required=False, help='max length of input sequence')
    parser.add_argument('--layers-per-ipu', type=int, default=3, nargs="+",
                        help='Number of decoder layers per pipeline stage')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default = 1)')
    parser.add_argument('--device-iterations', default=4,
                        type=int, required=False, help='Number of iterations run on the device before syncing with the host.')
    parser.add_argument("--eod-mask-loss", type=str_to_bool,
                        nargs="?", const=True, default=False, help="eod-mask-loss")
    parser.add_argument("--fp16", type=str_to_bool, nargs="?",
                        const=True, default=False, help="if use fp16")
    parser.add_argument('--log-interval', type=int,
                        default=10, help='log-interval')
    parser.add_argument("--matmul-proportion", type=float, nargs="+",
                        help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument('--executable-cache-dir', default=None, type=str, required=False,
                        help='executable cache dir')
    args = parser.parse_args()
    return args


class GPT2Wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_checkpoint)
        self.model.lm_head.weight = nn.Parameter(
            self.model.transformer.wte.weight.clone())

        for layer in self.model.transformer.h:
            gpt2_attn = OptimizedGPT2Attention(
                self.model.config, layer_idx=layer.attn.layer_idx)
            gpt2_attn.load_state_dict(layer.attn.state_dict())
            layer.attn = gpt2_attn

        print("-------------------- Device Allocation --------------------")
        print("Embedding  --> IPU 0")
        self.model.transformer.wte = poptorch.BeginBlock(
            self.model.transformer.wte, "wte", ipu_id=0)
        self.model.transformer.wpe = poptorch.BeginBlock(
            self.model.transformer.wpe, "wpe", ipu_id=0)

        layer_ipu = _get_layer_ipu(args.layers_per_ipu)
        for index, layer in enumerate(self.model.transformer.h):
            ipu = layer_ipu[index]
            self.model.transformer.h[index] = poptorch.BeginBlock(
                layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Layer {index:<2} --> IPU {ipu}")

    def forward(self, input_ids, labels, loss_mask):
        transformer_outputs = self.model.transformer(input_ids=input_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(
            lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        loss = torch.sum(losses.view(-1) *
                         loss_mask.contiguous().view(-1).float())

        return loss


def build_wikitext103_dataset(tokenizer):
    assert len(args.valid_data) == 1
    with open(args.valid_data[0], "rb") as reader:
        entire_data = reader.read().decode('utf-8')

    num_original_tokens = len(entire_data.strip().split(" "))
    entire_data = get_detokenizer(args.valid_data[0])(entire_data)
    tokenized_data = tokenizer.encode(entire_data)
    num_tokenized_tokens = len(tokenized_data)

    val_dataset = _LMDataset(tokenized_data, args.seq_length, 50256,
                             num_original_tokens, num_tokenized_tokens,
                             args.overlapping_eval)
    print(' > number of original tokens: {}, number of detokenized'
          'tokens: {}'.format(num_original_tokens, num_tokenized_tokens))

    return val_dataset


def main():
    """Main program."""

    # Set poptorch options
    opts = poptorch.Options().deviceIterations(args.device_iterations)
    opts.autoRoundNumIPUs(True)

    mem_prop = {
        f'IPU{i}': args.matmul_proportion[i]
        for i in range(len(args.matmul_proportion))
    }
    opts.setAvailableMemoryProportion(mem_prop)
    opts.setExecutionStrategy(
        poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))
    opts._Popart.set("saveInitializersToFile", "weights.bin")
    if args.executable_cache_dir:
        opts.enableExecutableCaching(args.executable_cache_dir)
    if args.tokenizer_type == 0:
        tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2', add_prefix_space=False)
    else:
        from tokenizer import build_megatron_tokenizer
        tokenizer = build_megatron_tokenizer(
            vocab_file="./tokenizer/gpt2-vocab-50256.json", merge_file="./tokenizer/gpt2-merges-50256.txt")
    dataset = build_wikitext103_dataset(tokenizer)
    data_loader = DataLoader(opts,
                             dataset,
                             shuffle=False,
                             batch_size=args.batch_size,
                             num_workers=4,
                             worker_init_fn=_WorkerInit(1234),
                             drop_last=True,
                             auto_distributed_partitioning=not isinstance(
                                 dataset, torch.utils.data.IterableDataset),
                             mode=DataLoaderMode.Sync)

    model = GPT2Wrapper(args)
    if args.fp16:
        model = poptorch.inferenceModel(model.half(), opts)
    else:
        model = poptorch.inferenceModel(model, opts)
    model.eval()
    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            # Forward evaluation.
            tokens, labels, loss_mask = process_batch(batch)
            output = model(tokens, labels, loss_mask)
            if iteration % args.log_interval == 0:
                print('> working on iteration: {}, loss: {}'.format(
                    iteration, output.mean()))

            total_output += output.float().sum()
        num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
        num_original_tokens = data_loader.dataset.num_original_tokens
        val_loss = total_output / (num_tokenized_tokens - 1)
        ppl = math.exp(min(20, val_loss))
        token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
        print("avg loss: {:.4E}, ppl: {:.4E}, adjusted ppl: {:.4E}, token ratio: {}".format(
            val_loss, ppl, adjusted_ppl, token_ratio))

    return total_output


if __name__ == "__main__":
    args = get_args()
    main()
