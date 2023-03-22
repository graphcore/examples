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

import argparse

import torch
import torch.nn as nn
import poptorch
from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from tasks.evaluate_utils import process_batch, _LambadaDataset
from tools import _get_layer_ipu, _WorkerInit, str_to_bool
from model.optimized_gpt2_attn import OptimizedGPT2Attention


def get_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--task", type=str, default="LAMBADA", choices=("WIKITEXT103", "LAMBADA"), help="task to evaluate"
    )
    parser.add_argument("--valid-data", nargs="*", default=None, help="path(s) to the validation data.")
    parser.add_argument(
        "--tokenizer-type", type=int, default=0, help="0: transformers.tokenizer, 1: Megatron.tokenizer"
    )
    parser.add_argument("--overlapping-eval", type=int, default=32, help="Sliding window for overlapping evaluation.")
    parser.add_argument(
        "--checkpoint-input-dir", default="", type=str, required=True, help="pretrained checkpoint to load"
    )
    parser.add_argument("--seq-length", default=128, type=int, required=False, help="max length of input sequence")
    parser.add_argument(
        "--layers-per-ipu", type=int, default=3, nargs="+", help="Number of decoder layers per pipeline stage"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size (default = 1)")
    parser.add_argument(
        "--device-iterations",
        default=4,
        type=int,
        required=False,
        help="Number of iterations run on the device before syncing with the host.",
    )
    parser.add_argument("--eod-mask-loss", type=str_to_bool, nargs="?", const=True, default=False, help="eod-mask-loss")
    parser.add_argument(
        "--strict-lambada", type=str_to_bool, nargs="?", const=True, default=False, help="strict-lambada"
    )
    parser.add_argument("--fp16", type=str_to_bool, nargs="?", const=True, default=False, help="if use fp16")
    parser.add_argument("--log-interval", type=int, default=10, help="log-interval")
    parser.add_argument(
        "--matmul-proportion", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul"
    )
    parser.add_argument("--executable-cache-dir", default=None, type=str, required=False, help="executable cache dir")
    args = parser.parse_args()
    return args


class GPT2Wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = GPT2LMHeadModel.from_pretrained(args.checkpoint_input_dir)
        self.model.lm_head.weight = nn.Parameter(self.model.transformer.wte.weight.clone())

        for layer in self.model.transformer.h:
            gpt2_attn = OptimizedGPT2Attention(self.model.config, layer_idx=layer.attn.layer_idx)
            gpt2_attn.load_state_dict(layer.attn.state_dict())
            layer.attn = gpt2_attn

        print("-------------------- Device Allocation --------------------")
        print("Embedding  --> IPU 0")
        self.model.transformer.wte = poptorch.BeginBlock(self.model.transformer.wte, "wte", ipu_id=0)
        self.model.transformer.wpe = poptorch.BeginBlock(self.model.transformer.wpe, "wpe", ipu_id=0)

        layer_ipu = _get_layer_ipu(args.layers_per_ipu)
        for index, layer in enumerate(self.model.transformer.h):
            ipu = layer_ipu[index]
            self.model.transformer.h[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Layer {index:<2} --> IPU {ipu}")

    def forward(self, input_ids, labels, loss_mask):
        transformer_outputs = self.model.transformer(input_ids=input_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        # pdb.set_trace()
        outputs = torch.argmax(lm_logits, -1)
        correct_ = (outputs == labels).float()
        ones = torch.ones(correct_.shape, dtype=correct_.dtype)
        correct = torch.where(loss_mask.bool(), correct_, ones.to(correct_.dtype))
        correct = correct.prod(-1)
        return correct.sum()


def build_lambada_dataset(tokenizer):
    """Build lambada dataset."""

    assert len(args.valid_data) == 1
    val_dataset = _LambadaDataset(args.valid_data[0], 50256, tokenizer, args.seq_length, args.strict_lambada)
    print(" > found {} samples.".format(len(val_dataset)))

    return val_dataset


def main():
    """Main program."""

    # Set poptorch options
    opts = poptorch.Options().deviceIterations(args.device_iterations)
    opts.autoRoundNumIPUs(True)

    mem_prop = {f"IPU{i}": args.matmul_proportion[i] for i in range(len(args.matmul_proportion))}
    opts.setAvailableMemoryProportion(mem_prop)
    opts.setExecutionStrategy(poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))
    opts._Popart.set("saveInitializersToFile", "weights.bin")
    if args.executable_cache_dir:
        opts.enableExecutableCaching(args.executable_cache_dir)
    if args.tokenizer_type == 0:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=False)
    else:
        from tokenizer import build_megatron_tokenizer

        tokenizer = build_megatron_tokenizer(
            vocab_file="./tokenizer/gpt2-vocab-50256.json", merge_file="./tokenizer/gpt2-merges-50256.txt"
        )
    dataset = build_lambada_dataset(tokenizer)
    data_loader = DataLoader(
        opts,
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=4,
        worker_init_fn=_WorkerInit(1234),
        drop_last=False,
        auto_distributed_partitioning=not isinstance(dataset, torch.utils.data.IterableDataset),
        mode=DataLoaderMode.Sync,
    )

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
                print(
                    "> working on iteration: {}, correct: {}/{}".format(
                        iteration, int(output.sum()), args.device_iterations
                    )
                )

            total_output += output.float().sum()

        num_examples = len(data_loader.dataset)
        acc = total_output / num_examples
        print(
            "number correct: {:.4E}, total examples: {:.4E}, avg accuracy: {:.4E}".format(
                total_output, num_examples, acc
            )
        )

    return total_output


if __name__ == "__main__":
    args = get_args()
    main()
