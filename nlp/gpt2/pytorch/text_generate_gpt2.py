# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import time
import ctypes
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import poptorch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from ipu_options import load_custom_ops
from tools import _get_layer_ipu, str_to_bool
from model.optimized_gpt2_attn import OptimizedGPT2AttentionBuffer, OptimizedGPT2AttentionCache

MODEL_CONFIG = {'gpt2': 'config/config.json', 'gpt2-medium': 'config/config_medium.json',
                'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}

logging.basicConfig(level=logging.INFO, format="%(message)s")


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name-or-path", type=str, default="gpt2",
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CONFIG.keys()))
    parser.add_argument('--tokenizer-type', type=int, default=0,
                        help='0: transformers.tokenizer, 1: Megatron.tokenizer')
    parser.add_argument('--temperature', default=1.2,
                        type=float, required=False, help='temperature')
    parser.add_argument('--repetition-penalty', default=2.0,
                        type=float, required=False, help="repetition_penalty")
    parser.add_argument('--topk', default=4, type=int,
                        required=False, help='topk to choice')
    parser.add_argument('--save-samples-path', type=str, default=None,
                        required=False, help="path to save generated text")
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt as input')
    parser.add_argument('--input-len', type=int, default=64,
                        help='Maximum input length')
    parser.add_argument('--output-len', type=int, default=128,
                        help='Maximum length of generated text')
    parser.add_argument("--batch-size", type=int, default=1,
                        help='batch size (default = 1)')
    parser.add_argument('--device-iterations', type=int,
                        default=1, help='device iterations (default = 1)')
    parser.add_argument("--single-ipu", type=str_to_bool, nargs="?",
                        const=True, default=False, help="single ipu or not")
    parser.add_argument('--layers-per-ipu', type=int, default=3,
                        nargs="+", help='Number of decoder layers per pipeline stage.')
    parser.add_argument("--matmul-proportion", type=float, nargs="+",
                        help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--fp16", type=str_to_bool, nargs="?",
                        const=True, default=False, help="run model in fp16")
    parser.add_argument("--stop-token", type=str, default="<|endoftext|>",
                        help='Token at which text generation is stopped')
    parser.add_argument("--poptorch-loop", type=str_to_bool, nargs="?", const=True,
                        default=False, help="using poptorch_loop to avoid too much streamcopy")
    return parser.parse_args()


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.count = args.output_len
        self.args = args
        if args.model_name_or_path:
            self.model = GPT2Model.from_pretrained(args.model_name_or_path)
        else:
            raise RuntimeError("--model-name-or-path must be set.")
        self.nop = poptorch.nop
        self.optimize()
        if not args.single_ipu:
            self.sharding_mapping()

    def optimize(self):
        self.model.config.batch = self.args.batch_size
        self.model.config.seq = self.args.input_len + self.args.output_len
        self.model.config.input_len = self.args.input_len
        self.model.config.output_len = self.args.output_len
        self.model.config.activation_function = "gelu"
        inner_dim = self.model.config.n_inner if self.model.config.n_inner is not None else 4 * \
            self.model.config.hidden_size
        for layer in self.model.h:
            if self.args.poptorch_loop:
                GPT2Attn = OptimizedGPT2AttentionCache(self.model.config)
            else:
                GPT2Attn = OptimizedGPT2AttentionBuffer(self.model.config)
            MLP = GPT2MLP(inner_dim, self.model.config)
            GPT2Attn.load_state_dict(layer.attn.state_dict(), strict=False)
            MLP.load_state_dict(layer.mlp.state_dict(), strict=False)
            layer.attn = GPT2Attn
            layer.mlp = MLP

    def forward(self, context, dynamic_mask, position_ids):
        if self.args.poptorch_loop:
            # 1 stage
            kv_size = (self.model.config.batch, self.model.config.n_head, self.model.config.seq, int(
                self.model.config.n_embd/self.model.config.n_head))
            # past key value
            past_key_values = [[torch.zeros(kv_size), torch.zeros(
                kv_size)] for _ in range(self.model.config.n_layer)]
            new_past_keys = []
            new_past_values = []
            position_ids_stage_1 = torch.arange(0, self.args.input_len, dtype=torch.long).unsqueeze(0)
            hidden_states = self.model(
                context, position_ids=position_ids_stage_1, past_key_values=None, return_dict=False)

            presents = hidden_states[1]
            for ((past_key, past_value), (present_key, present_value)) in zip(past_key_values, presents):
                past_key = torch.cat(
                    (present_key, past_key[:, :, :-self.model.config.input_len, :]), dim=-2)
                past_value = torch.cat(
                    (present_value, past_value[:, :, :-self.model.config.input_len, :]), dim=-2)
                new_past_keys.append(past_key)
                new_past_values.append(past_value)
            new_past_keys_tensor = torch.stack(new_past_keys, dim=0)
            new_past_values_tensor = torch.stack(new_past_values, dim=0)

            index_one_hot = torch.nn.functional.one_hot(
                position_ids, num_classes=self.args.input_len).to(torch.float)
            last_hidden = torch.matmul(index_one_hot, hidden_states[0]).view(
                self.args.batch_size, -1)
            next_token_logits = torch.matmul(
                last_hidden, self.model.wte.weight.T)
            (next_token_value, next_token) = torch.topk(next_token_logits, 1)
            new_context = next_token
            new_position_ids = position_ids + 1
            record = torch.ones(self.args.batch_size,
                                self.count).to(torch.int64) * (0)
            new_record = torch.cat(
                (next_token.to(torch.int64), record[:, :-1]), dim=-1)
            # 2 stage

            def body(context, dynamic_mask, position_ids, record, past_keys, past_values):
                past_key_values = []
                for index in range(self.model.config.n_layer):
                    key_ = past_keys[index, :, :, :, :]
                    value_ = past_values[index, :, :, :, :]
                    past_key_values.append([key_, value_])
                hidden_states = self.model(context, attention_mask=dynamic_mask,
                                           position_ids=position_ids, past_key_values=past_key_values, return_dict=False)
                presents = hidden_states[1]
                present_keys = torch.stack([k for (k, v) in presents], dim=0)
                present_values = torch.stack([v for (k, v) in presents], dim=0)

                next_token_logits = torch.matmul(
                    hidden_states[0], self.model.wte.weight.T).view(self.args.batch_size, -1)
                (next_token_value, next_token) = torch.topk(
                    next_token_logits, self.args.topk)
                # We simply do a random selection after topk to avoid repetitions
                # Notice: Here we use 'argmax' + 'randn' instead of 'randint' which is unsupported.
                random_choice_idx = torch.argmax(torch.randn((1, self.args.topk)), axis=1)
                next_token = next_token[:, random_choice_idx]

                next_dynamic_mask = torch.cat((torch.ones(self.args.batch_size, 1).to(
                    torch.int64), dynamic_mask[:, :-1]), dim=-1)
                next_id = next_token
                next_position_ids = position_ids + 1
                next_record = torch.cat(
                    (next_token.to(torch.int64), record[:, :-1]), dim=-1)

                return next_id, next_dynamic_mask, next_position_ids, next_record, present_keys, present_values
            new_context, dynamic_mask, new_position_ids, new_record, new_past_keys_tensor, new_past_values_tensor = poptorch.for_loop(
                self.count-1, body, [new_context, dynamic_mask, new_position_ids, new_record, new_past_keys_tensor, new_past_values_tensor])
            return new_record
        else:
            hidden_states = self.model(context, attention_mask=dynamic_mask,
                                       position_ids=position_ids, past_key_values=None, return_dict=False)
            hidden_states_ = self.nop(hidden_states[0])
            next_token_logits = torch.matmul(
                hidden_states_, self.model.wte.weight.T).view(self.args.batch_size, -1)
            (next_token_value, next_token) = torch.topk(
                next_token_logits, self.args.topk)
            # We simply do a random selection after topk to avoid repetitions
            # Notice: Here we use 'argmax' + 'randn' instead of 'randint' which is unsupported.
            random_choice_idx = torch.argmax(torch.randn((1, self.args.topk)), axis=1)
            next_token = next_token[:, random_choice_idx]

            next_dynamic_mask = torch.cat((torch.ones(self.args.batch_size, 1).to(
                torch.int64), dynamic_mask[:, :-1]), dim=-1)

            return next_token, next_dynamic_mask

    def sharding_mapping(self):
        print("-------------------- Device Allocation --------------------")
        print("Embedding  --> IPU 0")
        self.model.wte = poptorch.BeginBlock(self.model.wte, "emb", ipu_id=0)

        layer_ipu = _get_layer_ipu(self.args.layers_per_ipu)
        for index, layer in enumerate(self.model.h):
            ipu = layer_ipu[index]
            self.model.h[index] = poptorch.BeginBlock(
                layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Layer {index:<2} --> IPU {ipu}")
        self.nop = poptorch.BeginBlock(self.nop, ipu_id=0)


def main():
    # custom op
    load_custom_ops()
    args = set_args()
    if args.poptorch_loop and not args.single_ipu:
        raise("poptorch_loop did not support multi IPUs")
    model = GPT2Wrapper(args)
    if args.single_ipu:
        mem_prop = {'IPU0': 0.2}
    else:
        mem_prop = {
            f'IPU{i}': args.matmul_proportion[i]
            for i in range(len(args.matmul_proportion))
        }
    # Set poptorch options
    opts = poptorch.Options().deviceIterations(args.device_iterations)
    opts.autoRoundNumIPUs(True)
    opts.setAvailableMemoryProportion(mem_prop)
    opts._Popart.set("saveInitializersToFile", "weights.bin")
    if not args.single_ipu:
        opts.setExecutionStrategy(poptorch.ShardedExecution())
    if args.fp16:
        model.half()
    model = poptorch.inferenceModel(model.eval(), opts)

    if args.tokenizer_type == 0:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        from tokenizer import build_megatron_tokenizer
        tokenizer = build_megatron_tokenizer(
            vocab_file="./tokenizer/gpt2-vocab-50256.json", merge_file="./tokenizer/gpt2-merges-50256.txt")

    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path +
                            '/samples.txt', 'a', encoding='utf8')
        samples_file.write("Text generator record{}:\n".format(datetime.now()))

    while True:
        try:
            if args.prompt is not None:
                text = args.prompt
            else:
                text = input("Input: ")
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            txt_len = len(text_ids)
            if args.input_len < txt_len:
                print("Input text length {0} larger than limit {1}".format(
                    txt_len, args.input_len))
                continue

            if args.save_samples_path:
                samples_file.write("Input: {}\n".format(text))

            input_ids_all = torch.tensor(text_ids).long()
            all_ids = np.array([[text_ids[0]] for _ in range(args.batch_size)])
            input_ids = torch.ones(args.batch_size, 1).to(
                torch.int64)*text_ids.pop(0)
            position_ids = torch.zeros(args.batch_size, 1).to(torch.int64)
            dynamic_mask = torch.zeros(
                args.batch_size, args.input_len+args.output_len).to(torch.int64)
            dynamic_mask[:, 0] = 1
            model_time = []
            if args.poptorch_loop:
                padding_size = args.input_len - txt_len
                padding = torch.ones(args.batch_size, padding_size) * (0)
                input_ids_all = input_ids_all.repeat(args.batch_size, 1)
                input_ids_all_pad = torch.concat(
                    [input_ids_all.view(args.batch_size, -1), padding], axis=-1).to(torch.int64)
                dynamic_mask[:, :txt_len+1] = 1
                position_ids += txt_len - 1
                # compile
                start_time = time.time()
                in1_ = input_ids_all_pad.clone()
                in2_ = dynamic_mask.clone()
                in3_ = position_ids.clone()
                _ = model(in1_, in2_, in3_)
                end_time = time.time()
                model_time.append(end_time - start_time)

                start_time = time.time()
                record = model(input_ids_all_pad, dynamic_mask, position_ids)
                end_time = time.time()
                model_time.append(end_time - start_time)
                output_tokens = torch.flip(record, dims=[1]).to(torch.int64)
                all_ids = torch.concat([input_ids_all.view(
                    args.batch_size, -1).to(torch.int64), output_tokens], axis=-1)
                logging.info(
                    "latency avg per sentence: {0} ms/sentence_({1})".format(np.mean(model_time[1:])*1000, args.output_len))
                logging.info(
                    "Per-token latency avg: {} ms/token".format(np.mean(model_time[1:])*1000/args.output_len))
                logging.info("Batch size: {0}; Input length {1}; Output length {2}, throughput: {3} samples/sec \n".format(
                    args.batch_size, txt_len, args.output_len, args.batch_size / np.mean(model_time[1:])))
            else:
                for _ in range(args.input_len + args.output_len):
                    start_time = time.time()
                    input_ids, dynamic_mask = model(
                        input_ids.to(torch.int64), dynamic_mask.to(torch.int64), position_ids)
                    end_time = time.time()
                    model_time.append(end_time - start_time)
                    position_ids += 1
                    if len(text_ids) > 0:
                        input_ids = torch.ones(args.batch_size, 1).to(
                            torch.int64) * text_ids.pop(0)
                    all_ids = np.concatenate(
                        (all_ids, input_ids.view(args.batch_size, -1).numpy()), axis=1)
                logging.info(
                    "latency avg per sentence: {0} ms/sentence_({1})".format(np.sum(model_time[1:])*1000, args.output_len))
                logging.info(
                    "Per-token latency avg: {} ms/token".format(np.mean(model_time[1:])*1000))
                logging.info("Batch size: {0}; Input length {1}; Output length {2}, throughput: {3} samples/sec \n".format(
                    args.batch_size, txt_len, args.output_len, args.batch_size / np.sum(model_time[1:])))

            for batch in all_ids.tolist():
                text = tokenizer.decode(
                    batch, clean_up_tokenization_spaces=True)
                text = text[: text.find(args.stop_token)
                            if args.stop_token else None]
                logging.info(text)
                if args.save_samples_path:
                    samples_file.write("Output: {}\n".format("".join(text)))

            if args.prompt is not None:
                break

        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
