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
import pytest
import transformers

import import_helper
from tokenizer import build_megatron_tokenizer

base_dir = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.ipus(0)
def test_tokenzier():
    """
    Test that the GPT2Tokenizer from Huggingface and Megatron will give the same results.
    """
    tokenizer_a = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_b = build_megatron_tokenizer(
        vocab_file=base_dir + "/../tokenizer/gpt2-vocab-50256.json", merge_file=base_dir + "/../tokenizer/gpt2-merges-50256.txt")

    text = "Generative Pre-trained Transformer 2 (GPT-2) is an open-source artificial intelligence created" \
        "by OpenAI in February 2019.[1][2][3][4] GPT-2 translates text, answers questions, summarizes passages," \
        "[5] and generates text output on a level that, while sometimes indistinguishable from that of humans,[6]" \
        "can become repetitive or nonsensical when generating long passages.[7] It is a general-purpose learner;" \
        "it was not specifically trained to do any of these tasks, and its ability to perform them is an extension" \
        "of its general ability to accurately synthesize the next item in an arbitrary sequence.[8][5] GPT-2 was" \
        "created as a 'direct scale-up' of OpenAI's 2018 GPT model,[9] with a ten-fold increase in both its" \
        "parameter count and the size of its training dataset.[4]"

    tokens_a = tokenizer_a.encode(text)
    tokens_b = tokenizer_b.encode(text)

    assert tokens_a == tokens_b
