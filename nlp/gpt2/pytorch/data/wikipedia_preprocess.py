# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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
import argparse
from transformers import BertTokenizer, BertTokenizerFast, GPT2TokenizerFast
import pickle
import numpy as np
from tqdm import tqdm
import random


class WikicorpusTextFormatting:
    def __init__(self, wiki_path, output_filename, recursive=False):
        self.wiki_path = wiki_path
        self.recursive = recursive
        self.output_filename = output_filename

    # This puts one article per line
    def merge(self):
        with open(self.output_filename, mode='w', newline='\n') as ofile:
            for dirname in glob.glob(self.wiki_path + '/*/', recursive=False):
                for filename in glob.glob(dirname + 'wiki_*', recursive=self.recursive):
                    print(filename)
                    article_lines = []
                    article_open = False

                    with open(filename, mode='r', newline='\n') as file:
                        for line in file:
                            if '<doc id=' in line:
                                article_open = True
                            elif '</doc>' in line:
                                article_open = False
                                for oline in article_lines[1:]:
                                    if oline != '\n':
                                        ofile.write(oline.rstrip() + " ")
                                ofile.write("\n\n")
                                article_lines = []
                            else:
                                if article_open:
                                    article_lines.append(line)


def main(args):
    # Step 1: merge the data into one txt file
    wiki_path = args.input_file_path
    output_filename = args.output_file_path + \
        '/wikicorpus_en_one_article_per_line.txt'
    # wiki_formatter = WikicorpusTextFormatting(wiki_path, output_filename, recursive=True)
    # wiki_formatter.merge()

    # Step 2: tokenize the articles
    output_path = args.output_file_path + '/wikicorpus_en_one_article_per_line.pkl'
    print("preprocessing data,data path:{}, save path:{}".format(
        output_filename, output_path))

    if args.use_bpe:
        print('Generate and use BPE tokenizer...')
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

        # Initialize a tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Customize pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # And then train
        trainer = trainers.BpeTrainer(
            vocab_size=30522,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        tokenizer.train([
            output_filename
        ], trainer=trainer)

        # And Save it
        tokenizer.save("gpt2-bpe-tokenizer.json", pretty=True)

        # Use the generated file
        tokenizer = Tokenizer.from_file("gpt2-bpe-tokenizer.json")
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            'gpt2', add_prefix_space=False)

    data = open(output_filename, 'rb')
    train_data = data.readlines()

    text_len = []
    text_list = []
    with open(output_path, "w", encoding="utf-8") as f:
        for index, text in enumerate(tqdm(train_data)):
            utterances = text.decode("utf-8").split("\n")

            input_ids = []  # begin with [CLS]
            for utterance in utterances:
                if args.use_bpe:
                    input_ids += tokenizer.encode(utterance).ids
                else:
                    input_ids += tokenizer.encode(utterance,
                                                  add_special_tokens=False)
                    # end with eod
                    input_ids += tokenizer.encode('<|endoftext|>')
            if len(input_ids) >= args.min_length:
                text_len.append(len(input_ids))
                text_list.append(input_ids)
                # text_list += input_ids
    random.shuffle(text_list)
    len_mean = np.mean(text_len)
    len_median = np.median(text_len)
    len_max = np.max(text_len)
    with open(output_path, "wb") as f:
        pickle.dump(text_list, f)
    print("finish preprocessing data,the result is stored in {}".format(output_path))
    print("mean of text len:{},median of text len:{},max len:{}".format(
        len_mean, len_median, len_max))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file-path', required=True, type=str)
    parser.add_argument('--output-file-path', required=True, type=str)
    parser.add_argument('--use-bpe', action='store_true',
                        help='use bpe or GPT2 tokenizer')
    parser.add_argument('--min-length', default=10, type=int,
                        required=False, help='minimal length of dataset')
    args = parser.parse_args()
    main(args)
