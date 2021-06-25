# coding=utf-8
# Copyright (c) 2020 Graphcore Ltd. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file has been modified by Graphcore Ltd.

"""
This script has been adapated from the original google-research/bert repo found here:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py

Main changes:
  Remove dependency on TensorFlow
  Save files as binaries
  Optionally rearrange MLM tokens to the front of the sequence (required for un-packed BERT on IPU)
  Perform tokenization in parallel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tqdm import tqdm
import collections
import random
import argparse
import glob
import struct
from itertools import chain, repeat
from functools import reduce
from concurrent.futures import ProcessPoolExecutor

import tokenization

def parallel_tokenizer(chunk, tokenizer):
  indices, lines = chunk
  tokens = {}
  for i, index in enumerate(indices):
    line = tokenization.convert_to_unicode(lines[i])
    line = line.strip()
    t = tokenizer.tokenize(line)
    tokens[index] = t
  return tokens


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    mask_tokens, output_files, args, max_samples=-1):
  """Create Binary files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(open(output_file, "wb"))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(tqdm(instances)):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1] * len(masked_lm_ids)
    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["segment_ids"] = segment_ids
    features["masked_lm_positions"] = masked_lm_positions
    features["masked_lm_ids"] = masked_lm_ids
    features["masked_lm_weights"] = masked_lm_weights
    features["next_sentence_labels"] = [next_sentence_label]

    if not args.dont_rearrange_mlm_tokens_to_front:
      # -----------------------------------------
      # Main Change to original script. This handles the re-arranging of samples to put mask_tokens at the start.
      formatted_input = [0] * max_seq_length
      formatted_pos = [args.pad_position_value] * max_seq_length
      formatted_seg = [0] * max_seq_length
      formatted_label = [0] * mask_tokens
      current_mask_idx = 0
      current_seq_idx = mask_tokens
      for idx, input_id in enumerate(input_ids):
        if input_id == 0:
          continue
        try:
          masked_lm_idx = masked_lm_positions.index(idx)
          formatted_input[current_mask_idx] = input_id
          formatted_pos[current_mask_idx] = idx
          formatted_seg[current_mask_idx] = segment_ids[idx]
          formatted_label[current_mask_idx] = masked_lm_ids[masked_lm_idx]
          current_mask_idx += 1
        except ValueError:
          formatted_input[current_seq_idx] = input_id
          formatted_pos[current_seq_idx] = idx
          formatted_seg[current_seq_idx] = segment_ids[idx]
          current_seq_idx += 1

      mask_tokens_padding_idx = [current_mask_idx]
      sequence_padding_idx = [current_seq_idx]
      nsp_label = [next_sentence_label]

      # Pack into binary format
      line = reduce(lambda accl, i: accl + struct.pack('<I', i),
                    chain(formatted_input,
                          formatted_pos,
                          formatted_seg,
                          mask_tokens_padding_idx,
                          sequence_padding_idx,
                          formatted_label,
                          nsp_label), b'')
      writers[writer_index].write(line)
      # -------------------------------------------
    else:
      #   Using the packed data format for 1 seq per pack
      #   SEQ:
      #   input_ids, [sequence_length]
      #   input_mask, [sequence_length]
      #   segment_ids, [sequence_length]
      #   positions, [sequence_length] 
      #   MLM:
      #   masked_lm_positions, [mask_tokens + max_sequences_per_pack]
      #   masked_lm_ids, [mask_tokens + max_sequences_per_pack]
      #   masked_lm_weights, [mask_tokens + max_sequences_per_pack]
      #   NSP:
      #   next_sentence_positions, [max_sequences_per_pack]
      #   next_sentence_labels, [max_sequences_per_pack]
      #   next_sentence_weights, [max_sequences_per_pack]
      while len(masked_lm_positions) < (mask_tokens + 1):
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0)

      line = reduce(lambda accl, i: accl + struct.pack('<I', i),
                    chain(input_ids,
                          input_mask,
                          segment_ids,
                          list(range(max_seq_length)),
                          masked_lm_positions,
                          masked_lm_ids,
                          masked_lm_weights,
                          [0],
                          [next_sentence_label],
                          [1]), b'')
      writers[writer_index].write(line)


    writer_index = (writer_index + 1) % len(writers)
    total_written += 1

    if inst_index < 20:
      print("*** Example ***")
      print("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        print(
            "%s: %s" % (feature_name, " ".join([str(x) for x in features[feature_name]])))

    if max_samples != -1 and total_written >= max_samples:
      break

  for writer in writers:
    writer.close()

  print("Wrote %d total instances" % total_written)


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, mlm_prob,
                              mask_tokens, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  lines = []
  num_lines = 0
  for input_file in input_files:
    with open(input_file, "r") as reader:
      new_lines = reader.readlines() + ["\n"]
      num_lines += len(new_lines)
      lines.extend(new_lines)

  # Tokenization is slow and can be done in parallel
  start = time.time()
  chunks = []
  chunksize = 4000
  for i, t in enumerate(range(0, num_lines, chunksize)):
    line_chunk = lines[t:min(t+chunksize, num_lines)]
    chunks.append([range(i*chunksize, i*chunksize+len(line_chunk)), line_chunk])

  tokens = {}
  with ProcessPoolExecutor() as executor:
    work = chunks, repeat(tokenizer)
    for partial_result in executor.map(parallel_tokenizer, *work):
      tokens.update(partial_result)

  # Build documents
  for i in range(num_lines):
    # Empty lines are used as document delimiters
    if len(tokens[i]) == 0:
      all_documents.append([])
    else:
      all_documents[-1].append(tokens[i])
  print(f"Tokenizing {num_lines} lines from {len(all_documents)} docs took {time.time() - start:3.2f} seconds.")

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for dup in range(dupe_factor):
    print(f"*** Generating Duplicate {dup}/{dupe_factor} ***")
    for document_index in tqdm(range(len(all_documents))):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              mlm_prob, mask_tokens, vocab_words, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        mlm_prob, mask_tokens, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, mlm_prob, mask_tokens, vocab_words, rng, max_seq_length)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, mlm_prob,
                                 mask_tokens, vocab_words, rng,
                                 max_seq_length):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (args.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(mask_tokens,
                       max(1, int(round(len(tokens) * mlm_prob))))
  # MAJOR CHANGE: Make sure that the desired number of mask_tokens is used
  # to prevent issues with too many tokens needed. Ie
  # sequence_length 384, mask_tokens 60. if num_to_predict is 58
  # and len(tokens) is 384 then there will be 326 sequence_tokens with 60 additional tokens
  # reserved for mask tokens. This will give a total tokens of 386 which is too many.
  num_to_predict = max(num_to_predict, len(tokens) - max_seq_length + mask_tokens)

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(args):
  tokenizer = tokenization.FullTokenizer(
      vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

  input_files_glob = []
  for input_pattern in args.input_file.split(","):
    input_files_glob.extend(sorted(glob.glob(input_pattern)))

  rng = random.Random(args.seed)
  num_files = len(input_files_glob)
  print(f"*** Reading {num_files} input files in batches of {args.max_open_files}***")
  for i in tqdm(range(0, num_files, args.max_open_files)):
    input_files = input_files_glob[i:min(i + args.max_open_files, num_files)]
    for input_file in input_files:
      print(f"processing:  {input_file}")

    instances = create_training_instances(input_files, tokenizer, args.sequence_length,
                                          args.duplication_factor, args.short_seq_prob,
                                          args.mlm_prob, args.mask_tokens, rng)

    output_files = [args.output_file + f"_{i//args.max_open_files}"]
    print("*** Writing to output files ***")

    write_instance_to_example_files(instances, tokenizer, args.sequence_length,
                                    args.mask_tokens, output_files, args, args.max_samples)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    "PreTraining Data for Mlm/Next Sentence prediction")
  parser.add_argument("--input-file", type=str, required=True)
  parser.add_argument("--output-file", type=str, required=True)
  parser.add_argument("--vocab-file", type=str, required=True)
  parser.add_argument("--do-lower-case", action="store_true")
  parser.add_argument("--sequence-length", type=int, default=128)
  parser.add_argument("--mask-tokens", type=int, default=20)
  parser.add_argument("--seed", type=int, default=1984)
  parser.add_argument("--duplication-factor", type=int, default=6)
  parser.add_argument("--mlm-prob", type=float, default=0.15)
  parser.add_argument("--short-seq-prob", type=float, default=0.1)
  parser.add_argument("--max-samples", type=int, default=-1)
  parser.add_argument("--pad-position-value", type=int, default=384,
                      help="Value in the positional input for [PAD] tokens")
  parser.add_argument("--do-whole-word-mask", type=bool, default=False)
  parser.add_argument("--dont-rearrange-mlm-tokens-to-front", action="store_true")
  parser.add_argument("--max-open-files", type=int, default=1)
  args = parser.parse_args()
  main(args)
