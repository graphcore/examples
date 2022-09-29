#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizer

import popxl

from config import BertConfig
from utils.checkpoint import hf_mapping
from squad_inference import squad_inference_phased
from modelling.squad import BertSquadHead

torch.manual_seed(42)

# --- Data ---
context = "Scotland is a country that is part of the United Kingdom. Covering the northern third of the island of " \
          "Great Britain, mainland Scotland has a 96 mile (154 km) border with England to the southeast and is " \
          "otherwise surrounded by the Atlantic Ocean to the north and west, the North Sea to the northeast and the " \
          "Irish Sea to the south. In addition, Scotland includes more than 790 islands; principally within the " \
          "Northern Isles and the Hebrides archipelagos. "
QandA = [
    ("How many islands are there in Scotland?", "more than 790"),
    ("What sea is to the south of Scotland?", "irish sea"),
    ("How long is Scotland's border in km?", "154"),
    ("Where is England in relation to scotland?", "southeast")
]
questions, answers = zip(*QandA)

# --- HF example ---
tokenizer = BertTokenizer.from_pretrained('csarron/bert-base-uncased-squad-v1')
hf_model = BertForQuestionAnswering.from_pretrained(
    'csarron/bert-base-uncased-squad-v1')


def answer_from_logits(start_logits, end_logits, tokens):
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)

    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]

    return answer


def bert_answer_hf(Q, context):
    input_ids = tokenizer.encode(Q, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    hf_output = hf_model(torch.tensor([input_ids]))  # type: ignore
    answer = answer_from_logits(
        hf_output.start_logits, hf_output.end_logits, tokens)
    return answer


print("HUGGINGFACE OUTPUT")
for Q, A in QandA:
    answer = bert_answer_hf(Q, context)
    print(f"Question: {Q:45} Truth: {A:10}\t BERT: {answer}")

# --- popxl example ---
config = BertConfig()
config.model.sequence_length = 128
config.model.hidden_size = hf_model.config.hidden_size
config.model.attention.heads = hf_model.config.num_attention_heads
config.model.layers = hf_model.config.num_hidden_layers
config.model.embedding.vocab_size = hf_model.config.vocab_size
config.model.embedding.max_positional_length = hf_model.config.max_position_embeddings
config.model.dtype = popxl.float32

config.execution.micro_batch_size = len(questions)

# Prep data
words = []
positions = []
token_types = []
masks = []
tokens = []
for question_i in questions:
    output = tokenizer.encode_plus(question_i, context, max_length=config.model.sequence_length, padding='max_length',
                                   return_tensors='np')
    tokens.append(tokenizer.convert_ids_to_tokens(output.input_ids.flatten()))
    words.append(output.input_ids.astype('uint32'))
    token_types.append(output.token_type_ids.astype('uint32'))
    masks.append(((output.attention_mask - 1)*1000).astype('float32'))
input_data = [np.asarray(words).flatten(), np.asarray(token_types).flatten(), np.asarray(
    masks).flatten()]  # this is different from other demo because BertLayers reshape input

session = squad_inference_phased(config)

weights = hf_mapping(config, session, hf_model.bert)
weights.update(BertSquadHead.hf_mapping(config, session.state.squad, hf_model.qa_outputs))
session.write_variables_data(weights)

print("PopXL OUTPUT")
inputs = dict(zip(session.inputs, input_data))
with session:
    outputs = session.run(inputs)[session.outputs[0]]
for Q, A, tokens_i, logits in zip(questions, answers, tokens, outputs):
    start_logits, end_logits = np.split(logits, 2, -1)
    answer = answer_from_logits(torch.tensor(
        start_logits), torch.tensor(end_logits), tokens_i)
    print(f"Question: {Q:45} Truth: {A:10}\t BERT: {answer}")
