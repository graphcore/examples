#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import transformers
import torch
import poptorch


import argparse


def get_options():
    parser = argparse.ArgumentParser(
        description='Run BERT inference on IPU using the PopTorch framework.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--batch-size',
        default=2,
        type=int,
        help="Batch size. Since this model contains a two-stage pipeline the minimum batch size is 2."
    )
    parser.add_argument(
        '--context-file',
        default="context.txt",
        type=argparse.FileType('r'),
        help='File containing the "context" input information.'
    )
    parser.add_argument(
        '--questions-file',
        default="questions.txt",
        type=argparse.FileType('r'),
        help='File containing the "questions" inputs, each question must be on a new line.'
    )
    return parser.parse_args()


def read_inputs(options):
    context = options.context_file.read()
    questions = options.questions_file.readlines()
    questions = [q.rstrip() for q in questions]

    # Pad last batch with empty question if required
    questions += [""] * (len(questions) % options.batch_size)
    return context, questions


if __name__ == '__main__':
    # Parse command line arguments.
    options = get_options()

    # Pre-trained model and tokenizer.
    tokenizer = transformers.BertTokenizer.from_pretrained(
        'mrm8488/bert-medium-finetuned-squadv2', return_token_type_ids=True)
    model = transformers.BertForQuestionAnswering.from_pretrained(
        'mrm8488/bert-medium-finetuned-squadv2')

    # Parse command-line arguments.
    context, questions = read_inputs(options)

    num_questions = len(questions)
    batch_size = options.batch_size
    num_batches = num_questions // batch_size

    # Pipeline the model over two IPUs. You must have at least as many batches (questions) as you have IPUs.
    model.bert.embeddings.position_embeddings = poptorch.BeginBlock(
        layer_to_call=model.bert.embeddings.position_embeddings, ipu_id=1)


    # Wrap PyTorch model insde a PopTorch InferenceModel. This will make the model run on the IPU.
    opts = poptorch.Options().deviceIterations(batch_size)
    inference_model = poptorch.inferenceModel(model, options=opts)

    # Process inputs in batches.
    for batch_idx in range(num_batches):
        input_pairs = [
            (questions[batch_size*batch_idx + i], context)
            for i in range(batch_size)]

        batched_encoding = tokenizer.batch_encode_plus(
            input_pairs,
            max_length=110,
            pad_to_max_length='right'
        )

        # Convert to PyTorch tensors.
        input_batch = torch.tensor(batched_encoding["input_ids"])
        attention_batch = torch.tensor(batched_encoding["attention_mask"])

        # Execute on IPU.
        start_score_pop, end_scores_pop = inference_model(
            input_batch, attention_batch)

        # Process outputs.
        for i, (start_score, end_score) in enumerate(zip(start_score_pop, end_scores_pop)):
            answer_start, answer_stop = start_score.argmax(), end_score.argmax()
            answer_ids = input_batch[i][answer_start:answer_stop + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids,
                                                            skip_special_tokens=True)
            answer = tokenizer.convert_tokens_to_string(answer_tokens)

            print(f"Question: {questions[batch_size*batch_idx + i]}")
            print(f"Answer: {answer}")
