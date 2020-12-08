# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import logging
import os
import sys
from multiprocessing import Pool
import numpy as np
import download_utils
import utils


logger = logging.getLogger("dataset-encode")


def encode(encoder, lines, seq_len, num_processes=48):
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            encoded = pool.map(encoder.encode, lines)
    else:
        encoded = [encoder.encode(l) for l in lines]

    num_seqs = len(encoded)

    encoded = [x[:seq_len] for x in encoded if len(x) >= seq_len]
    num_after = len(encoded)
    discarded = num_seqs - num_after
    logger.debug(f"Encoded {num_after} articles.")
    logger.debug(f"Discarded {discarded} articles.")
    logger.debug(f"Kept {100 * num_after / num_seqs}% of the dataset")
    return encoded


def load_articles(file_iterator):
    articles = []
    was_empty = False
    current_article = ""
    for line in file_iterator:
        stripped_line = line.strip()
        is_empty = len(stripped_line) == 0 or stripped_line[0] == "="

        if was_empty and is_empty and current_article != "":
            articles.append(current_article)
            current_article = ""
        elif not is_empty:
            current_article += line

        was_empty = is_empty
    if current_article != "":
        articles.append(current_article)
    return articles


def encode_and_save(args):
    input_path = dataset_path / f"wiki.{p}.raw"
    output_path = dataset_path / f"seqlen_{args.sequence_length}.{p}.cache.np_{np.__version__}"

    logger.info("Loading articles...")
    with open(input_path) as fh:
        articles = load_articles(fh)

    logger.info("Encoding...")
    encoded = encode(encoder, articles, args.sequence_length, num_processes=args.num_processes)

    logger.info(f"Writing to {output_path}...")
    np.save(output_path, encoded)

    logger.info("Done")

    # Return the max token to get the length of vocabulary
    return np.max(encoded)


def get_encoder(gpt2_path):
    try:
        sys.path.append(gpt2_path)
        import src.encoder as encoder

        # The encoder assumes that the models are available in it's cwd.
        cwd = os.getcwd()
        os.chdir(gpt2_path)
        enc = encoder.get_encoder("124M", "models")
        os.chdir(cwd)
        return enc
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("Could not locate GPT-2 Repository. Please clone from "
                                  "https://github.com/openai/gpt-2 and provide the path "
                                  "using the --gpt2-repo-path flag.") from err


if __name__ == "__main__":
    args = utils.parse_args()

    logger = utils.setup_logger()

    dataset_path = download_utils.get_dataset_path(
        args.dataset_name, args.dataset_dir)

    if not dataset_path.exists():
        download_utils.download_and_extract(
            args.dataset_name, args.dataset_dir)

    dataset_parts = ["train", "test", "valid"]

    encoder = get_encoder(args.gpt2_repo_path)

    vocab_size = 0
    for p in dataset_parts:
        logger.info(f"Encoding {args.dataset_name}.{p}, sequence-length {args.sequence_length}")
        vocab_size = np.max([encode_and_save(args), vocab_size])

    if not args.disable_create_vocab:
        vocab_path = dataset_path / f"seqlen_{args.sequence_length}.vocab"
        logger.info(f"Writing the vocab to {vocab_path}")
        text_vocab = [encoder.decode([word_token]) for word_token in range(vocab_size)]
        np.save(vocab_path, text_vocab)
