# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="./datasets",
                        help="The path to the dataset directories. If the requested dataset "
                             "doesn't exist, it will be downloaded there.")
    parser.add_argument("--sequence-length", type=int, default=128,
                        help="Desired sequence length to use in the dataset. Longer sequences "
                             "will be cut based on token count. Shorter ones will be discarded")
    parser.add_argument("--num-processes", type=int, default=48,
                        help="The maximum number of processes to use while decoding sequences.")
    parser.add_argument("--dataset-name", type=str, default="wikitext-103",
                        choices=("wikitext-2", "wikitext-103", "wikitext-fake"),
                        help="The dataset to encode. Currently only wikitext-2 and 103 are "
                             "supported.")
    parser.add_argument("--gpt2-repo-path", required=True,
                        help="The path to the GPT-2 repo containing the encoder (clone from: "
                             "https://github.com/openai/gpt-2.")
    parser.add_argument("--disable-create-vocab", action="store_true", help="Disable creating a vocabulary file")

    return parser.parse_args()


def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dataset-encode")
    return logger
