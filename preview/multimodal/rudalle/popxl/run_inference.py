# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import logging

import numpy as np

from config import InferenceConfig
from inference import init, postprocess, preprocess


def main(text):
    # config
    config = InferenceConfig()

    session, inputs_host_steam, out_stream, word_offsets = init(config)

    words_offsetted, _ = preprocess(config, text)

    logging.info("Attach to IPUs")

    seed = np.array([np.random.randint(0, np.iinfo(np.uint32).max),
                     np.random.randint(0, np.iinfo(np.uint32).max)],
                    dtype=np.uint)
    seed = np.repeat(seed.reshape(1, -1), words_offsetted.shape[0], axis=0)
    input_data = [words_offsetted, word_offsets, seed]
    inputs = dict(zip(inputs_host_steam, input_data))

    with session:
        outputs_popxl = session.run(inputs)

    logging.info("popxl output")
    postprocess(outputs_popxl, out_stream, filename=(text+".png")[:255])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='The sunrise', help='The input text')
    args = parser.parse_args()

    try:
        main(args.text)
    except Exception as e:
        logging.exception(e)  # Log time of exception
        raise
