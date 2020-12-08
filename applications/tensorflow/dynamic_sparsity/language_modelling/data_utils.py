# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import numpy as np
import logging
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()

logger = logging.getLogger(os.path.basename(__file__))


def load_wiki_data(path, seq_len, vocab_size, training=True):
    logger.info("Loading dataset")
    directory = "seqlen_" + str(seq_len) + "/"
    filename = "seqlen_" + str(seq_len) + "." + ("train" if training else "test") + ".cache.np_1.19.1.npy"
    dataset_path = os.path.join(path, directory + filename)
    data = np.load(dataset_path)
    # process the data so that it fits the required vocab size. All extra tokens will be set to the last
    # token
    input_encodings = np.array([[datum if datum < vocab_size - 1 else vocab_size - 1 for datum in sequence]
                               for sequence in data]).astype(np.int32)

    vocab_file_name = "seqlen_" + str(seq_len) + "/" + "seqlen_" + str(seq_len) + ".vocab.npy"
    vocab_path = os.path.join(path, vocab_file_name)

    logger.info("Loading vocab file")
    if os.path.isfile(vocab_path):
        vocab = np.load(vocab_path)
        if vocab_size > len(vocab):
            logging.error(f"The vocab file is smaller than the requested vocab size."
                          f"Expected at least {vocab_size} but got {len(vocab)}")
        # Keep only the wanted vocab size
        vocab = vocab[0: vocab_size - 1].tolist()
        # Set the last token as unknown
        vocab.append("UNK")
    else:
        vocab = None
        logger.error("Vocab file could not be found")
    return input_encodings, vocab


def make_synthetic_data(n_samples, opts):
    logger.info(f"Generating random dummy data")
    bos_id = opts.source_bos_id
    eos_id = opts.source_eos_id
    pad_id = opts.source_pad_id
    S = opts.source_sequence_length
    V = opts.source_vocab_length
    P = 0  # padding tokens
    bos = np.ones([n_samples, 1], dtype=np.int32) * bos_id
    text = np.random.randint(0, V, size=(n_samples, S - P - 2)).astype(np.int32)
    eos = np.ones([n_samples, 1], dtype=np.int32) * eos_id
    pad = np.ones([n_samples, P], dtype=np.int32) * pad_id

    return np.concatenate([bos, text, eos, pad], axis=-1).astype(np.int32)


def make_dataset(opts, use_synthetic_data=True, training=True):
    if use_synthetic_data:
        input_encodings = make_synthetic_data((opts.repeat_count * 2 *
                                              (opts.gradient_accumulation_count
                                               if opts.pipeline else 1)) if training
                                              else 1000, opts)
        vocab = None
    else:
        input_encodings, vocab = load_wiki_data(opts.data_dir, opts.source_sequence_length, opts.source_vocab_length, training)

    logger.info(f"Loaded dataset containing {len(input_encodings)} sequences.")

    # Make dataset from encodings
    with tf.device('cpu'):
        dataset = tf.data.Dataset.from_tensor_slices((input_encodings))
        if not opts.disable_dataset_cache:
            dataset = dataset.cache()

        if opts.shuffle:
            buffer_size = (
                opts.shuffle_buffer_size
                if 'shuffle_buffer_size' in opts else
                len(input_encodings))
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.repeat()
        dataset = dataset.batch(opts.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(opts.gradient_accumulation_count * opts.repeat_count
                                   if opts.pipeline else opts.repeat_count)
    return dataset, len(input_encodings), vocab


def decode_prediction(prediction, target, vocab):
    if vocab is None:
        logger.error("Vocabulary is None. Could not decode prediction")
        return

    text_pred = ''.join([vocab[int(token)] + " " for token in prediction[0]])
    text_target = ''.join([vocab[int(token)] + " " for token in target[0]])

    return text_pred, text_target
