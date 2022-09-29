# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import sys
from typing import Tuple
import logging
import popxl


class Config(object):
    def __init__(
            self,
            vocab_size=16384,
            image_vocab_size=8192,
            layers=24,
            n_embd=2048,
            n_head=16,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            attn_pdrop=0.1,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            text_seq_len=128,
            image_seq_len=1024,
            scale_attn_weights=True,
            attn_factor=1,
            eval=True,
            loss_img_weight=7,
            loss_scaling=1,
            dtype=popxl.float16,
    ):
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size
        self.total_vocab_size = vocab_size + image_vocab_size
        self.layers = layers
        self.n_embd = n_embd
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.seq_len = text_seq_len + image_seq_len
        self.image_tokens_per_dim = int(image_seq_len**0.5)
        self.n_positions = text_seq_len + 1 + self.image_tokens_per_dim*2
        self.scale_attn_weights = scale_attn_weights
        self.attn_factor = attn_factor
        self.eval = eval
        self.loss_img_weight = loss_img_weight
        self.loss_scaling = loss_scaling
        self.dtype = dtype


def logging_setup():
    """Setup logging"""
    log_level = os.environ.get('APP_LOG_LEVEL', 'INFO')
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info(f"Staring. Process id: {os.getpid()}")


class InferenceConfig(Config):
    def __init__(self,
                 micro_batch_size=1,
                 replication=1,
                 device_iterations=1,
                 ipus=8,
                 available_memory_proportion: Tuple[float, ...] = (0.6,),
                 top_k=2048,
                 top_p=0.99,
                 ):
        super().__init__()
        self.micro_batch_size = micro_batch_size
        self.device_iterations = device_iterations
        self.replication = replication
        self.ipus = ipus
        self.available_memory_proportion = available_memory_proportion
        self.scale_attn_weights = True
        self.input_len = self.text_seq_len
        self.output_len = self.image_seq_len
        self.total_seq_len = self.text_seq_len + self.image_seq_len
        self.num_text_tokens = self.vocab_size + self.text_seq_len
        self.n_positions = self.text_seq_len + 1 + self.image_tokens_per_dim**2
        self.total_vocab_size = self.vocab_size + self.text_seq_len + self.image_vocab_size
        self.top_k = top_k
        self.top_p = top_p

        # Parameters of image decoder
        self.z_channel = 256
        self.image_size = 256
        self.channels = [512, 256, 128, 128]
        self.n_blocks = 3

        logging_setup()
