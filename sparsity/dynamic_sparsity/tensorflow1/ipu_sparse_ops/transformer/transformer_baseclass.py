# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import json
import os
from argparse import ArgumentParser, Action
from logging import getLogger
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope

tf.disable_v2_behavior()
logger = getLogger(os.path.basename(__file__))


class NullContextManager(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class TransformerOptions(ArgumentParser):
    """
    The TransformerOptions are a subclass of ArgumentParser which is
    better than a plain dict or namedtuple since you can type print_help()
    or print_usage() to discover what options are available and how they are
    used.
    Any argument the developer includes here in the future will be populated
    in the transformer also. Note that the - char change to _ for attributes
    """

    def __init__(self):
        ArgumentParser.__init__(self)
        TransformerOptions.add_all_arguments(self)

    @staticmethod
    def dtype_action():
        class dtypeAction(Action):
            def __call__(self, parser, namespace, value, option_string=None):
                setattr(namespace, self.dest, tf.as_dtype(value))
        return dtypeAction

    @staticmethod
    def add_all_arguments(parser):
        # We add all args outside the initialize to make it easy to
        # add these to any arbitrary parser
        parser.add_argument('--dtype', default='float32',
                            choices=["float32", "float16"],
                            action=TransformerOptions.dtype_action(),
                            help='The floating point precision to use.')
        parser.add_argument('--random-seed', default=42, type=int,
                            help='Set the random seed for the model.')
        parser.add_argument('--batch-size', type=int, default=1,
                            help='Set (machine aka micro batch) batch size.')
        parser.add_argument("--num-shards", type=int, default=1)
        parser.add_argument('--encoder-layers', type=int, default=2,
                            help='Number of encoder layers.')
        parser.add_argument('--decoder-layers', type=int, default=2,
                            help='Number of decoder layers.')
        parser.add_argument('--attention-heads', type=int, default=1,
                            help='Number of attention heads in each attention block.')
        parser.add_argument('--hidden-length', type=int, default=64,
                            help='The size of the embedding dimension.')
        parser.add_argument('--qkv-length', type=int, default=128,
                            help='The size of the attention embedding dimension.')
        parser.add_argument('--ff-length', type=int, default=256,
                            help="Typically 4x of the hidden dimension.")
        parser.add_argument('--embedding-length', type=int, default=32,
                            help='The size of the embedding dimension. Typically less than the hidden size')
        parser.add_argument("--source-sequence-length", type=int, default=128,
                            help="The size of the source sequence dimension")
        parser.add_argument("--target-sequence-length", type=int, default=128,
                            help="The size of the target sequence dimension")
        parser.add_argument("--source-vocab-length", type=int, default=None,
                            help="Size of the source vocabulary")
        parser.add_argument("--target-vocab-length", type=int, default=None,
                            help="Size of the target vocabulary")
        parser.add_argument("--target-bos-id", type=int, default=None,
                            help="Beginning of sentence token id in target dict")
        parser.add_argument("--target-pad-id", type=int, default=None,
                            help="PAD token id in target dict")
        parser.add_argument("--target-eos-id", type=int, default=None,
                            help="End of sentence token id in target dict")
        parser.add_argument("--source-pad-id", type=int, default=None,
                            help="PAD token id in source dict")
        parser.add_argument("--exclude-embedding", action="store_true",
                            help="Remove the gather lookups for the source and target")
        parser.add_argument("--include-projection-bias", action='store_true', help="Add bias to final projection")
        parser.add_argument("--dropout-keep-prob", default=0.9, type=float,
                            help="This dropout keep prob applies universally to all dropouts in the transformer")
        parser.add_argument("--config-file", type=str,
                            help="The .json file from which to populate TransfromerOptions")

        group = parser.add_argument_group("Attention options")
        group.add_argument("--exclude-attention-biases", action="store_false",
                           help="Remove biases to attention layer linear projections")
        group.add_argument("--disable-concat-qkv", action="store_true",
                           help="Forces to run q, k and v matmuls as 3 separate passes")
        group.add_argument("--use-static-sparse-autoregressive-attention", action="store_true",
                           help="This will compute the autoregressive q@kT interaction matrix in a sparse way, saving memory and compute")

        parser.add_argument("--outline-encoder", action='store_true', help="Enable per-encoder outlining.")
        parser.add_argument("--no-outline-dropout", action='store_true', help="Disable outlining of dropout.")
        parser.add_argument("--encoder-stage-dense-matmul-options", type=json.loads,
                            default=None,
                            help="Can be used to override the default dense matmul options in encoder stages.")
        parser.add_argument("--projection-stage-dense-matmul-options", type=json.loads,
                            default=None,
                            help="Can be used to override the default dense matmul options in the projection stage.")

    def set_from_dict(self, **kwargs):
        """
        For those wishing to define the model using a dictionary of parameters
        this provides a workaround.
        """
        args_strings = []
        for key, value in kwargs:
            args_strings += ["--" + key.replace("_", "-") + "=" + value]
        return self.parse_args(args_strings)


class Transformer(object):
    def __init__(self, transformerOpts):
        # If a config file is provided, use it to set the defaults
        # Load the model configuration
        if transformerOpts.config_file is not None:
            logger.info(f"Reading config file... {transformerOpts.config_file}")
            with open(transformerOpts.config_file, "r") as f:
                presets = json.load(f)
            # Compare presets to parsed args
            for key, value in presets.items():
                if key in ["config_file"]:
                    continue
                elif key not in vars(transformerOpts):
                    setattr(transformerOpts, key, value)
                else:
                    old_value = getattr(transformerOpts, key)
                    if old_value != value:
                        logger.info(f"Overriding TransformerOption {key} from value {old_value} to value {value} from specified config file.")
                        setattr(transformerOpts, key, value)

        # Populate the transformer properties from the
        # transformer options argparse
        for field, value in vars(transformerOpts).items():
            setattr(self, field, value)

        # If we haven't defined the target vocab length,
        # we can use the source vocab length instead,
        # especially considering we are using tied embeddings
        if self.target_vocab_length is None:
            if self.exclude_embedding:
                raise Exception("When not using embeddings, the target vocab size must not be None.")
            self.target_vocab_length = self.source_vocab_length

    def __call__(*args, **kwargs):
        raise ValueError("The Transformer class is not callable -> \n"
                         "i.e. use transformer.seq2seq or transformer.language_model instead")

    def seq2seq(self, source, target, source_mask=None, target_mask=None):
        """
        Builds a seq2seq transformer
        """
        # Manual sharding scheme (for n layers)
        # 0: embeddings, projection
        # 1: encoder_layers/(n-1) and decoder_layers/(n-1)
        # ...
        # n  encoder_layers/(n-1) and decoder_layers/(n-1)
        if self.num_shards > 1:
            per_ipu = int(self.encoder_layers / (self.num_shards - 1) + 0.5)
            self.encoder_layers_placement = [i // per_ipu for i in range(self.encoder_layers)]
            per_ipu = int(self.decoder_layers / (self.num_shards - 1) + 0.5)
            self.decoder_layers_placement = [i // per_ipu for i in range(self.decoder_layers)]
        else:
            self.encoder_layers_placement = [0 for i in range(self.encoder_layers)]
            self.decoder_layers_placement = [0 for i in range(self.decoder_layers)]

        # Embedding
        # the tgt token look-up-table is reused in output
        # projection (aka tied embedding)
        with ipu.scopes.ipu_shard(0):
            if not self.exclude_embedding:
                source_embd, self.source_token_lut = self.embedding(source, True, 'encoder_')
                target_embd, self.target_token_lut = self.embedding(target, False, 'decoder_')
            else:
                source_embd = source
                target_embd = target
        self.tied_embedding = self.target_token_lut

        # Encoder
        encoder_out = source_embd
        for i in range(self.encoder_layers):
            with ipu.scopes.ipu_shard(self.encoder_layers_placement[i]):
                encoder_out = self.encoder_layer(encoder_out, source_mask, f'encoder_{i}')

        # Bottleneck
        # Pass along the normed output of the encoder
        with self.namescope("encoder_decoder_bottleneck"):
            with ipu.scopes.ipu_shard(0):
                normed_encoder_out = self.norm(encoder_out)

        # Decoder
        decoder_out = target_embd
        for i in range(self.decoder_layers):
            with ipu.scopes.ipu_shard(self.decoder_layers_placement[i]):
                decoder_out = self.decoder_layer(decoder_out, normed_encoder_out, target_mask, f'decoder_{i}')

        # Projection (logits prediction)
        with ipu.scopes.ipu_shard(0):
            with self.namescope("projection_prenorm"):
                normed_decoder_out = self.norm(decoder_out)
            x = self.projection(normed_decoder_out)
        return x

    def get_language_model_device_placement(self):
        # Manual sharding scheme (for n layers)
        # 0: embedding, projection
        # 1: encoder_layers/(n-1)
        # ...
        # n  encoder_layers/(n-1)
        if self.num_shards > 1:
            per_ipu = int(self.encoder_layers / (self.num_shards - 1) + 0.5)
            self.encoder_layers_placement = [i // per_ipu + 1 for i in range(self.encoder_layers)]
        else:
            self.encoder_layers_placement = [0 for i in range(self.encoder_layers)]

    def embedding_stage(self, source, source_mask, last_itr, *args, enable_dense_grad=True, sparse_embeddings=False, **kwargs):
        source_embd, self.source_token_lut = self.embedding(source, True,
                                                            compute_dense_grad=enable_dense_grad and last_itr,
                                                            sparse_embeddings=sparse_embeddings)
        self.tied_embedding = self.source_token_lut

        return (source_embd, source_mask, last_itr, *args, source)

    def encoder_stage(self, x, source_mask, last_itr, *args, enable_dense_grad=True, **kwargs):
        idx = kwargs.get('idx', 0)

        if self.outline_encoder:
            @ipu.outlined_function
            def f(x, source_mask, last_itr):
                return self.encoder_layer(x, source_mask,
                                          compute_dense_grad=enable_dense_grad and last_itr,
                                          debug_name=f'encoder_{idx}')

            encoder_out = f(x, source_mask, last_itr)
        else:
            encoder_out = self.encoder_layer(x, source_mask,
                                             compute_dense_grad=enable_dense_grad and last_itr,
                                             debug_name=f'encoder_{idx}')

        # we output encoder twice so that we have the same output as the projection layer
        return (encoder_out, source_mask, last_itr, *args)

    def projection_stage(self, encoder_out, source_mask, last_itr, *args, enable_dense_grad=True, sparse_embeddings=False, **kwargs):
        if self.exclude_embedding:
            raise Exception("The projection layer uses tied embeddings. Consider turning --exclude-embedding off")
        with self.namescope("projection_prenorm"):
            normed_encoder_out = self.norm(encoder_out)
            x = self.projection(normed_encoder_out, compute_dense_grad=enable_dense_grad and last_itr, sparse_embeddings=sparse_embeddings)
        return (x, source_mask, last_itr, *args)

    def language_model_stages(self, enable_dense_grad=True, add_projection_layer=True, sparse_embeddings=False):
        stages = []
        self.get_language_model_device_placement()

        device_mapping = [0]
        device_mapping.extend(self.encoder_layers_placement)

        stage_options = []

        if not self.exclude_embedding:
            stages.append(partial(self.embedding_stage, enable_dense_grad=enable_dense_grad, sparse_embeddings=sparse_embeddings))
            stage_options.append(ipu.pipelining_ops.PipelineStageOptions())
        else:
            raise Exception("Pipelined training excpects the first stage to be an embedding stage")

        for i in range(self.encoder_layers):
            stages.append(partial(self.encoder_stage, idx=i, enable_dense_grad=enable_dense_grad))
            stage_options.append(ipu.pipelining_ops.PipelineStageOptions(
                matmul_options=self.encoder_stage_dense_matmul_options))

        if add_projection_layer:
            device_mapping.append(0)
            stages.append(partial(self.projection_stage, enable_dense_grad=enable_dense_grad, sparse_embeddings=sparse_embeddings))
            stage_options.append(ipu.pipelining_ops.PipelineStageOptions(
                matmul_options=self.projection_stage_dense_matmul_options))

        return stages, device_mapping, stage_options

    def language_model(self, source, source_mask=None, add_projection_layer=True,
                       last_itr=False, enable_dense_grad=True, sparse_embeddings=False):
        """
        Builds a stack of encoder layers but no decoder layers. Typically used in language
        modelling applications e.g. BERT (no mask) or GPT (causal mask)
        """
        # Target seq has the same properties as input seq
        self.target_sequence_length = self.source_sequence_length
        self.target_vocab_length = self.source_vocab_length
        self.target_bos_id = self.source_bos_id
        self.target_pad_id = self.source_pad_id
        self.target_eos_id = self.source_eos_id

        self.get_language_model_device_placement()

        # Embedding
        # the src token look-up-table is reused in output
        # projection (aka tied embedding)
        if not self.exclude_embedding:
            with ipu.scopes.ipu_shard(0):
                source_embd, *_ = self.embedding_stage(source, source_mask=None,
                                                       last_itr=last_itr, enable_dense_grad=enable_dense_grad,
                                                       sparse_embeddings=sparse_embeddings)
        else:
            source_embd = source
            self.encoder_layers_placement = [mapping - 1 for mapping in self.encoder_layers_placement]

        # Encoder
        encoder_out = source_embd
        for i in range(self.encoder_layers):
            with ipu.scopes.ipu_shard(self.encoder_layers_placement[i]):
                encoder_out, *_ = self.encoder_stage(encoder_out, source_mask,
                                                     last_itr=last_itr,
                                                     idx=f'encoder_{i}',
                                                     enable_dense_grad=enable_dense_grad)

        # Projection (logits prediction)
        if add_projection_layer:
            with ipu.scopes.ipu_shard(0):
                logits, *_ = self.projection_stage(encoder_out,
                                                   source_mask=None,
                                                   last_itr=last_itr,
                                                   enable_dense_grad=enable_dense_grad,
                                                   sparse_embeddings=sparse_embeddings)
                return logits
        else:
            return encoder_out

    def encoder_layer(self, x, mask, compute_dense_grad=False, debug_name=''):
        with self.namescope(debug_name):
            residual = x
            x = self.norm(x)  # pre-norm
            x = self.attention(x, x, x, mask,
                               is_self_attention=True,
                               compute_dense_grad=compute_dense_grad)
            x = self.shortcut(x, residual, 'sc1')
            residual = x
            x = self.norm(x)  # pre-norm
            x = self.feed_forward(x, compute_dense_grad=compute_dense_grad)
            x = self.shortcut(x, residual, 'sc2')
        return x

    def decoder_layer(self, x, encoder_x, mask, debug_name=''):
        with self.namescope(debug_name):

            with self.namescope("self_attention"):
                residual = x
                x = self.norm(x)  # pre-norm
                x = self.attention(x, x, x, mask,
                                   is_self_attention=True)
                x = self.shortcut(x, residual, 'sc1')

            with self.namescope("encoder_attention"):
                residual = x
                x = self.norm(x)  # pre-norm
                x = self.attention(x, encoder_x, encoder_x, mask)
                x = self.shortcut(x, residual, 'sc2')

            with self.namescope("ffn_block"):
                residual = x
                x = self.norm(x)  # pre-norm
                x = self.feed_forward(x)
                x = self.shortcut(x, residual, 'sc3')
        return x

    def shortcut(self, x, residual, debug_name):
        with self.namescope(debug_name):
            x = self.dropout(x)
            x = self.add(x, residual)
        return x

    def reshape(self, tensor, newShape, debugPrefix=''):
        return tf.reshape(tensor, newShape)

    def embedding(self, x, src, compute_dense_grad=False, sparse_embeddings=False):  # x[batch_size, sequence_length] -> x[batch_size, sequence_length, embedding_length]
        raise NotImplementedError

    def projection(self, x, compute_dense_grad=False):  # x[batch_size, sequence_length, embedding_len] -> x[batch_size, sequence_length, target_vocab_length]
        raise NotImplementedError

    def add(self, x, y):  # -> x
        raise NotImplementedError

    def norm(self, x, length):  # -> x
        raise NotImplementedError

    def feed_forward(self, x, compute_dense_grad=False):  # -> x
        raise NotImplementedError

    def attention(self, in_q, in_k, in_v, mask=None, compute_dense_grad=False):  # in_qkv[batch_size,sequence_length, embedding_len] -> x[batch_size,sequence_length, embedding_len]
        raise NotImplementedError

    def position_encoder(self, x):  # -> x
        raise NotImplementedError

    def dropout(self, x):  # -> x
        if self.dropout_keep_prob is not None and self.dropout_keep_prob < 1:
            drop_rate = 1 - self.dropout_keep_prob
            if self.no_outline_dropout:
                x = ipu.rand_ops.dropout(x, rate=drop_rate,
                                         seed=np.array([0, self.random_seed],
                                                       dtype=np.int32))
                return x
            else:
                @ipu.outlined_function
                def f(x):
                    return ipu.rand_ops.dropout(x, rate=drop_rate,
                                                seed=np.array([0, self.random_seed],
                                                              dtype=np.int32))
                return f(x)
        else:
            return x

    def namescope(self, debug_string):
        return NullContextManager()
