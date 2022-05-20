# coding=utf-8
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. limitations under the License.

import argparse
import logging
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow import pywrap_tensorflow


def filter_optimizer(variables_name):
    optimizer_names = ["adam", "momentum", "adamw", "momentum", "lamb", 'accum']
    flag = False
    for name in optimizer_names:
        if name.lower() in variables_name.lower():
            flag = True
    return flag


def truncate_vocab(embeddings, required_vocab_size):
    vocab_size, _hidden_size = embeddings.shape
    if vocab_size != required_vocab_size:
        logging.info(f"Truncating embeddings from vocab size {vocab_size} to {required_vocab_size}.")
    return embeddings[:required_vocab_size, :]


def is_qkv_tensor(tensor_name):
    qkv_tensors = ["attention/self/key", "attention/self/query", "attention/self/value"]
    for n in qkv_tensors:
        if n in tensor_name:
            return True
    return False


def convert_apps_ckpt_to_research(
    ckpt_file,
    output_dir,
    num_embed_split,
    vocab_size,
    use_attention_bias,
    use_qkv_bias,
    use_cls_layer,
    baseline,
    dtype
):
    saved_variables = []
    qkv = {'query': {}, 'key': {}, 'value': {}}
    glu_weight = {'value': {}, 'gate': {}}
    glu_bias = {'value': {}, 'gate': {}}

    def add_variable(old_tensor, new_tensor):
        logging.info(f"{old_tensor} -> {new_tensor}")
        saved_variables.append(new_tensor)

    graph = tf.Graph()
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    with graph.as_default():
        sess = tf.compat.v1.Session()
        for old_tensor_name in sorted(var_to_shape_map):
            # Filter out the optimizer variables
            if 'global_step' in old_tensor_name:
                continue
            if filter_optimizer(old_tensor_name):
                continue
            if not use_cls_layer and "transform" in old_tensor_name:
                logging.info("Discarding dense layer before MLM loss.")
                continue
            if not use_attention_bias and "output/dense/bias" in old_tensor_name:
                logging.info("Discarding attention biases.")
                continue

            this_tensor_dtype = tf.float16
            if 'cls/predictions/output_bias' in old_tensor_name:
                this_tensor_dtype = tf.float32

            tensor_value = tf.cast(reader.get_tensor(old_tensor_name), dtype=this_tensor_dtype)

            new_name = old_tensor_name

            new_name = 'all/' + new_name

            if "Norm" in new_name:
                new_name = new_name.replace("LayerNorm", "GroupNorm")

            if "/layer_" in new_name and "encoder" in new_name:

                if new_name.endswith('kernel') and '/dwconv/' not in new_name:
                    new_name = new_name.replace("dense/kernel", "weight")
                    new_name = new_name.replace("kernel", "weight")
                if new_name.endswith('dense/bias'):
                    new_name = new_name.replace("dense/bias", "bias")

                if '/feed_forward_' in new_name:
                    new_name = new_name.replace("feed_forward_", "boom")

                    if '/intermediate/' in new_name:
                        new_name = new_name.replace("/intermediate/", "/up/")
                    if '/output/mixer/' in new_name:
                        new_name = new_name.replace("/output/mixer/", "/mixer/")
                    if '/output/' in new_name and '/output/mixer/' not in new_name:
                        new_name = new_name.replace("/output/", "/down/")

                    if baseline:
                        raise NotImplementedError

                if '/attention/' in new_name:
                    new_name = new_name.replace("attention/projection", "attention/output")
                    new_name = new_name.replace("/self/", "/qkv/")
                    layer_num = int(new_name.split('layer_')[1].split('/')[0])
                    if '/qkv/' in new_name and 'weight' in new_name:
                        if 'query' in new_name:
                            qkv['query'][layer_num] = tensor_value
                        if 'key' in new_name:
                            qkv['key'][layer_num] = tensor_value
                        if 'value' in new_name:
                            qkv['value'][layer_num] = tensor_value
                        continue

                    if '/qkv/' in new_name and '/key/bias' in new_name:
                        continue

                    if baseline:
                        raise NotImplementedError


                if '/convolution/' in new_name:
                    new_name = new_name.replace("/convolution/", "/conv/")
                    layer_num = int(new_name.split('layer_')[1].split('/')[0])

                    if "/pre/glu/" in new_name:
                        if '/values/' in new_name and 'weight' in new_name:
                            glu_weight['value'][layer_num] = tensor_value
                        if '/values/' in new_name and 'bias' in new_name:
                            glu_bias['value'][layer_num] = tensor_value
                        if '/gates/' in new_name and 'weight' in new_name:
                            glu_weight['gate'][layer_num] = tensor_value
                        if '/gates/' in new_name and 'bias' in new_name:
                            glu_bias['gate'][layer_num] = tensor_value
                        continue

            elif new_name.startswith("all/GroupNorm"):
                new_name = new_name.replace("all/", "all/bert/encoder/post_layers/")

            elif "word_embeddings" in new_name and num_embed_split == 2:
                vocab_dim = tensor_value.shape[0] // 2
                top = tensor_value[:vocab_dim]
                bottom = tensor_value[vocab_dim:]
                top_name = new_name.replace("/word_embeddings", "/s0/word_embeddings")
                bottom_name = new_name.replace("/word_embeddings", "/s1/word_embeddings")
                top_var = tf.Variable(top, name=top_name)
                bottom_var = tf.Variable(bottom, name=bottom_name)
                add_variable(old_tensor_name, top_var)
                add_variable(old_tensor_name, bottom_var)
                continue

            new_var = tf.Variable(tensor_value, name=new_name)
            add_variable(old_tensor_name, new_var)

        for n in range(len(qkv['query'])):
            qkv_var = tf.concat([qkv['query'][n], qkv['key'][n], qkv['value'][n]], axis=-1)
            new_var = tf.Variable(qkv_var, name=f"all/bert/encoder/layer_{n}/attention/qkv/weight")
            add_variable(qkv['query'][n], new_var)

        if not baseline:
            for n in range(len(glu_weight['value'])):
                glu_weight_tensor = tf.concat([glu_weight['value'][n], glu_weight['gate'][n]], axis=-1)
                glu_bias_tensor = tf.concat([glu_bias['value'][n], glu_bias['gate'][n]], axis=-1)
                glu_weight_var = tf.Variable(glu_weight_tensor, name=f"all/bert/encoder/layer_{n}/conv/pre/weight")
                glu_bias_var = tf.Variable(glu_bias_tensor, name=f"all/bert/encoder/layer_{n}/conv/pre/bias")
                add_variable(glu_weight['value'][n], glu_weight_var)
                add_variable(glu_bias['value'][n], glu_bias_var)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        _dir_name, ckpt_name = os.path.split(ckpt_file)
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)

        num_params = np.sum([np.prod(v.shape) for v in saved_variables])
        print(f"Number of parameters saved: {num_params}")


def convert_research_ckpt_to_apps(
    ckpt_file,
    output_dir,
    num_embed_split,
    vocab_size,
    use_attention_bias,
    use_qkv_bias,
    use_cls_layer,
    baseline,
    dtype,
):

    saved_variables = []
    split_embeddings = []

    def add_variable(old_tensor, new_tensor):
        logging.info(f"{old_tensor} -> {new_tensor}")
        saved_variables.append(new_tensor)

    graph = tf.Graph()
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    with graph.as_default():
        sess = tf.compat.v1.Session()
        for old_tensor_name in sorted(var_to_shape_map):
            # Filter out the optimizer variables
            if 'global_step' in old_tensor_name:
                continue
            if filter_optimizer(old_tensor_name):
                continue
            if not use_cls_layer and "transform" in old_tensor_name:
                logging.info("Discarding dense layer before MLM loss.")
                continue
            if not use_attention_bias and "output/dense/bias" in old_tensor_name:
                logging.info("Discarding attention biases.")
                continue


            this_tensor_dtype = dtype

            tensor_value = tf.cast(reader.get_tensor(old_tensor_name), dtype=this_tensor_dtype)

            new_name = old_tensor_name

            if new_name.startswith('all/'):
                new_name = new_name[4:]

            if "Norm" in new_name:
                new_name = new_name.replace("GroupNorm", "LayerNorm")

            if "/layer_" in new_name and "encoder" in new_name:

                if new_name.endswith('weight') and '/dwconv/' not in new_name:
                    new_name = new_name.replace("weight", "dense/kernel")
                if new_name.endswith('bias'):
                    new_name = new_name.replace("bias", "dense/bias")

                if '/boom' in new_name:
                    new_name = new_name.replace("boom", "feed_forward_")

                    if '/up/' in new_name:
                        new_name = new_name.replace("/up/", "/intermediate/")
                    if '/down/' in new_name:
                        new_name = new_name.replace("/down/", "/output/")
                    if '/mixer/' in new_name:
                        new_name = new_name.replace("/mixer/", "/output/mixer/")

                    if baseline:
                        new_name = new_name.replace("/feed_forward_", "")
                        new_name = new_name.replace("/postnorm/", "/output/")

                if '/conv/' in new_name:
                    new_name = new_name.replace("/conv/", "/convolution/")

                    if "/pre/" in new_name:
                        hidden_dim = tensor_value.shape[-1] // 2
                        values = tensor_value[..., :hidden_dim]
                        gates = tensor_value[..., hidden_dim:]
                        values_name = new_name.replace("/pre/", "/pre/glu/values/")
                        gates_name = new_name.replace("/pre/", "/pre/glu/gates/")
                        values_var = tf.Variable(values, name=values_name)
                        gates_var = tf.Variable(gates, name=gates_name)
                        add_variable(old_tensor_name, values_var)
                        add_variable(old_tensor_name, gates_var)
                        continue


                if '/attention/' in new_name:
                    new_name = new_name.replace("attention/output", "attention/projection")
                    new_name = new_name.replace("/qkv/", "/self/")
                    if '/self/' in new_name:
                        new_name = new_name.replace("dense/bias", "bias")
                        if 'self/dense/kernel' in new_name:
                            hidden_dim = tensor_value.shape[-1] // 3
                            q = tensor_value[..., :hidden_dim]
                            k = tensor_value[..., hidden_dim: 2 * hidden_dim]
                            v = tensor_value[..., 2 * hidden_dim: 3 * hidden_dim]
                            q_name = new_name.replace("/dense/kernel", "/query/kernel")
                            k_name = new_name.replace("/dense/kernel", "/key/kernel")
                            v_name = new_name.replace("/dense/kernel", "/value/kernel")
                            q_var = tf.Variable(q, name=q_name)
                            add_variable(old_tensor_name, q_var)
                            k_var = tf.Variable(k, name=k_name)
                            add_variable(old_tensor_name, k_var)
                            v_var = tf.Variable(v, name=v_name)
                            add_variable(old_tensor_name, v_var)
                            continue

                    if baseline:
                        new_name = new_name.replace("/postnorm/", "/projection/")

            elif 'encoder/post_layers' in new_name:
                new_name = new_name.replace("bert/encoder/post_layers/", "")

            elif "word_embeddings" in new_name:
                split_match = re.search("/s\d/", new_name)
                if split_match and num_embed_split == 1:
                    split_embeddings.append(tensor_value)
                    continue
                else:
                    pass

            new_var = tf.Variable(tensor_value, name=new_name)
            add_variable(old_tensor_name, new_var)

        if split_embeddings and num_embed_split == 1:
            merged_embedding = tf.concat(split_embeddings, axis=0)
            new_var = tf.Variable(merged_embedding, name="bert/embeddings/word_embeddings")
            add_variable(split_embeddings[0], new_var)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        _dir_name, ckpt_name = os.path.split(ckpt_file)
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)

        num_params = np.sum([np.prod(v.shape) for v in saved_variables])
        print(f"Number of parameters saved: {num_params}")


def convert_google_ckpt_to_gc(
    ckpt_file,
    output_dir,
    num_embed_split,
    vocab_size,
    use_attention_bias,
    use_qkv_bias,
    use_cls_layer,
    dtype,
):

    saved_variables = []

    def add_variable(old_tensor, new_tensor):
        logging.info(f"{old_tensor} -> {new_tensor}")
        saved_variables.append(new_tensor)

    graph = tf.Graph()
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        for tensor_name in sorted(var_to_shape_map):
            # Filter out the optimizer variables
            if filter_optimizer(tensor_name):
                continue
            if not use_cls_layer and "transform" in tensor_name:
                logging.info("Discarding dense layer before MLM loss.")
                continue
            if not use_attention_bias and "output/dense/bias" in tensor_name:
                logging.info("Discarding attention biases.")
                continue

            this_tensor_dtype = dtype
            if "cls/squad/" in tensor_name:
                # Keep SQuAD output dense layer weights as float32
                this_tensor_dtype = tf.float32
                tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=this_tensor_dtype)

            else:
                #  Cast all other tensors to required precision.
                tensor_value = tf.cast(
                    reader.get_tensor(tensor_name), dtype=this_tensor_dtype
                )

            if "word_embeddings" in tensor_name and num_embed_split > 1:
                # Split word_embeddings when num_split>1
                logging.info(
                    f"Splitting word embeddings info {num_embed_split} splits."
                )
                word_embeddings = truncate_vocab(tensor_value, vocab_size)
                hidden_size = np.shape(word_embeddings)[1]
                assert vocab_size % num_embed_split == 0
                size_per_slice = int(vocab_size / num_embed_split)
                for i in range(num_embed_split):
                    start_idx = i * size_per_slice
                    end_idx = (i + 1) * size_per_slice
                    we_pieces = tf.Variable(
                        word_embeddings[start_idx:end_idx, :],
                        shape=(size_per_slice, hidden_size),
                        name=f"bert/embeddings/s{i}/word_embeddings",
                    )
                    add_variable(tensor_name, we_pieces)

            #  Truncate word embeddings to  vocab_size
            elif "word_embeddings" in tensor_name:
                full_word_embeddings = tf.Variable(truncate_vocab(tensor_value, vocab_size), name=tensor_name)
                add_variable(tensor_name, full_word_embeddings)

            # Rename tensor
            elif "attention/output" in tensor_name:
                new_name = tensor_name.replace("attention/output", "attention/projection")
                proj = tf.Variable(tensor_value, name=new_name)
                add_variable(tensor_name, proj)

            elif is_qkv_tensor(tensor_name):
                # We will process self-attention parameters outside the loop
                continue

            else:
                others_var = tf.Variable(tensor_value, name=tensor_name)
                add_variable(tensor_name, others_var)

        # Concatenate or split QKV
        layer_re = re.compile('.*/layer_([0-9]+)/.*')
        matches = [layer_re.match(k) for k in var_to_shape_map.keys()]
        num_hidden_layers = max([int(m.group(1)) for m in matches if m is not None]) + 1

        logging.info("Concatenate query, key, value layers into one.")
        for i in range(num_hidden_layers):
            layer_name = f"bert/encoder/layer_{i}/attention/self"
            # Combine query,key,value to qkv_weight
            qkv_weight = []
            qkv_bias = []
            for name in ["query", "key", "value"]:
                weight_name = layer_name + f"/{name}/kernel"
                bias_name = layer_name + f"/{name}/bias"
                weight = tf.cast(reader.get_tensor(weight_name), dtype=dtype)
                bias = tf.cast(reader.get_tensor(bias_name), dtype=dtype)

                add_variable(weight_name, tf.Variable(weight, name=weight_name))

                # The QKV bias is always concantenated
                qkv_bias.append(bias)

            if use_qkv_bias:
                qkv_bias = tf.concat(qkv_bias, axis=0)
                qkv_b = tf.Variable(qkv_bias, shape=qkv_bias.shape, name=layer_name + "/qkv_bias")
                add_variable("qkv_bias", qkv_b)


        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        _dir_name, ckpt_name = os.path.split(ckpt_file)
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)

        num_params = np.sum([np.prod(v.shape) for v in saved_variables])
        print(f"Number of parameters saved: {num_params}")


def convert_gc_ckpt_to_google(
    ckpt_file, output_dir=None, include_qkv_bias=False, dtype=tf.float32
):
    graph = tf.Graph()
    dir_name, ckpt_name = os.path.split(os.path.abspath(ckpt_file))
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    with graph.as_default():
        sess = tf.Session()
        num_hidden_layers = 0
        word_embeddings = []
        new_variables = []
        keep_variables = []
        for tensor_name in var_to_shape_map:
            logging.info(f"Loading {tensor_name}")
            # Filter the optimizer variables
            if filter_optimizer(tensor_name):
                continue

            tensor_value = tf.cast(reader.get_tensor(tensor_name), dtype=dtype)
            if "word_embeddings" in tensor_name:
                word_embeddings.append(tensor_name)
            elif "attention" in tensor_name:
                layer_idx = int(tensor_name.split("/")[2].split("_")[-1])
                num_hidden_layers = max(layer_idx, num_hidden_layers)
                if "qkv_bias" in tensor_name and include_qkv_bias:
                    hidden_size = tensor_value.shape[0] // 3
                    query_bias = tensor_value[:hidden_size]
                    key_bias = tensor_value[hidden_size:2*hidden_size]
                    value_bias = tensor_value[2*hidden_size:]
                    qb = tf.Variable(
                        query_bias, name=tensor_name.replace("qkv_bias", "query/bias")
                    )
                    kb = tf.Variable(
                        key_bias, name=tensor_name.replace("qkv_bias", "key/bias")
                    )
                    vb = tf.Variable(
                        value_bias, name=tensor_name.replace("qkv_bias", "value/bias")
                    )
                    new_variables.extend([qb, kb, vb])
                # rename projection to output
                elif "projection" in tensor_name:
                    new_name = tensor_name.replace("projection", "output")

                    proj = tf.Variable(tensor_value, name=new_name)
                    new_variables.append(proj)
            else:
                var = tf.get_variable(tensor_name, shape=tensor_value.shape, dtype=dtype)
                keep_variables.append(var)

        # Combine split embeddings
        word_embeddings = np.sort(word_embeddings)
        embeddings_vals = [reader.get_tensor(k) for k in word_embeddings]
        unit_embeddings = np.vstack(embeddings_vals)
        logging.debug(f"Concated word_embeddings shape: {unit_embeddings.shape}")
        we = tf.Variable(
            unit_embeddings,
            dtype=dtype,
            shape=unit_embeddings.shape,
            name="bert/embeddings/word_embeddings",
        )
        new_variables.append(we)
        saved_variables = new_variables + keep_variables
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(var_list=saved_variables)
        output_file = os.path.join(output_dir, ckpt_name)
        saver.save(sess, output_file)
        print("Saved to :" + output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checkpoint conversion tool.",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("action",
                        type=str,
                        choices=["graphcore2google", "google2graphcore", "research2apps", "apps2research"],
                        help="""Type of conversion between models""")
    parser.add_argument("checkpoint_path",
                        type=str,
                        help="""Path to the existing checkpoint to convert.""")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="converted_ckpt",
        help="""Path to the output directory where the converted model will be stored."""
    )
    # Model-specific arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30400,
        help="""Vocabulary size of the converted model. The vocabulary list will be truncated if the initial model was trained with a larger vocabulary than vocab_size.""",
    )
    parser.add_argument(
        "--num_embeddings_splits",
        type=int,
        default=1,
        help="""Number word embeddings splits.""",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="""Convert GroupBERT or baseline checkpoint.""",
    )
    parser.add_argument(
        "--use_attention_bias",
        action="store_true",
        default=True,
        help="""Use attention bias in converted model.""",
    )
    parser.add_argument(
        "--use_qkv_bias",
        action="store_true",
        default=True,
        help="""Use biases in the QKV layer.""",
    )
    parser.add_argument(
        "--use_cls_layer",
        action="store_true",
        default=False,
        help="""Use a final dense layer.""",
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=[16, 32],
        default=16,
        help="""Default floating-point precision of converted model""",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    precision_type = None
    if args.precision == 16:
        precision_type = tf.float16
    elif args.precision == 32:
        precision_type = 32
    else:
        raise ValueError(f"Unsupported precision format {args.precision}")

    if args.action.lower() == "google2graphcore":
        convert_google_ckpt_to_gc(
            args.checkpoint_path,
            args.output_dir,
            num_embed_split=args.num_embeddings_splits,
            vocab_size=args.vocab_size,
            use_attention_bias=args.use_attention_bias,
            use_qkv_bias=args.use_qkv_bias,
            use_cls_layer=args.use_cls_layer,
            dtype=precision_type,
        )

    elif args.action.lower() == "graphcore2google":
        convert_gc_ckpt_to_google(
            args.checkpoint_path,
            args.output_dir,
            num_embed_split=args.num_embeddings_splits,
            vocab_size=args.vocab_size,
            use_attention_bias=args.use_attention_bias,
            use_qkv_bias=args.use_qkv_bias,
            use_cls_layer=args.use_cls_layer,
            dtype=precision_type,
        )

    elif args.action.lower() == "research2apps":
        convert_research_ckpt_to_apps(
            args.checkpoint_path,
            args.output_dir,
            num_embed_split=args.num_embeddings_splits,
            vocab_size=args.vocab_size,
            use_attention_bias=args.use_attention_bias,
            use_qkv_bias=args.use_qkv_bias,
            use_cls_layer=args.use_cls_layer,
            baseline=args.baseline,
            dtype=precision_type,
        )

    elif args.action.lower() == "apps2research":
        convert_apps_ckpt_to_research(
            args.checkpoint_path,
            args.output_dir,
            num_embed_split=args.num_embeddings_splits,
            vocab_size=args.vocab_size,
            use_attention_bias=args.use_attention_bias,
            use_qkv_bias=args.use_qkv_bias,
            use_cls_layer=args.use_cls_layer,
            baseline=args.baseline,
            dtype=precision_type,
        )
