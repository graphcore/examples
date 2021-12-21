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
# limitations under the License.
import h5py
import tensorflow as tf
import numpy as np


def get_ckpt_mapper(opts):
    name_mapper = {}
    embeddings_mapper = {
        "decoder/position_embeddings/embeddings:0": "layer_with_weights-4/embeddings",
        "embeddings/charactor_embeddings/weight:0": "layer_with_weights-0/weight",
        "embeddings/position_embeddings/embeddings:0": "layer_with_weights-0/position_embeddings/embeddings"
    }
    name_mapper.update(**embeddings_mapper)

    f0_embeddings_mapper = {
        "f0_embeddings/kernel:0": "layer_with_weights-1/kernel",
        "f0_embeddings/bias:0": "layer_with_weights-1/bias",
    }
    name_mapper.update(**f0_embeddings_mapper)

    energy_embeddings_mapper = {
        "energy_embeddings/kernel:0": "layer_with_weights-2/kernel",
        "energy_embeddings/bias:0": "layer_with_weights-2/bias",
    }
    name_mapper.update(**energy_embeddings_mapper)

    # Encoder layer
    encoder_mapper = {}
    for i in range(opts["encoder_num_hidden_layers"]):
        encoder_mapper.update(**{
            f"encoder/layer_._{i}/attention/self/query/kernel:0": f"layer_with_weights-3/layer/{i}/attention/self_attention/query/kernel",
            f"encoder/layer_._{i}/attention/self/query/bias:0": f"layer_with_weights-3/layer/{i}/attention/self_attention/query/bias",
            f"encoder/layer_._{i}/attention/self/key/kernel:0": f"layer_with_weights-3/layer/{i}/attention/self_attention/key/kernel",
            f"encoder/layer_._{i}/attention/self/key/bias:0": f"layer_with_weights-3/layer/{i}/attention/self_attention/key/bias",
            f"encoder/layer_._{i}/attention/self/value/kernel:0": f"layer_with_weights-3/layer/{i}/attention/self_attention/value/kernel",
            f"encoder/layer_._{i}/attention/self/value/bias:0": f"layer_with_weights-3/layer/{i}/attention/self_attention/value/bias",
            f"encoder/layer_._{i}/attention/output/dense/kernel:0": f"layer_with_weights-3/layer/{i}/attention/dense_output/dense/kernel",
            f"encoder/layer_._{i}/attention/output/dense/bias:0": f"layer_with_weights-3/layer/{i}/attention/dense_output/dense/bias",
            f"encoder/layer_._{i}/attention/output/LayerNorm/gamma:0": f"layer_with_weights-3/layer/{i}/attention/dense_output/LayerNorm/gamma",
            f"encoder/layer_._{i}/attention/output/LayerNorm/beta:0": f"layer_with_weights-3/layer/{i}/attention/dense_output/LayerNorm/beta",
            f"encoder/layer_._{i}/intermediate/conv1d_1/kernel:0": f"layer_with_weights-3/layer/{i}/intermediate/conv1d_1/kernel",
            f"encoder/layer_._{i}/intermediate/conv1d_1/bias:0": f"layer_with_weights-3/layer/{i}/intermediate/conv1d_1/bias",
            f"encoder/layer_._{i}/intermediate/conv1d_2/kernel:0": f"layer_with_weights-3/layer/{i}/intermediate/conv1d_2/kernel",
            f"encoder/layer_._{i}/intermediate/conv1d_2/bias:0": f"layer_with_weights-3/layer/{i}/intermediate/conv1d_2/bias",
            f"encoder/layer_._{i}/output/LayerNorm/gamma:0": f"layer_with_weights-3/layer/{i}/bert_output/LayerNorm/gamma",
            f"encoder/layer_._{i}/output/LayerNorm/beta:0": f"layer_with_weights-3/layer/{i}/bert_output/LayerNorm/beta"
        })
    name_mapper.update(**encoder_mapper)

    # decoder
    decoder_mapper = {}
    for i in range(opts["decoder_num_hidden_layers"]):
        # self attention
        for j in ["query", "key", "value"]:
            for m in ["kernel", "bias"]:
                decoder_mapper.update(**{
                    f"decoder/layer_._{i}/attention/self/{j}/{m}:0": f"layer_with_weights-{i+5}/attention/self_attention/{j}/{m}",
                })
        # attention output
        decoder_mapper.update(**{
            f"decoder/layer_._{i}/attention/output/dense/kernel:0": f"layer_with_weights-{i+5}/attention/dense_output/dense/kernel",
            f"decoder/layer_._{i}/attention/output/dense/bias:0": f"layer_with_weights-{i+5}/attention/dense_output/dense/bias",
            f"decoder/layer_._{i}/attention/output/LayerNorm/gamma:0": f"layer_with_weights-{i+5}/attention/dense_output/LayerNorm/gamma",
            f"decoder/layer_._{i}/attention/output/LayerNorm/beta:0": f"layer_with_weights-{i+5}/attention/dense_output/LayerNorm/beta",
        })
        # intermediate
        for j in [1, 2]:
            decoder_mapper.update(**{
                f"decoder/layer_._{i}/intermediate/conv1d_{j}/kernel:0": f"layer_with_weights-{i+5}/intermediate/conv1d_{j}/kernel",
                f"decoder/layer_._{i}/intermediate/conv1d_{j}/bias:0": f"layer_with_weights-{i+5}/intermediate/conv1d_{j}/bias"
            })

        # intermediate output
        decoder_mapper.update(**{
            f"decoder/layer_._{i}/output/LayerNorm/gamma:0": f"layer_with_weights-{i+5}/bert_output/LayerNorm/gamma",
            f"decoder/layer_._{i}/output/LayerNorm/beta:0": f"layer_with_weights-{i+5}/bert_output/LayerNorm/beta"
        })
    name_mapper.update(**decoder_mapper)

    # f0/energy/duration predictor
    predictor_mapper = {}
    di = 0
    for name, layerid in zip(["duration", "f0", "energy"], [11, 12, 13]):
        for i in range(opts["variant_predictor_num_conv_layers"]):
            predictor_mapper.update(**{
                f"{name}_predictor/conv_._{i}/kernel:0": f"layer_with_weights-{layerid}/conv_layers/{i}/0/kernel",
                f"{name}_predictor/conv_._{i}/bias:0": f"layer_with_weights-{layerid}/conv_layers/{i}/0/bias",
                f"{name}_predictor/LayerNorm_._{i}/gamma:0": f"layer_with_weights-{layerid}/conv_layers/{i}/2/gamma",
                f"{name}_predictor/LayerNorm_._{i}/beta:0": f"layer_with_weights-{layerid}/conv_layers/{i}/2/beta",

            })
        if di == 0:
            predictor_mapper.update(**{
                f"{name}_predictor/dense/kernel:0": f"layer_with_weights-{layerid}/output_layer/kernel",
                f"{name}_predictor/dense/bias:0": f"layer_with_weights-{layerid}/output_layer/bias"
            })
        else:
            predictor_mapper.update(**{
                f"{name}_predictor/dense_{di}/kernel:0": f"layer_with_weights-{layerid}/output_layer/kernel",
                f"{name}_predictor/dense_{di}/bias:0": f"layer_with_weights-{layerid}/output_layer/bias"
            })
        di += 1
    name_mapper.update(**predictor_mapper)

    # mel before
    melb_mapper = {
        "mel_before/kernel:0": "layer_with_weights-9/kernel",
        "mel_before/bias:0": "layer_with_weights-9/bias"
    }
    name_mapper.update(**melb_mapper)

    # postnet
    postnet_mapper = {}
    for i in range(opts["postnet_num_conv_layers"]):
        postnet_mapper.update(**{
            f"postnet/conv_._{i}/kernel:0": f"layer_with_weights-10/conv_batch_norm/{i}/0/kernel",
            f"postnet/conv_._{i}/bias:0": f"layer_with_weights-10/conv_batch_norm/{i}/0/bias",
            f"postnet/batch_norm_._{i}/gamma:0": f"layer_with_weights-10/conv_batch_norm/{i}/0/gamma",
            f"postnet/batch_norm_._{i}/beta:0": f"layer_with_weights-10/conv_batch_norm/{i}/0/beta"
        })
    name_mapper.update(**postnet_mapper)
    print(len(name_mapper))
    return name_mapper


def get_h5_mapper(opts):
    name_mapper = {}
    embedding_mapper = {
        'embeddings/charactor_embeddings/weight:0': 'tf2_fastspeech2/embeddings/charactor_embeddings/weight:0',
        'embeddings/position_embeddings/embeddings:0': 'tf2_fastspeech2/embeddings/position_embeddings/embeddings:0',
        'decoder/position_embeddings/embeddings:0': 'tf2_fastspeech2/decoder/position_embeddings/embeddings:0',
    }
    name_mapper.update(**embedding_mapper)
    # f0/energy embeddings
    emb_mapper = {}
    for prefix in ['f0', 'energy']:
        emb_mapper.update(**{
            f'{prefix}_embeddings/kernel:0': f'tf2_fastspeech2/{prefix}_embeddings/kernel:0',
            f'{prefix}_embeddings/bias:0': f'tf2_fastspeech2/{prefix}_embeddings/bias:0',
        })
    name_mapper.update(**emb_mapper)

    # encoder
    encoder_mapper = {}
    for i in range(opts["encoder_num_hidden_layers"]):
        encoder_mapper.update(**{
            f"encoder/layer_._{i}/attention/self/query/kernel:0": f'tf2_fastspeech2/encoder/layer_._{i}/attention/self/query/kernel:0',
            f"encoder/layer_._{i}/attention/self/query/bias:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/self/query/bias:0",
            f"encoder/layer_._{i}/attention/self/key/kernel:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/self/key/kernel:0",
            f"encoder/layer_._{i}/attention/self/key/bias:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/self/key/bias:0",
            f"encoder/layer_._{i}/attention/self/value/kernel:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/self/value/kernel:0",
            f"encoder/layer_._{i}/attention/self/value/bias:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/self/value/bias:0",
            f"encoder/layer_._{i}/attention/output/dense/kernel:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/output/dense/kernel:0",
            f"encoder/layer_._{i}/attention/output/dense/bias:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/output/dense/bias:0",
            f"encoder/layer_._{i}/attention/output/LayerNorm/gamma:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/output/LayerNorm/gamma:0",
            f"encoder/layer_._{i}/attention/output/LayerNorm/beta:0": f"tf2_fastspeech2/encoder/layer_._{i}/attention/output/LayerNorm/beta:0",
            f"encoder/layer_._{i}/intermediate/conv1d_1/kernel:0": f"tf2_fastspeech2/encoder/layer_._{i}/intermediate/conv1d_1/kernel:0",
            f"encoder/layer_._{i}/intermediate/conv1d_1/bias:0": f"tf2_fastspeech2/encoder/layer_._{i}/intermediate/conv1d_1/bias:0",
            f"encoder/layer_._{i}/intermediate/conv1d_2/kernel:0": f"tf2_fastspeech2/encoder/layer_._{i}/intermediate/conv1d_2/kernel:0",
            f"encoder/layer_._{i}/intermediate/conv1d_2/bias:0": f"tf2_fastspeech2/encoder/layer_._{i}/intermediate/conv1d_2/bias:0",
            f"encoder/layer_._{i}/output/LayerNorm/gamma:0": f"tf2_fastspeech2/encoder/layer_._{i}/output/LayerNorm/gamma:0",
            f"encoder/layer_._{i}/output/LayerNorm/beta:0": f"tf2_fastspeech2/encoder/layer_._{i}/output/LayerNorm/beta:0"
        })
    name_mapper.update(**encoder_mapper)

    # decoder
    decoder_mapper = {}
    for i in range(opts["decoder_num_hidden_layers"]):
        decoder_mapper.update(**{
            f"decoder/layer_._{i}/attention/self/query/kernel:0": f'tf2_fastspeech2/decoder/layer_._{i}/attention/self/query/kernel:0',
            f"decoder/layer_._{i}/attention/self/query/bias:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/self/query/bias:0",
            f"decoder/layer_._{i}/attention/self/key/kernel:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/self/key/kernel:0",
            f"decoder/layer_._{i}/attention/self/key/bias:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/self/key/bias:0",
            f"decoder/layer_._{i}/attention/self/value/kernel:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/self/value/kernel:0",
            f"decoder/layer_._{i}/attention/self/value/bias:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/self/value/bias:0",
            f"decoder/layer_._{i}/attention/output/dense/kernel:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/output/dense/kernel:0",
            f"decoder/layer_._{i}/attention/output/dense/bias:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/output/dense/bias:0",
            f"decoder/layer_._{i}/attention/output/LayerNorm/gamma:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/output/LayerNorm/gamma:0",
            f"decoder/layer_._{i}/attention/output/LayerNorm/beta:0": f"tf2_fastspeech2/decoder/layer_._{i}/attention/output/LayerNorm/beta:0",
            f"decoder/layer_._{i}/intermediate/conv1d_1/kernel:0": f"tf2_fastspeech2/decoder/layer_._{i}/intermediate/conv1d_1/kernel:0",
            f"decoder/layer_._{i}/intermediate/conv1d_1/bias:0": f"tf2_fastspeech2/decoder/layer_._{i}/intermediate/conv1d_1/bias:0",
            f"decoder/layer_._{i}/intermediate/conv1d_2/kernel:0": f"tf2_fastspeech2/decoder/layer_._{i}/intermediate/conv1d_2/kernel:0",
            f"decoder/layer_._{i}/intermediate/conv1d_2/bias:0": f"tf2_fastspeech2/decoder/layer_._{i}/intermediate/conv1d_2/bias:0",
            f"decoder/layer_._{i}/output/LayerNorm/gamma:0": f"tf2_fastspeech2/decoder/layer_._{i}/output/LayerNorm/gamma:0",
            f"decoder/layer_._{i}/output/LayerNorm/beta:0": f"tf2_fastspeech2/decoder/layer_._{i}/output/LayerNorm/beta:0"
        })
    name_mapper.update(**decoder_mapper)

    # f0/energy/duration predictor
    predictor_mapper = {}
    for offset, prefix in enumerate(['duration', 'f0', 'energy']):
        for i in range(opts["variant_predictor_num_conv_layers"]):
            predictor_mapper.update(**{
                f"{prefix}_predictor/conv_._{i}/kernel:0": f"{prefix}_predictor/conv_._{i}/kernel:0",
                f"{prefix}_predictor/conv_._{i}/bias:0": f"{prefix}_predictor/conv_._{i}/bias:0",
                f"{prefix}_predictor/LayerNorm_._{i}/gamma:0": f"{prefix}_predictor/LayerNorm_._{i}/gamma:0",
                f"{prefix}_predictor/LayerNorm_._{i}/beta:0": f"{prefix}_predictor/LayerNorm_._{i}/beta:0",
            })
        # last dense layer in predictor
        if prefix == "duration":
            predictor_mapper.update(**{
                f'{prefix}_predictor/dense_{offset+4}/kernel:0': f'tf2_fastspeech2/{prefix}_predictor/dense_3/kernel:0',
                f'{prefix}_predictor/dense_{offset+4}/bias:0': f'tf2_fastspeech2/{prefix}_predictor/dense_3/bias:0',
            })
        else:
            predictor_mapper.update(**{
                f'{prefix}_predictor/dense_{offset+4}/kernel:0': f'tf2_fastspeech2/{prefix}_predictor/dense_{offset}/kernel:0',
                f'{prefix}_predictor/dense_{offset+4}/bias:0': f'tf2_fastspeech2/{prefix}_predictor/dense_{offset}/bias:0',
            })
    name_mapper.update(**predictor_mapper)

    # mel before
    melb_mapper = {
        "mel_before/kernel:0": 'tf2_fastspeech2/mel_before/kernel:0',
        "mel_before/bias:0": 'tf2_fastspeech2/mel_before/bias:0'
    }
    name_mapper.update(**melb_mapper)

    # postnet
    postnet_mapper = {}
    for i in range(opts["postnet_num_conv_layers"]):
        postnet_mapper.update(**{
            f"postnet/conv_._{i}/kernel:0": f"tf2_fastspeech2/postnet/conv_._{i}/kernel:0",
            f"postnet/conv_._{i}/bias:0": f"tf2_fastspeech2/postnet/conv_._{i}/bias:0",
            f"postnet/batch_norm_._{i}/gamma:0": f"tf2_fastspeech2/postnet/batch_norm_._{i}/gamma:0",
            f"postnet/batch_norm_._{i}/beta:0": f"tf2_fastspeech2/postnet/batch_norm_._{i}/beta:0"
        })
    name_mapper.update(**postnet_mapper)
    print(len(name_mapper))
    return name_mapper


def parse_ckpt(ckpt_file):
    reader = tf.compat.v1.NewCheckpointReader(ckpt_file)
    variable_to_shape = reader.get_variable_to_shape_map()
    variable_to_shape = {
        k: v for k, v in variable_to_shape.items() if "optimizer" not in k}
    weights = {k: reader.get_tensor(k) for k, v in variable_to_shape.items()}
    weights_value = {k.strip(".ATTRIBUTES/VARIABLE_VALUE"): reader.get_tensor(k)
                     for k, v in weights.items() if isinstance(v, np.ndarray)}
    return weights_value


def parse_h5(h5_file, rename=True, redundant_decoder_name=False):
    f = h5py.File(h5_file, 'r')
    tensor_names = []

    def _find_name(name):
        if isinstance(f[name], h5py.Dataset):
            tensor_names.append(name)

    f.visit(_find_name)
    pretrained_weights = {}
    for tname in tensor_names:
        pretrained_weights[tname] = f.get(tname)[()]  # get datasets value
    rename_weights = {}  # rename to match model.weights name
    for k, v in pretrained_weights.items():
        print(k, v.shape)
        if redundant_decoder_name and "decoder" in k:
            newname = "/".join(k.split("/")[2:])
        else:
            newname = "/".join(k.split("/")[1:])
        rename_weights[newname] = v
    return rename_weights if rename else pretrained_weights
