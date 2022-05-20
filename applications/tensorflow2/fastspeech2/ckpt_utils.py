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


def parse_ckpt(ckpt_file):
    reader = tf.compat.v1.NewCheckpointReader(ckpt_file)
    variable_to_shape = reader.get_variable_to_shape_map()
    variable_to_shape = {
        k: v for k, v in variable_to_shape.items() if "optimizer" not in k}
    weights = {k: reader.get_tensor(k) for k, v in variable_to_shape.items()}
    weights_value = {k.strip(".ATTRIBUTES/VARIABLE_VALUE"): reader.get_tensor(k)
                     for k, v in weights.items() if isinstance(v, np.ndarray)}
    return weights_value


def get_gpu_predictor_mapper():
    # predictor mapper
    mapper = {
        'duration_predictor/conv_._0/kernel:0': 'duration_predictor/sequential_3/conv_._0/kernel:0',
        'duration_predictor/conv_._0/bias:0': 'duration_predictor/sequential_3/conv_._0/bias:0',
        'duration_predictor/LayerNorm_._0/gamma:0': 'duration_predictor/sequential_3/LayerNorm_._0/gamma:0',
        'duration_predictor/LayerNorm_._0/beta:0': 'duration_predictor/sequential_3/LayerNorm_._0/beta:0',
        'duration_predictor/conv_._1/kernel:0': 'duration_predictor/sequential_3/conv_._1/kernel:0',
        'duration_predictor/conv_._1/bias:0': 'duration_predictor/sequential_3/conv_._1/bias:0',
        'duration_predictor/LayerNorm_._1/gamma:0': 'duration_predictor/sequential_3/LayerNorm_._1/gamma:0',
        'duration_predictor/LayerNorm_._1/beta:0': 'duration_predictor/sequential_3/LayerNorm_._1/beta:0',
        'duration_predictor/dense/kernel:0': 'duration_predictor/dense_3/kernel:0',
        'duration_predictor/dense/bias:0': 'duration_predictor/dense_3/bias:0',
        'f0_predictor/conv_._0/kernel:0': 'f0_predictor/sequential_1/conv_._0/kernel:0',
        'f0_predictor/conv_._0/bias:0': 'f0_predictor/sequential_1/conv_._0/bias:0',
        'f0_predictor/LayerNorm_._0/gamma:0': 'f0_predictor/sequential_1/LayerNorm_._0/gamma:0',
        'f0_predictor/LayerNorm_._0/beta:0': 'f0_predictor/sequential_1/LayerNorm_._0/beta:0',
        'f0_predictor/conv_._1/kernel:0': 'f0_predictor/sequential_1/conv_._1/kernel:0',
        'f0_predictor/conv_._1/bias:0': 'f0_predictor/sequential_1/conv_._1/bias:0',
        'f0_predictor/LayerNorm_._1/gamma:0': 'f0_predictor/sequential_1/LayerNorm_._1/gamma:0',
        'f0_predictor/LayerNorm_._1/beta:0': 'f0_predictor/sequential_1/LayerNorm_._1/beta:0',
        'f0_predictor/dense_1/kernel:0': 'f0_predictor/dense_1/kernel:0',
        'f0_predictor/dense_1/bias:0': 'f0_predictor/dense_1/bias:0',
        'energy_predictor/conv_._0/kernel:0': 'energy_predictor/sequential_2/conv_._0/kernel:0',
        'energy_predictor/conv_._0/bias:0': 'energy_predictor/sequential_2/conv_._0/bias:0',
        'energy_predictor/LayerNorm_._0/gamma:0': 'energy_predictor/sequential_2/LayerNorm_._0/gamma:0',
        'energy_predictor/LayerNorm_._0/beta:0': 'energy_predictor/sequential_2/LayerNorm_._0/beta:0',
        'energy_predictor/conv_._1/kernel:0': 'energy_predictor/sequential_2/conv_._1/kernel:0',
        'energy_predictor/conv_._1/bias:0': 'energy_predictor/sequential_2/conv_._1/bias:0',
        'energy_predictor/LayerNorm_._1/gamma:0': 'energy_predictor/sequential_2/LayerNorm_._1/gamma:0',
        'energy_predictor/LayerNorm_._1/beta:0': 'energy_predictor/sequential_2/LayerNorm_._1/beta:0',
        'energy_predictor/dense_2/kernel:0': 'energy_predictor/dense_2/kernel:0',
        'energy_predictor/dense_2/bias:0': 'energy_predictor/dense_2/bias:0'

    }
    # GPU weight names --> IPU weight names
    return {v: k for k, v in mapper.items()}


def load_weights_from_h5file(h5_file_path):
    f = h5py.File(h5_file_path, 'r')
    tensor_names = []

    def _find_name(name):
        if isinstance(f[name], h5py.Dataset):
            tensor_names.append(name)

    f.visit(_find_name)
    pretrained_weights = {}
    for tname in tensor_names:
        pretrained_weights[tname] = f.get(tname)[()]  # get datasets value
    return pretrained_weights


def rename_pretrained_weights(pretrained_weights, mode='ipu', debug=False):
    """
    Rename pretrained weights to match model.weights name.
    It's a bit different where loading weights between GPU and IPU pretrained h5 files.
    Furthermore, we got another different name scope while saving weights in unit test.
    So we provide two modes for renaming the weights.

    Args:
        pretrained_weights: Dict[str:numpy.array]. Pretrained weights from `load_weights_from_h5file` function.
        mode: Str. Different renaming mode include ['gpu', 'ipu']. 'gpu' means the pre-trained weights
                are saved on the GPU. 'ipu' means the pre-trained weights are saved on the IPU.
        debug: Bool. Whether logging more information about renaming or not.

    Returns:
        Dict[str:numpy.array]. The renamed weights.
    """
    assert mode in ['gpu', 'ipu'], "Only support one of ['gpu', 'ipu']."
    rename_weights = {}
    for k, v in pretrained_weights.items():
        if mode == "gpu":
            # decoder/tf_fast_speech2/decoder/layer_._0/attention/output/LayerNorm/beta:0 ==> decoder/layer_._0/attention/output/LayerNorm/beta:0,
            newname = "/".join(k.split("/")[2:])
        elif mode == "ipu":
            # rename `decoder/layer_._0/decoder/layer_._0/attention/output/LayerNorm/beta:0` to
            # `decoder/layer_._0/attention/output/LayerNorm/beta:0`
            if "decoder" in k:
                newname = "/".join(k.split("/")[2:])
            # rename `f0_embeddings/f0_embeddings/bias:0` to `f0_embeddings/bias:0`.
            # Only IPU pretrained weights have such names.
            else:
                newname = "/".join(k.split("/")[1:])
        else:
            newname = k
        if debug:
            print(f"rename {k} ==> {newname}, {v.shape}")
        rename_weights[newname] = v
    return rename_weights


def set_weights(ckpt_path, model, mode="ipu", debug=False):
    reload_weights = load_weights_from_h5file(ckpt_path)
    reload_weights = rename_pretrained_weights(
        reload_weights, mode=mode, debug=debug)
    weights_to_restore = []
    wi_names = [w.name for w in model.weights]
    if mode == "gpu":
        predictore_mapper = get_gpu_predictor_mapper()
        rename_weights = {}
        for k, v in reload_weights.items():
            if k in predictore_mapper.keys():
                rename_weights[predictore_mapper[k]] = v
            else:
                if "moving_" in k:
                    # drop moving_mean and moving_variance
                    continue
                rename_weights[k] = v
        reload_weights = rename_weights

    not_match = [k for k in reload_weights.keys() if k not in wi_names]
    if len(not_match) > 0:
        print(f"Mismatch: {len(not_match)}\n{not_match}")
    for w in model.weights:
        weights_to_restore.append(reload_weights[w.name])
    assert len(weights_to_restore) == len(reload_weights) == len(wi_names), \
        f"Weights loading failed.Loaded {len(weights_to_restore)}/{len(wi_names)}."
    model.set_weights(weights_to_restore)
    return model


def parse_h5(h5_file, is_gpu=True, debug=False):
    """
    Parse weights(*.h5) file from whether pretrained on GPU or IPU.
    """
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
        # rename `decoder/layer_._0/decoder/layer_._0/attention/output/LayerNorm/beta:0` to
        # `decoder/layer_._0/attention/output/LayerNorm/beta:0`
        if is_gpu or "decoder" in k:
            newname = "/".join(k.split("/")[2:])
        # rename `f0_embeddings/f0_embeddings/bias:0` to `f0_embeddings/bias:0`.
        # Only IPU pretrained weights have such names.
        else:
            newname = "/".join(k.split("/")[1:])
        if debug:
            print(f"rename {k} ==> {newname}, {v.shape}")
        rename_weights[newname] = v
    return rename_weights
