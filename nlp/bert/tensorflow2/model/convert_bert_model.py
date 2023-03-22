# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import random
from copy import deepcopy

import tensorflow as tf
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from transformers.models.bert.modeling_tf_bert import (
    TFBertEmbeddings,
    TFBertEncoder,
    TFBertLayer,
    TFBertLMPredictionHead,
    TFBertMainLayer,
    TFBertOutput,
    TFBertSelfAttention,
    TFBertSelfOutput,
)

from keras_extensions.model_transformations import (
    convert_to_functional,
    ModelAddRecomputationCheckpoints,
    ModelExpansion,
    ModelOptimization,
    ModelOutlining,
    ModelReplacing,
)
from model.ipu_self_output import IpuTFBertSelfOutput
from model.ipu_custom_keras_layers import IpuDropoutCustom
from model.ipu_embeddings_layer import IpuTFBertEmbeddings
from model.ipu_lm_prediction_head import IpuTFBertLMPredictionHead
from model.ipu_self_attention import IpuTFBertSelfAttention


def is_any_of_input_layers_of_class(input_layers, layer_class):
    for layer in input_layers:
        if isinstance(layer, layer_class):
            return True
    return False


def post_process_bert_input_layer(input_layer, layer_name, input_layers_of, input_names_of):
    """
    In custom models, it is common that some layers have multiple outputs that are not meant to be used together
    (e.g., when adding heads for different tasks on top of a pretrained NLP model). In this case, it is needed
    to indicate which outputs should be connected to the next layer when propagating a symbolic tensor layer by layer
    (e.g., when expanding or replacing a layer). This function, indicates which outputs from the previous layers should
    be used for some particular layers of the BERT model. In particular, the MLM head uses the `last_hidden_state`
    output from the `TFBertMainLayer` layer, or from the `TFBertEncoder` if `TFBertMainLayer` has been expanded.

    :param input_layer: Symbolic tensor at the input and to be propagated through the current layer.
    :param layer_name: String with name of the current layer.
    :param input_layers_of: Dict of input layers (symbolic tensors) that are input to each layer ({layer: input_layer}).
    :param input_names_of: Dict of names of layers that are input to every layer ({layer_name: input layer names}}.
    :return: Symbolic tensor that should be the input of the layer being processed.
    """
    if layer_name == "gather_masked_outputs":
        last_hidden_state = input_layer[0].__dict__["last_hidden_state"]
        masked_lm_positions = input_layer[1]
        return last_hidden_state, masked_lm_positions

    if is_any_of_input_layers_of_class(input_layers_of[layer_name], TFBertMainLayer):
        if layer_name in ["mlm___cls", "predictions", "qa_outputs"]:
            return input_layer.__dict__["last_hidden_state"]
        else:
            return input_layer.__dict__["pooler_output"]

    if is_any_of_input_layers_of_class(input_layers_of[layer_name], TFBertEncoder):
        return input_layer.__dict__["last_hidden_state"]

    return input_layer


def copy_weights_layer_with_input_shape_hidden_states_func(layer, new_layer, batch_size, seq_length):
    # Copy weights if the input_shape of the layer is hidden_states:
    # [batch_size, sequence_length, hidden_size]
    new_layer.build((batch_size, seq_length, new_layer.hidden_size))
    new_layer.set_weights(layer.get_weights())


def copy_lm_prediction_head_weights_func(layer, new_layer, batch_size, seq_length, use_cls_layer, use_prediction_bias):
    # Copy weights if the input_shape of the layer is hidden_states:
    # [batch_size, sequence_length, hidden_size]
    # Note the order of the weights in TFBertLMPredictionHead is as follows,
    # where H denotes the hidden size:
    # 0 prediction bias (vocab_size,), 1 prediction transform kernel (H, H), 2 prediction transform bias (H,)
    # 3 and 4 transform layer norm gamma or beta (H,), 5 word_embedding (H, H), 6 token type embedding (2, H)
    # 7 position embedding (max_position_embeddings, H), 8 and 9 layer norm gamma or beta (H,)
    new_layer.build((batch_size, seq_length, new_layer.hidden_size))
    layer_weights = layer.get_weights()
    if not use_cls_layer:
        del layer_weights[1:5]
    if not use_prediction_bias:
        del layer_weights[0]
    new_layer.set_weights(layer_weights)


def copy_self_output_weights_func(layer, new_layer, batch_size, seq_length, use_projection_bias):
    # Note the order of the weights in TFBertSelfOutput is as follows,
    # where H denotes the hidden size:
    # 0 dense layer kernel (H, H), 1 dense layer bias (H,),
    # 2 layer norm gamma (H,), 3 layer norm beta (H,)
    # Note the order of the weights in IpuTFBertSelfOutput is as follows:
    # 0 layer norm gamma (H,), 1 layer norm beta (H,)
    # 2 dense layer kernel (H, H)
    new_layer.build((batch_size, seq_length, new_layer.config.hidden_size))
    if use_projection_bias:
        new_layer.set_weights(layer.get_weights())
    else:
        layer_weights = layer.get_weights()
        new_weights = list()
        # Skip the bias when copying the weights.
        new_weights.extend(layer_weights[2:])
        new_weights.append(layer_weights[0])
        new_layer.set_weights(new_weights)


def copy_self_attention_weights_func(layer, new_layer, use_qkv_bias, use_qkv_split):
    # concatenate q k v weights [all_head_size, all_head_size]*3 from layer
    # and set the qkv weight [all_head_size, all_head_size*3] to new layer
    # there are 6 tensors in the old layer: q_weight index 0, q_bias index 1,
    # k_weight index 2, k_bias index 3, v_weight index 4, v_bias index 5.
    layer_weights = layer.get_weights()
    if not use_qkv_bias:
        if not use_qkv_split:
            new_layer.build((new_layer.all_head_size, new_layer.all_head_size * 3))
            new_weights = list()
            new_weights.append(tf.concat([layer_weights[0], layer_weights[2], layer_weights[4]], axis=-1))
        else:
            new_layer.build((new_layer.all_head_size, new_layer.all_head_size))
            new_weights = list()
            new_weights.append(layer_weights[0])
            new_weights.append(layer_weights[2])
            new_weights.append(layer_weights[4])
    else:
        new_layer.build((new_layer.all_head_size, new_layer.all_head_size))
        new_weights = layer_weights
    new_layer.set_weights(new_weights)


def convert_tf_bert_model(
    hf_model,
    dataset,
    post_process_input_fn,
    replace_layers=True,
    use_outlining=True,
    enable_recomputation=True,
    embedding_serialization_factor=1,
    rename_outputs=None,
    use_prediction_bias=False,
    use_cls_layer=False,
    use_qkv_bias=False,
    use_qkv_split=False,
    use_projection_bias=False,
):
    """
    Convert original subclass model to a functional one and transform it to optimise performance.
    :param hf_model: Original subclass model.
    :param dataset: Dataset used to train the model.
    :param post_process_input_fn: Handler to function to process the input layer for layers that don't use all the
        outputs from previous layers.
    :param replace_layers: Flag indicating if some layers should be replaced.
    :param use_outlining: Flag indicating if some layers should be outlined.
    :param embedding_serialization_factor: If greater than 1, the embedding lookup will be broken up into
        serialization_factor smaller lookups, serialized along the 0th dimension, reducing the maximum memory at the
        cost of extra computation.
    :param rename_outputs: Dictionary with names of the outputs that have to be renamed.
    :param use_prediction_bias: Flag indicating if bias units should be included in the prediction head.
    :param use_cls_layer: Flag indicating if CLS layer is included in the MLM Prediction head.
    :param use_qkv_bias: Flag indicating if bias units should be included in the self-attention QKV blocks.
    :param use_qkv_split: Flag indicating if self-attention QKV blocks should be split or merged.
    :param use_projection_bias: Flag indicating if attention projection bias is used.
    :return: IPU optimised functional model with pipeline stages and IPU optimised layers.
    """
    model = convert_to_functional(hf_model, dataset)
    model.summary(print_fn=logging.info)
    batch_size, seq_length = model.get_layer("bert").input_shape

    def copy_weights_layer_with_input_shape_hidden_states(layer, new_layer):
        copy_weights_layer_with_input_shape_hidden_states_func(layer, new_layer, batch_size, seq_length)

    def copy_lm_prediction_head_weights(layer, new_layer):
        copy_lm_prediction_head_weights_func(
            layer, new_layer, batch_size, seq_length, use_cls_layer, use_prediction_bias
        )

    def copy_self_attention_weights(layer, new_layer):
        copy_self_attention_weights_func(layer, new_layer, use_qkv_bias, use_qkv_split)

    def copy_self_output_weights(layer, new_layer):
        copy_self_output_weights_func(layer, new_layer, batch_size, seq_length, use_projection_bias)

    to_replace = [
        {
            TFBertEmbeddings: {
                "new_class": IpuTFBertEmbeddings,
                "new_params": {
                    "config": deepcopy(hf_model.config),
                    "serialization_factor": embedding_serialization_factor,
                },
                "copy_weights": True,
                "copy_weights_func": copy_weights_layer_with_input_shape_hidden_states,
            }
        },
        {
            TFBertLMPredictionHead: {
                "new_class": IpuTFBertLMPredictionHead,
                "new_params": {
                    "config": deepcopy(hf_model.config),
                    "input_embeddings": lambda: model.get_layer("bert").embeddings,
                    "use_cls_layer": use_cls_layer,
                    "use_prediction_bias": use_prediction_bias,
                    "serialization_factor": embedding_serialization_factor,
                },
                "copy_weights": True,
                "copy_weights_func": copy_lm_prediction_head_weights,
            }
        },
        {
            TFBertSelfAttention: {
                "new_class": IpuTFBertSelfAttention,
                "new_params": {
                    "config": deepcopy(hf_model.config),
                    "use_qkv_bias": use_qkv_bias,
                    "use_qkv_split": use_qkv_split,
                },
                "copy_weights": True,
                "copy_weights_func": copy_self_attention_weights,
            }
        },
        {
            TFBertSelfOutput: {
                "new_class": IpuTFBertSelfOutput,
                "new_params": {"config": deepcopy(hf_model.config), "use_projection_bias": use_projection_bias},
                "copy_weights": True,
                "copy_weights_func": copy_self_output_weights,
            }
        },
        {Dropout: {"new_class": IpuDropoutCustom}},
        {LayerNormalization: {"new_class": IpuLayerNormalization, "copy_weights": True}},
    ]

    to_outline = {
        Dense: {"outline_kwargs": {}},
        TFBertOutput: {"outline_kwargs": {}},
        TFBertSelfOutput: {"outline_kwargs": {}},
        IpuTFBertSelfAttention: {"outline_kwargs": {}},
        IpuTFBertSelfOutput: {"outline_kwargs": {}},
    }

    to_expand = [
        TFBertMainLayer,
        TFBertEncoder,
    ]
    add_recomputation_checkpoints_after = [TFBertLayer]

    if rename_outputs is not None:
        logging.info(f"Attempting to rename outputs: {rename_outputs}")
        model = ModelOptimization().rename_outputs(rename_outputs, model)
    if replace_layers:
        # We pass the layers in order to ensure the layer replacement happens
        # within the layers that have been replaced.
        for to_replace_dict in to_replace:
            logging.info(f"Attempting to replace layer: {to_replace_dict.keys()}")
            model = ModelReplacing(to_replace_dict).process_model("all", model, post_process_input_fn)
    if use_outlining:
        logging.info(f"Attempting to outline layers: {to_outline}")
        model = ModelOutlining(to_outline).process_model("all", model, post_process_input_fn)
    if enable_recomputation:
        logging.info(f"Attempting to add recomputation checkpoints after layers: {add_recomputation_checkpoints_after}")
        model = ModelAddRecomputationCheckpoints(add_recomputation_checkpoints_after).process_model(
            "all", model, post_process_input_fn
        )
    for layer_id in to_expand:
        logging.info(f"Attempting to expand layers: {to_expand}")
        model = ModelExpansion().process_model(layer_id, model, post_process_input_fn)
        model.summary(print_fn=logging.info)
    return model
