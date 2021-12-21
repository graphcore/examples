# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from copy import deepcopy

from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.python.ipu.keras.layers import Dropout as IpuDropout
from tensorflow.python.ipu.keras.layers import LayerNormalization as IpuLayerNormalization
from transformers.models.bert.modeling_tf_bert import (
    TFBertEmbeddings,
    TFBertEncoder,
    TFBertLayer,
    TFBertLMPredictionHead,
    TFBertMainLayer
)

from keras_extensions.model_transformations import (
    convert_to_functional,
    ModelAddRecomputationCheckpoints,
    ModelExpansion,
    ModelOptimization,
    ModelOutlining,
    ModelReplacing
)
from model.ipu_embeddings_layer import IpuTFBertEmbeddings
from model.ipu_lm_prediction_head import IpuTFBertLMPredictionHead


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
    if layer_name == 'gather_masked_outputs':
        last_hidden_state = input_layer[0].__dict__['last_hidden_state']
        masked_lm_positions = input_layer[1]
        return last_hidden_state, masked_lm_positions

    if is_any_of_input_layers_of_class(input_layers_of[layer_name], TFBertMainLayer):
        if layer_name in ['mlm___cls', 'predictions', 'qa_outputs']:
            return input_layer.__dict__['last_hidden_state']
        else:
            return input_layer.__dict__['pooler_output']

    if is_any_of_input_layers_of_class(input_layers_of[layer_name], TFBertEncoder):
        return input_layer.__dict__['last_hidden_state']

    return input_layer


def convert_tf_bert_model(
        hf_model,
        dataset,
        post_process_input_fn,
        replace_layers=True,
        use_outlining=True,
        embedding_serialization_factor=1,
        rename_outputs=None
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
    :return: IPU optimised functional model with pipeline stages and IPU optimised layers.
    """
    model = convert_to_functional(hf_model, dataset)
    model.summary()

    def copy_weights_layer_with_input_shape_hidden_states(layer, new_layer):
        # Copies weights if the input_shape of the layer is hidden_states:
        # [batch_size, sequence_length, hidden_size]
        batch_size, seq_length = model.get_layer('bert').input_shape
        new_layer.build((batch_size, seq_length, new_layer.hidden_size))
        new_layer.set_weights(layer.get_weights())

    to_replace = [
        {TFBertEmbeddings: {
            "new_class": IpuTFBertEmbeddings,
            "new_params": {
                "config": deepcopy(hf_model.config),
                "serialization_factor": embedding_serialization_factor,
                # Ensure name of embeddings layer is the same as original
                # name of layer to ensure this layer is checkpointed
                # correctly.
                "name": "embeddings",
            },
            "copy_weights": True,
            "copy_weights_func": copy_weights_layer_with_input_shape_hidden_states
        }},
        {TFBertLMPredictionHead: {
            "new_class": IpuTFBertLMPredictionHead,
            "new_params": {
                "config": deepcopy(hf_model.config),
                "input_embeddings": lambda: model.get_layer("bert").embeddings,
                "serialization_factor": embedding_serialization_factor,
                # Ensure name of lm prediction head layer is the same as
                # original name of layer to ensure this layer is
                # checkpointed correctly.
                "name": "predictions",
            },
            "copy_weights": True,
            "copy_weights_func": copy_weights_layer_with_input_shape_hidden_states
        }},
        {Dropout: {'new_class': IpuDropout}},
        {LayerNormalization: {
            "new_class": IpuLayerNormalization,
            "copy_weights": True
        }},
    ]
    to_outline = {
        Dense: {"outline_kwargs": {}},
    }
    to_expand = [
        TFBertMainLayer,
        TFBertEncoder,
    ]
    add_recomputation_checkpoints_after = [TFBertLayer]

    if rename_outputs is not None:
        print(f"Attempting to rename outputs: {rename_outputs}")
        model = ModelOptimization().rename_outputs(rename_outputs, model)
    if replace_layers:
        # We pass the layers in order to ensure the layer replacement happens
        # within the layers that have been replaced.
        for to_replace_dict in to_replace:
            print(f"Attempting to replace layer: {to_replace_dict.keys()}")
            model = ModelReplacing(to_replace_dict).process_model('all', model, post_process_input_fn)
    if use_outlining:
        print(f"Attempting to outline layers: {to_outline}")
        model = ModelOutlining(to_outline).process_model('all', model, post_process_input_fn)
    print(f"Attempting to add recomputation checkpoints after layers: {add_recomputation_checkpoints_after}")
    model = ModelAddRecomputationCheckpoints(add_recomputation_checkpoints_after).process_model(
        'all', model, post_process_input_fn)
    for layer_id in to_expand:
        print(f"Attempting to expand layers: {to_expand}")
        model = ModelExpansion().process_model(layer_id, model, post_process_input_fn)
        model.summary()
    return model
