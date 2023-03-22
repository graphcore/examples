# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from collections import defaultdict
from copy import deepcopy
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.python import ipu
from ipu_tensorflow_addons.keras.layers import Dropout as IpuDropout
from ipu_tensorflow_addons.keras.layers import LayerNormalization as IpuLayerNormalization
from ipu_tensorflow_addons.keras.layers import SerialDense
from keras.utils.layer_utils import count_params
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_tf_bert import (
    TFBertEmbeddings,
    TFBertLMPredictionHead,
    TFBertSelfOutput,
    TFBertSelfAttention,
)
from model.convert_bert_model import (
    copy_weights_layer_with_input_shape_hidden_states_func,
    copy_lm_prediction_head_weights_func,
    copy_self_attention_weights_func,
    copy_self_output_weights_func,
)
from model.ipu_custom_keras_layers import IpuDropoutCustom
from model.ipu_self_output import IpuTFBertSelfOutput
from model.ipu_embeddings_layer import IpuTFBertEmbeddings
from model.ipu_lm_prediction_head import IpuTFBertLMPredictionHead
from model.ipu_self_attention import IpuTFBertSelfAttention
from utilities.ipu_utils import set_random_seeds
from keras_extensions.model_transformations import (
    convert_to_functional,
    ModelAddRecomputationCheckpoints,
    ModelExpansion,
    ModelOutlining,
    ModelReplacing,
)
from tests.utils import create_sample, EmbeddingModel, LMPredictionHeadModel, TFBertSelfOutputModel, SelfAttentionModel


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, mock_custom_param=None):  # Used to test
        super().__init__()
        self.mock_custom_param = mock_custom_param
        self.dense_1 = Dense(8, activation="relu", dtype=dtype, name="dense_1")
        self.dropout_1 = Dropout(0.3, name="dropout_1")
        self.norm_1 = LayerNormalization(axis=0, name="norm_1")
        self.dense_2 = Dense(24, activation="relu", name="dense_2")
        self.dropout_2 = Dropout(0.2, name="dropout_2")
        self.norm_2 = LayerNormalization(axis=(0, 1, 2), name="norm_2")
        self.dense_3 = Dense(4, activation="softmax", name="dense_3")
        self.norm_3 = LayerNormalization(axis=(0, 2), name="norm_3")

    def call(self, inputs, **kwargs):
        x = self.dense_1(inputs)
        x = self.dropout_1(x)
        x = self.norm_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.norm_2(x)
        x = self.dense_3(x)
        x = self.norm_3(x)
        return x


class CustomLayerFp16(CustomLayer):
    def __init__(self):
        super().__init__(dtype=tf.float16)


class CustomTFBertEmbeddings(TFBertEmbeddings):
    pass


class SimpleCustomLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        self.dense = Dense(8, activation="softmax")
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        return self.dense(inputs)


class SimpleCustomModel(tf.keras.Model):
    def __init__(self):
        self.dense = tf.keras.layers.Dense(4, activation="softmax")
        super().__init__()

    def call(self, inputs, training=None, mask=None):
        return self.dense(inputs)


def func_model_with_no_dense(batch_size):
    input_shape = (10, 120)
    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size)
    out = tf.keras.layers.Conv1D(32, 3, input_shape=input_shape)(input_layer)
    model = tf.keras.Model(input_layer, out)
    x_val = tf.random.normal((batch_size, *model.input_shape[1:]))
    return model, x_val


def func_model_with_dense(batch_size):
    input_shape = (4, 32)
    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size)
    x = tf.keras.layers.Dense(32, activation="relu")(input_layer)
    out = tf.keras.layers.Dense(6, activation="softmax")(x)
    model = tf.keras.Model(input_layer, out)
    x_val = tf.random.normal((batch_size, *input_shape))
    return model, x_val


def func_model_with_dense_dropout_and_layer_normalization(batch_size):
    input_shape = (4, 32)
    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size)
    x = tf.keras.layers.Dense(32, activation="relu")(input_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.LayerNormalization()(x)
    model = tf.keras.Model(input_layer, out)
    x_val = tf.random.normal((batch_size, *input_shape))
    return model, x_val


def func_model_with_custom_subclass(batch_size):
    input_shape = (3, 13)
    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size, name="input_1")
    out = CustomLayer(mock_custom_param="Mock!")(input_layer)
    model = tf.keras.Model(input_layer, out)
    x_val = tf.random.normal((batch_size, *input_shape))
    return model, x_val


def func_model_with_multiple_custom_subclass_and_heads(batch_size):
    input_shape = (3, 13)
    input_layer = tf.keras.Input(shape=input_shape, batch_size=batch_size, name="input_1")
    x1 = SimpleCustomLayer(name="custom_1")(input_layer)
    x2 = SimpleCustomLayer(name="custom_2")(input_layer)
    x = tf.keras.layers.Add(name="add")([x1, x2])
    out1 = tf.keras.layers.Dense(4, name="out_1")(x)
    out2 = tf.keras.layers.Dense(5, name="out_2")(x)
    model = tf.keras.Model(input_layer, [out1, out2])
    x_val = tf.random.normal((batch_size, *input_shape))
    return model, x_val


def func_model_tf_bert_embeddings(batch_size, len_seq, config):
    sample = create_sample(batch_size, len_seq)
    model = EmbeddingModel(config, TFBertEmbeddings)
    model = convert_to_functional(model, sample)
    return model, sample


def func_model_tf_bert_lm_prediction_head(batch_size, len_seq, config, model):
    sample = np.random.rand(batch_size, len_seq, config.hidden_size)
    model = LMPredictionHeadModel(config, TFBertLMPredictionHead)
    model = convert_to_functional(model, sample)
    return model, sample


def func_model_tf_bert_self_output(batch_size, len_seq, config):
    sample = np.random.rand(batch_size, len_seq, config.hidden_size)
    model = TFBertSelfOutputModel(config, TFBertSelfOutput, batch_size, len_seq)
    model = convert_to_functional(model, sample)
    return model, sample


def func_model_tf_bert_self_attention(batch_size, len_seq, config):
    sample = np.random.rand(batch_size, len_seq, config.hidden_size)
    model = SelfAttentionModel(config, TFBertSelfAttention)
    model = convert_to_functional(model, sample)
    return model, sample


def recursive_check_layer(layer, assert_fn):
    if layer.submodules:
        for submodule in layer.submodules:
            recursive_check_layer(submodule, assert_fn)
    assert_fn(layer)


def count_layers(layers):
    num_layers_class = defaultdict(int)
    for l in layers:
        num_layers_class[type(l)] += 1
    return num_layers_class


def check_replace_layers(to_replace_dict, model_func, model_func_args=dict(), batch_size=32, decimal=5):
    def assert_fn(l):
        assert type(l) not in to_replace_dict

    set_random_seeds()
    model, x_val = model_func(batch_size=batch_size, **model_func_args)
    model.compile()
    layers_per_type = count_layers(model.submodules)
    y = model(x_val)

    model_replacing = ModelReplacing(to_replace_dict)
    new_outputs, _ = model_replacing.get_outputs_processed_model(model, "all")
    new_model = tf.keras.Model(model.inputs, new_outputs)
    new_model.compile()
    layers_per_type_2 = count_layers(new_model.submodules)
    y_2 = new_model(x_val)
    tf.debugging.assert_near(y, y_2, atol=10**-decimal)
    assert len(layers_per_type) == len(layers_per_type_2)
    for t in layers_per_type:
        if t in to_replace_dict:
            new_class_t = to_replace_dict[t]["new_class"]
            assert layers_per_type[t] == layers_per_type_2[new_class_t]
        else:
            assert layers_per_type[t] == layers_per_type_2[t]
    for layer in new_model.layers:
        recursive_check_layer(layer, assert_fn)


@pytest.mark.parametrize(
    "func, expected_names",
    [
        (
            func_model_with_custom_subclass,
            [
                "input_1",
                "custom_layer.dense_1",
                "custom_layer.dropout_1",
                "custom_layer.norm_1",
                "custom_layer.dense_2",
                "custom_layer.dropout_2",
                "custom_layer.norm_2",
                "custom_layer.dense_3",
                "custom_layer.norm_3",
            ],
        ),
        (
            func_model_with_multiple_custom_subclass_and_heads,
            ["input_1", "custom_1", "custom_2", "add", "out_1", "out_2"],
        ),
    ],
)
def test_model_expansion(func, expected_names):
    model, x_val = func(batch_size=32)
    expanded_model = ModelExpansion().process_model(CustomLayer, deepcopy(model))
    num_model_params = count_params(model.trainable_weights)
    num_expanded_model_params = count_params(expanded_model.trainable_weights)
    assert num_model_params == num_expanded_model_params
    names = [layer.name for layer in expanded_model.layers]
    assert names == expected_names


@pytest.mark.parametrize(
    "func",
    [
        func_model_with_dense,
        func_model_with_dense_dropout_and_layer_normalization,
        func_model_with_no_dense,
    ],
)
def test_replace_standard_layers(func):
    to_replace_dict = {
        Dense: {
            "new_class": SerialDense,
            "new_params": {"serialization_factor": 2, "serialization_dimension": "kernel_rows"},
            "copy_weights": True,
        },
        Dropout: {"new_class": IpuDropout},
        LayerNormalization: {"new_class": IpuLayerNormalization, "copy_weights": True},
    }

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        check_replace_layers(to_replace_dict, func)


def test_replace_standard_layers_in_custom_layer():
    to_replace_dict = {
        Dense: {
            "new_class": SerialDense,
            "new_params": {"serialization_factor": 2, "serialization_dimension": "kernel_rows"},
            "copy_weights": True,
        },
        Dropout: {"new_class": IpuDropout},
        LayerNormalization: {"new_class": IpuLayerNormalization},
    }

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        check_replace_layers(to_replace_dict, func_model_with_custom_subclass)


@pytest.mark.parametrize(
    "func",
    [
        func_model_with_dense,
        func_model_with_dense_dropout_and_layer_normalization,
        func_model_with_no_dense,
    ],
)
def test_replace_standard_layers_with_extended_keras_layers(func):
    to_replace_dict = {Dropout: {"new_class": IpuDropoutCustom}}

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        check_replace_layers(to_replace_dict, func)


def test_replace_custom_layer():
    batch_size = 32

    def copy_weights_custom_layer_fp16(layer, new_layer):
        input_layer = tf.keras.Input(shape=layer.input_shape[1:], batch_size=batch_size)
        new_layer(input_layer)
        new_layer.set_weights(layer.get_weights())

    to_replace_dict = {
        CustomLayer: {
            "new_class": CustomLayerFp16,
            "copy_weights": True,
            "copy_weights_func": copy_weights_custom_layer_fp16,
        }
    }
    check_replace_layers(to_replace_dict, func_model_with_custom_subclass, batch_size=batch_size, decimal=2)


def test_replace_embeddings():
    """
    Replace a model with a TFBertEmbeddings layer, which although it has extra arguments in init, it
    doesn't have overwritten the get_config() method.
    :return: None
    """
    batch_size = 2
    sequence_length = 128
    config = BertConfig()

    def copy_weights_tf_bert_embeddings(layer, new_layer):
        copy_weights_layer_with_input_shape_hidden_states_func(layer, new_layer, batch_size, sequence_length)

    set_random_seeds()
    to_replace_dict = {
        TFBertEmbeddings: {
            "new_class": IpuTFBertEmbeddings,
            "new_params": {
                "config": config,
                "serialization_factor": 2,
            },
            "copy_weights": True,
            "copy_weights_func": copy_weights_tf_bert_embeddings,
        }
    }
    model_func_args = {"len_seq": sequence_length, "config": config}

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        check_replace_layers(to_replace_dict, func_model_tf_bert_embeddings, model_func_args, batch_size)


def test_replace_lm_prediction_head():
    """
    Replace a model with a TFBertLMPredictionHead layer.
    :return: None
    """
    batch_size = 2
    sequence_length = 128
    use_cls_layer = True
    use_prediction_bias = True
    config = BertConfig()
    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        model = LMPredictionHeadModel(config, TFBertLMPredictionHead)

        def copy_weights_tf_bert_lm_prediction_head(layer, new_layer):
            copy_lm_prediction_head_weights_func(
                layer, new_layer, batch_size, sequence_length, use_cls_layer, use_prediction_bias
            )

        set_random_seeds()
        to_replace_dict = {
            TFBertLMPredictionHead: {
                "new_class": IpuTFBertLMPredictionHead,
                "new_params": {
                    "config": config,
                    "input_embeddings": lambda: model.embedding,
                    "use_prediction_bias": use_prediction_bias,
                    "use_cls_layer": use_cls_layer,
                    "serialization_factor": 2,
                },
                "copy_weights": True,
                "copy_weights_func": copy_weights_tf_bert_lm_prediction_head,
            },
        }
        model_func_args = {"len_seq": sequence_length, "config": config, "model": model}
        check_replace_layers(to_replace_dict, func_model_tf_bert_lm_prediction_head, model_func_args, batch_size)


@pytest.mark.parametrize("use_qkv_bias_flag, use_qkv_split_flag", [(True, True), (False, True)])
def test_replace_self_attention(use_qkv_bias_flag, use_qkv_split_flag):
    """
    Replace a model with a TFBertSelfAttention layer.
    :return: None
    """
    batch_size = 2
    sequence_length = 128
    use_qkv_bias = use_qkv_bias_flag
    use_qkv_split = use_qkv_split_flag
    config = BertConfig()
    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        def copy_weights_tf_bert_self_attention(layer, new_layer):
            copy_self_attention_weights_func(layer, new_layer, use_qkv_bias, use_qkv_split)

        set_random_seeds()
        to_replace_dict = {
            TFBertSelfAttention: {
                "new_class": IpuTFBertSelfAttention,
                "new_params": {"config": config, "use_qkv_bias": use_qkv_bias, "use_qkv_split": use_qkv_split},
                "copy_weights": True,
                "copy_weights_func": copy_weights_tf_bert_self_attention,
            },
        }
        model_func_args = {"len_seq": sequence_length, "config": config}
        check_replace_layers(to_replace_dict, func_model_tf_bert_self_attention, model_func_args, batch_size)


def test_replace_self_output():
    """
    Replace a model with a TFBertSelfOutput layer.
    :return: None
    """
    batch_size = 2
    sequence_length = 128
    use_projection_bias = True
    config = BertConfig()
    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():

        def copy_weights_self_output(layer, new_layer):
            copy_self_output_weights_func(layer, new_layer, batch_size, sequence_length, use_projection_bias)

        set_random_seeds()
        to_replace_dict = {
            TFBertSelfOutput: {
                "new_class": IpuTFBertSelfOutput,
                "new_params": {
                    "config": config,
                    "use_projection_bias": use_projection_bias,
                },
                "copy_weights": True,
                "copy_weights_func": copy_weights_self_output,
            },
        }
        model_func_args = {"len_seq": sequence_length, "config": config}
        check_replace_layers(to_replace_dict, func_model_tf_bert_self_output, model_func_args, batch_size)


def check_replaced_layers_are_trackable(to_replace_dict, model_func, model_func_args=dict(), batch_size=32):
    model, _ = model_func(batch_size=batch_size, **model_func_args)

    def append_id(layer, list_of_layers, to_replace_dict, cond):
        if layer.submodules:
            for submodule in layer.submodules:
                if cond(to_replace_dict, submodule):
                    list_of_layers.append(id(submodule))
                append_id(submodule, list_of_layers, to_replace_dict, cond)

    list_of_layers_for_replacing = []
    append_id(
        model,
        list_of_layers_for_replacing,
        to_replace_dict,
        lambda to_replace_dict, submodule: any(isinstance(submodule, key) for key in to_replace_dict),
    )

    model_replacing = ModelReplacing(to_replace_dict)
    new_outputs, _ = model_replacing.get_outputs_processed_model(model, "all")
    new_model = tf.keras.Model(model.inputs, new_outputs)

    list_of_new_layers = []
    append_id(
        new_model,
        list_of_new_layers,
        to_replace_dict,
        lambda to_replace_dict, submodule: any(isinstance(submodule, v["new_class"]) for v in to_replace_dict.values()),
    )

    for trackable_reference in new_model._trackable_saver._graph_view._breadth_first_traversal()[0]:
        assert id(trackable_reference) not in list_of_layers_for_replacing
    for new_layer in list_of_new_layers:
        assert new_layer in [
            id(t_ref) for t_ref in new_model._trackable_saver._graph_view._breadth_first_traversal()[0]
        ]


@pytest.mark.parametrize(
    "func",
    [
        func_model_with_no_dense,
        func_model_with_dense,
        func_model_with_custom_subclass,
        func_model_with_dense_dropout_and_layer_normalization,
    ],
)
def test_replaced_standard_layer_is_trackable(func):
    to_replace_dict = {LayerNormalization: {"new_class": IpuLayerNormalization}}

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        check_replaced_layers_are_trackable(to_replace_dict, func)


def test_replaced_custom_layer_is_trackable():
    batch_size = 2
    sequence_length = 128
    config = BertConfig()

    def copy_weights_tf_bert_embeddings(layer, new_layer):
        new_layer.build((batch_size, sequence_length, new_layer.hidden_size))
        new_layer.set_weights(layer.get_weights())

    to_replace_dict = {
        TFBertEmbeddings: {
            "new_class": IpuTFBertEmbeddings,
            "new_params": {
                "config": config,
                "serialization_factor": 2,
            },
            "copy_weights": True,
            "copy_weights_func": copy_weights_tf_bert_embeddings,
        }
    }
    model_func_args = {"len_seq": sequence_length, "config": config}

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        check_replaced_layers_are_trackable(to_replace_dict, func_model_tf_bert_embeddings, model_func_args, batch_size)


@pytest.mark.parametrize(
    "func",
    [
        func_model_with_no_dense,
        func_model_with_dense,
        func_model_with_custom_subclass,
    ],
)
@pytest.mark.parametrize(
    "to_outline_dict",
    [
        {Dense: {"outline_kwargs": {}}, Dropout: {"outline_kwargs": {}}},
        {Dense: {"outline_kwargs": {"unique_sharding": True}}},
    ],
)
def test_outline_layer(func, to_outline_dict):
    set_random_seeds()
    batch_size = 32

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model, x_val = func(batch_size)
        model.compile()
        model.summary()
        y = model.predict(x_val)

        model_outlining = ModelOutlining(to_outline_dict)
        new_outputs, new_names = model_outlining.get_outputs_processed_model(model, "all")
        new_model = tf.keras.Model(model.inputs, new_outputs)
        model_outlining.rename_outputs(new_names, new_model)
        new_model.compile()
        new_model.summary()
        y2 = new_model.predict(x_val)

    np.testing.assert_almost_equal(y, y2)

    def assert_fn(layer):
        if any(isinstance(layer, to_outline) for to_outline in to_outline_dict):
            assert ModelOutlining.outline_layer_inplace.__qualname__ in layer.call.__qualname__

    for layer in new_model.layers:
        recursive_check_layer(layer, assert_fn)


@pytest.mark.parametrize(
    "func",
    [
        func_model_with_no_dense,
        func_model_with_dense,
        func_model_with_custom_subclass,
    ],
)
@pytest.mark.parametrize(
    "to_checkpoint",
    [
        [Dense, Dropout],
        [Dense],
    ],
)
def test_insert_recomputation_checkpoint_after_layer(func, to_checkpoint):
    set_random_seeds()
    batch_size = 32

    cfg = ipu.config.IPUConfig()
    cfg.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        model, x_val = func(batch_size)
        model.compile()
        model.summary()
        y = model.predict(x_val)

        model_outlining = ModelAddRecomputationCheckpoints(to_checkpoint)
        new_outputs, new_names = model_outlining.get_outputs_processed_model(model, "all")
        new_model = tf.keras.Model(model.inputs, new_outputs)
        model_outlining.rename_outputs(new_names, new_model)
        new_model.compile()
        new_model.summary()
        y2 = new_model.predict(x_val)

    np.testing.assert_almost_equal(y, y2)

    def assert_fn(layer):
        # Tests output of layer is from a recomputation checkpoint operation
        look_for_op_containing = "add_recomputation_checkpoint"
        if any(isinstance(layer, checkpointed) for checkpointed in to_checkpoint):
            assert look_for_op_containing in layer.call.__qualname__
        else:
            assert look_for_op_containing not in layer.call.__qualname__

    for layer in new_model.layers:
        recursive_check_layer(layer, assert_fn)
