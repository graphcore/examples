# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tensorflow import keras
from typing import Callable


def edit_functional_model(model: keras.Model, user_fn: Callable, copy_weights: bool = True):

    generated_tensors = {}

    def __copy_layer(layer):
        layer_copy = layer.from_config(layer.get_config())
        layer_copy._name = layer._name
        return layer_copy

    def __apply_fn(original_layer, inputs):
        copied_layer = __copy_layer(original_layer)
        inputs = inputs[0] if len(inputs) == 1 else inputs

        new_layer_output = user_fn(copied_layer, inputs)
        if new_layer_output is not None:
            return new_layer_output

        return copied_layer(inputs) if not isinstance(copied_layer, keras.layers.InputLayer) else inputs

    def __get_tensor_from_layer(tensor):
        if tensor.ref() in generated_tensors.keys():
            return generated_tensors[tensor.ref()]

        prev_layer = tensor._keras_history.layer
        input_tensors = prev_layer.input if isinstance(prev_layer.input, list) else [prev_layer.input]

        if not isinstance(prev_layer, keras.layers.InputLayer):
            input_tensors = [__get_tensor_from_layer(input_tensor) for input_tensor in input_tensors]

        new_output_tensors = __apply_fn(prev_layer, input_tensors)
        if isinstance(new_output_tensors, tuple):
            for new_output_tensor, old_output_tensor in zip(new_output_tensors, prev_layer.output):
                generated_tensors[old_output_tensor.ref()] = new_output_tensor
        else:
            generated_tensors[tensor.ref()] = new_output_tensors
        return generated_tensors[tensor.ref()]

    def __copy_weights(src_model, dest_model):
        i, j = 0, 0

        while i < len(dest_model.layers) and j < len(src_model.layers):
            found_layer = False

            for k in range(j, len(src_model.layers)):
                if (dest_model.layers[i].name) == src_model.layers[k].name:
                    dest_model.layers[i].set_weights(src_model.layers[k].get_weights())
                    found_layer = True
                    i, j = i + 1, k + 1

            if not found_layer:
                i += 1

        return dest_model

    model_outputs = []

    for output in model.outputs:
        new_output = __get_tensor_from_layer(output)
        model_outputs.append(new_output)

    new_model = keras.Model(model.inputs, model_outputs)
    if copy_weights:
        new_model = __copy_weights(model, new_model)

    return new_model
