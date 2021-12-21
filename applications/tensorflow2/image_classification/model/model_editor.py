# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from tensorflow import keras
import logging


class ModelEditor:

    def __init__(self, model):
        self.model = model

    def update_model_with_func(self, user_fn, copy_weights=True):
        self.user_fn = user_fn
        self.generated_tensors = {}
        model_outputs = []

        for output in self.model.outputs:
            new_output = self.get_tensor_from_layer(output)
            model_outputs.append(new_output)

        new_model = keras.Model(self.model.inputs, model_outputs)
        if copy_weights:
            new_model = self.__copy_weights(new_model)

        self.generated_tensors.clear()
        self.model = new_model

        return self.model

    def get_tensor_from_layer(self, tensor):

        if tensor.ref() in self.generated_tensors.keys():
            return self.generated_tensors[tensor.ref()]

        prev_layer = tensor._keras_history.layer
        input_tensors = prev_layer.input if isinstance(prev_layer.input, list) else [prev_layer.input]
        if not isinstance(prev_layer, keras.layers.InputLayer):
            input_tensors = [self.get_tensor_from_layer(input_tensor) for input_tensor in input_tensors]

        new_output_tensors = self.__apply_fn(prev_layer, input_tensors, self.user_fn)
        if isinstance(new_output_tensors, tuple):
            for new_output_tensor, old_output_tensor in zip(new_output_tensors, prev_layer.output):
                self.generated_tensors[old_output_tensor.ref()] = new_output_tensor
        else:
            self.generated_tensors[tensor.ref()] = new_output_tensors
        return self.generated_tensors[tensor.ref()]

    def __apply_fn(self, original_layer, inputs, user_fn):
        copied_layer = ModelEditor.__copy_layer(original_layer)
        inputs = inputs[0] if len(inputs) == 1 else inputs

        new_layer_output = user_fn(copied_layer, inputs)
        if new_layer_output is not None:
            return new_layer_output

        return copied_layer(inputs) if not isinstance(copied_layer, keras.layers.InputLayer) else inputs

    @staticmethod
    def __copy_layer(layer):
        layer_copy = layer.from_config(layer.get_config())
        layer_copy._name = layer._name
        return layer_copy

    def __copy_weights(self, new_model):
        i, j = 0, 0
        self.not_copied_weights = []

        while i < len(new_model.layers) and j < len(self.model.layers):
            found_layer = False

            for k in range(j, len(self.model.layers)):
                if (new_model.layers[i].name) == self.model.layers[k].name:
                    new_model.layers[i].set_weights(self.model.layers[k].get_weights())
                    found_layer = True
                    i, j = i + 1, k + 1

            if not found_layer:
                i += 1

        return new_model
