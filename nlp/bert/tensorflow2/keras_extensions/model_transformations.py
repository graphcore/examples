# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from collections import defaultdict
import inspect
import logging

import numpy as np
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from keras.utils import tf_inspect
from tensorflow.python import ipu
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework.tensor_spec import TensorSpec

logger = logging.getLogger(__name__)

KERAS_STANDARD_LAYERS = tuple(x[0] for x in inspect.getmembers(tf.keras.layers, inspect.isclass) if x[0] != "Layer")


def get_keras_input_from_numpy_and_tf(dataset, name=None):
    strategy = distribution_strategy_context.get_strategy()
    return tf.keras.Input(
        shape=dataset.shape[1:],
        batch_size=dataset.shape[0] * strategy.num_replicas_in_sync,
        name=name,
        dtype=dataset.dtype,
    )


def get_keras_input_from_tf_dataset(dataset):
    strategy = distribution_strategy_context.get_strategy()
    return tf.keras.Input(
        shape=dataset.element_spec.shape[1:],
        batch_size=dataset.element_spec.shape[0] * strategy.num_replicas_in_sync,
        name=dataset.element_spec.name,
        dtype=dataset.element_spec.dtype,
    )


def get_keras_input_from_dict(dataset):
    input_layers = dict()
    for key, val in dataset.items():
        if isinstance(val, (np.ndarray, tf.Tensor)):
            input_layers[key] = get_keras_input_from_numpy_and_tf(val, key)
        else:
            raise TypeError(
                f"Not supported data scheme. Expected `np.ndarray`, but received {type(val)}. "
                f"You might have to implement this case."
            )
    return input_layers


def get_keras_input_from_dict_of_tf_datasets(dataset):
    strategy = distribution_strategy_context.get_strategy()
    input_layers = dict()
    for key, val in dataset.items():
        assert isinstance(val, TensorSpec), (
            f"Not supported data schema, expected `TensorSpec` but "
            f"received `{type(val)}`. You have might to implement this case."
        )
        input_layers[key] = tf.keras.Input(
            shape=val.shape[1:], batch_size=val.shape[0] * strategy.num_replicas_in_sync, name=key, dtype=val.dtype
        )
    return input_layers


def get_input_layers(dataset):
    """
    Gets a dataset and returns a symbolic input that matches the specs in the dataset.
    :param dataset: Dataset used to train the model.
    :return: Symbolic Keras tensors with same specs as the actual dataset.
    """
    if isinstance(dataset, np.ndarray):
        return get_keras_input_from_numpy_and_tf(dataset)
    if isinstance(dataset, tf.data.Dataset) and not isinstance(dataset.element_spec, tuple):
        return get_keras_input_from_tf_dataset(dataset)
    if isinstance(dataset, tf.data.Dataset) and isinstance(dataset.element_spec, tuple):
        training_spec = dataset.element_spec[0]  # dataset.element_spec is a tuple (training, label) sample.
        if isinstance(training_spec, dict):
            return get_keras_input_from_dict_of_tf_datasets(dataset.element_spec[0])
    if isinstance(dataset, dict):
        return get_keras_input_from_dict(dataset)
    raise ValueError(f"Not supported dataset.")


def convert_to_functional(model_subclass, dataset):
    """
    Convert original subclass model to a functional one. We pass the flag `training=None` for subclass models whose
    layers pass the `training` flag to other layers explicitly, so if their default value was `training=False` this will
    be overwritten by None, such that the layers can infer the training mode from the context.
    :param model_subclass: Keras model of a custom class, which is neither Functional, nor Sequential.
    :param dataset: Dataset used to train the model.
    :return: Functional Keras model.
    """
    input_layers = get_input_layers(dataset)
    if "training" in tf_inspect.getfullargspec(model_subclass.call).args:
        model_func = tf.keras.Model(input_layers, model_subclass.call(input_layers, training=None))
    else:
        model_func = tf.keras.Model(input_layers, model_subclass.call(input_layers))
    return model_func


class ModelOptimization:
    def __init__(self):
        pass

    @staticmethod
    def add_parent_prefix_to_sublayers(model):
        """
        When expanding a custom layer, it is easy to loose track of where the new sublayers come from. In addition, if
        we have multiple layers of the same type, they can result in multiple sublayers with the same name. This
        function avoids this problem by adding the parent layer as prefix to the sublayer names.
        :param model: Model whose layers will be renamed with parent prefix.
        :return: None
        """
        for layer in model.layers:
            for attr_key, attr_val in layer.__dict__.items():
                if isinstance(attr_val, tf.keras.layers.Layer):
                    if layer.name not in attr_val.name:
                        attr_val._name = layer.name + "." + attr_val.name
        return

    @staticmethod
    def adjust_training_flag(kwargs):
        """
        For models that have been converted from subclass models, and whose layers pass the training flag to other
        layers explicitly, and that had a default value set to False instead of None, require to adjust this default
        value to None, so that it will take the value from the context.
        """
        if "training" in kwargs:
            kwargs["training"] = None
        return kwargs

    @staticmethod
    def check_custom_layer_id(custom_layer_id):
        if not (isinstance(custom_layer_id, str) or inspect.isclass(custom_layer_id)):
            raise ValueError(f"Argument custom_layer_id={custom_layer_id} must be a string or a class")

    @staticmethod
    def get_in_out_names_of_every_layer(model):
        """
        Find the input layers of each layer to describe the model as a graph with nodes and edges.
        :param model: Model to be explored and described as a graph.
        :return: (graph_in_edges, graph_out_edges, input_tensors)
            - graph_in_edges: Dict describing the graph as nodes (i.e., layer names) and the lists of their incoming
                edges (i.e., names of layers whose outputs are inputs to this layer).
            - graph_out_edges:  Dict describing the graph as nodes (i.e., layer names) and the lists of their outgoing
                edges (i.e., names of layers whose inputs include the the outputs of this layer).
            - input_tensors: Dict with the layer names and their symbolic Keras tensors inputs.
        """
        names = list()
        for layer in model.layers:
            names.append(layer.name)

        graph_in_edges = defaultdict(list)
        graph_out_edges = defaultdict(list)
        input_tensors = defaultdict(list)

        for layer in model.layers:
            for node in layer.outbound_nodes:
                outbound_name = node.outbound_layer.name
                if outbound_name in names and outbound_name not in graph_out_edges[layer.name]:
                    # Clear output avoiding duplication and phantom nodes.
                    graph_in_edges[outbound_name].append(layer.name)
                    input_tensors[outbound_name].append(layer)
                    graph_out_edges[layer.name].append(outbound_name)
        return graph_in_edges, graph_out_edges, input_tensors

    @staticmethod
    def get_input_layer(layer_name, input_tensors, graph_in_edges, graph_out_edges, post_process_fn):
        """
        Get all the symbolic Keras tensors that are input to any given layer.
        :param layer_name: Name of the layer for which we want to get its input layers.
        :param input_tensors: Dict of layers that are input for every other layer.
        :param graph_in_edges: Dict with names of input layers for every layer.
        :param graph_out_edges: Dict with names of output layers for every layer.
        :param post_process_fn: Function to postprocess the input layer for specific layers.
        :return: Symbolic Keras tensor(s) that are input to the layer given by layer_name.
        """
        input_layer = [graph_out_edges[key] for key in graph_in_edges[layer_name]]
        if len(input_layer) == 1:
            input_layer = input_layer[0]
        if post_process_fn is not None:
            input_layer = post_process_fn(input_layer, layer_name, input_tensors, graph_in_edges)
        return input_layer

    def get_layer_output(self, layer, args=None, kwargs=None):
        if isinstance(layer, tf.keras.layers.InputLayer):
            return layer.output
        else:
            kwargs = self.adjust_training_flag(kwargs)
            logger.debug(f"Getting layer {layer} output passing args {args} and" f" kwargs {kwargs} to call()")
            return layer(*args, **kwargs)

    @staticmethod
    def get_layer_simplified_name(layer_name):
        return layer_name[layer_name.rfind("/") + 1 :]

    def get_expected_input_layer_item(self, item, expected_name, expected_shape):
        if isinstance(item, KerasTensor):
            layer_var = item
        elif isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], KerasTensor):
            layer_var = item[0]
        else:
            raise ValueError(f"Not implemented: len(input_layer) == 1, but provided {len(item)}.")

        name = self.get_layer_simplified_name(layer_var.name)
        match_name = False
        match_shape = False
        if name == expected_name:
            match_name = True
        if layer_var.shape == expected_shape:
            match_shape = True

        return match_name, match_shape, layer_var

    def get_outputs_processed_model(self, initial_model, custom_layer_id, post_process_fn=None):
        """
        Take an initial model with a custom layer and process the layer. The process could be expanding a custom subclass
        layer or replacing a layer with another one.
        :param initial_model: Model being processed.
        :param custom_layer_id: Identifier of the layer(s) to be processed, currently supported name or class as valid ids.
        :param post_process_fn: Function to process the output of the input layer to the layer being processed.
        :return: Output tensor of the new model.
        """
        self.check_custom_layer_id(custom_layer_id)

        out_tensor_of = dict()
        custom_model_outputs_dict = dict()
        output_names = dict()

        graph_in_edges, graph_out_edges, input_tensors = self.get_in_out_names_of_every_layer(initial_model)

        graph_out_edges["root"] = [name for name in initial_model.input_names]
        order = self.postorder(graph_out_edges, "root")

        for v in order:
            layer = initial_model.get_layer(v)
            input_layer = self.get_input_layer(v, input_tensors, graph_in_edges, out_tensor_of, post_process_fn)
            args = layer.inbound_nodes[0].call_args
            kwargs = layer.inbound_nodes[0].call_kwargs
            new_args, new_kwargs = self.replace_args_with_new_tensor(args, kwargs, input_layer)

            if self.is_to_be_processed(v, layer, custom_layer_id):
                out_tensor_of[v] = self.process_layer(layer, new_args, new_kwargs, parent=initial_model)
            else:
                out_tensor_of[v] = self.get_layer_output(layer, new_args, new_kwargs)

            if v in initial_model.output_names:
                custom_model_outputs_dict[v] = out_tensor_of[v]
                if v == custom_layer_id:
                    self.update_output_names(output_names, out_tensor_of[v].name, v)

        new_outputs = [custom_model_outputs_dict[name] for name in initial_model.output_names]

        return new_outputs, output_names

    @staticmethod
    def is_custom_layer(layer):
        return not type(layer).__name__ in KERAS_STANDARD_LAYERS

    @staticmethod
    def is_name_or_class_match(name, layer, custom_layer_id):
        if isinstance(custom_layer_id, str) and name == custom_layer_id:
            return True
        if inspect.isclass(custom_layer_id) and isinstance(layer, custom_layer_id):
            return True
        return False

    def is_to_be_processed(self, name, layer, custom_layer_id):
        return False

    @staticmethod
    def postorder(graph, root):
        """
        Traverse the graph with DFS and return the order of nodes such that every node goes after all its inputs.
        :param graph: Graph described as a dict with nodes and the lists of their outgoing edges.
        :param root: Root node.
        :return: List with the sorted nodes.
        """
        explored = set()
        order = list()

        def dfs(node):
            explored.add(node)
            for child in graph[node]:
                if child not in explored:
                    dfs(child)
            order.append(node)

        dfs(root)
        order.pop()
        order.reverse()
        return order

    def process_layer(self, layer, layer_args, layer_kwargs, **kwargs):
        raise NotImplementedError("This method should be implemented by its child classes.")

    def process_model(self, custom_layer_id, initial_model, input_process_fn=None):
        """
        Main method of the class that processes the layers of a model as specified by the child classes model.
        :param custom_layer_id: String or class identifying the layers to be expanded or replaced.
        :param initial_model: Model to be processed.
        :param input_process_fn: Optional function handler to choose the output of the input layer for custom layers.
        :return: New model with expanded layer and same output names.
        """
        self.add_parent_prefix_to_sublayers(initial_model)
        new_outputs, new_names = self.get_outputs_processed_model(initial_model, custom_layer_id, input_process_fn)
        new_model = tf.keras.Model(initial_model.inputs, new_outputs)
        self.rename_outputs(new_names, new_model)
        return new_model

    @staticmethod
    def rename_outputs(new_output_names, model):
        """
        Rename model outputs.
        :param new_output_names: Dictionary with keys being the names of the output layers after expansion and values being
            the name of the output layer before expansion.
        :param model: Keras model after expansion.
        :return: Model with renamed layers and outputs.
        """
        for new, old in new_output_names.items():
            if new in model.output_names:
                layer = model.get_layer(new)
                layer._name = old
                model.output_names[model.output_names.index(new)] = old
        return model

    def replace_args_with_new_tensor(self, args, kwargs, input_layer):
        new_args = self.replace_args(args, input_layer)
        new_kwargs = self.replace_kwargs(kwargs, input_layer)
        return new_args, new_kwargs

    def replace_args(self, args, input_layer):
        new_args = [self.update_arg_with_input_layer(arg, input_layer) for arg in args]
        return new_args

    def replace_kwargs(self, kwargs, input_layer):
        new_kwargs = {key: self.update_arg_with_input_layer(val, input_layer) for key, val in kwargs.items()}
        return new_kwargs

    def update_arg_with_input_layer(self, arg, input_layer):
        if not isinstance(arg, KerasTensor):
            return arg

        expected_name = self.get_layer_simplified_name(arg.name)
        expected_shape = arg.shape

        if isinstance(input_layer, KerasTensor):
            return self.update_arg_with_input_layer_keras_tensor(input_layer, expected_name, expected_shape)

        if isinstance(input_layer, (list, tuple)):
            return self.update_arg_with_input_layer_tuple(input_layer, expected_name, expected_shape)

        return self.update_arg_with_input_layer_custom_object(input_layer, expected_name, expected_shape)

    def update_arg_with_input_layer_keras_tensor(self, input_layer, expected_name, expected_shape):
        name = self.get_layer_simplified_name(input_layer.name)
        if expected_name != name:
            logger.debug(f"Input layer name `{name}` different from expected `{expected_name}` by call method.")
        shape = input_layer.shape
        assert shape == expected_shape, f"Input layer shape ({shape}) different from expected ({expected_shape})."
        return input_layer

    def update_arg_with_input_layer_tuple(self, input_layer, expected_name, expected_shape):
        candidates = list()
        for layer in input_layer:
            match_name, match_shape, item = self.get_expected_input_layer_item(layer, expected_name, expected_shape)
            if match_name and match_shape:
                return item
            if match_shape:
                candidates.append(item)

        # If there is a single candidate, use it.
        if len(candidates) == 0:
            raise ValueError(f"No input layer have the expected shape ({expected_shape}).")
        if len(candidates) > 1:
            raise ValueError(f"Not possible to identify the input layer have the expected shape({expected_shape}).")
        return candidates[0]

    def update_arg_with_input_layer_custom_object(self, input_layer, expected_name, expected_shape):
        candidates = list()
        input_layer_dict = input_layer.__dict__
        for key, val in input_layer_dict.items():
            match_name, match_shape, item = self.get_expected_input_layer_item(val, expected_name, expected_shape)
            if match_name and match_shape:
                input_layer_dict[key] = item
                return input_layer
            if match_shape:
                candidates.append(item)

        # If there is a single candidate, use it.
        if len(candidates) == 0:
            raise ValueError(f"No input layer have the expected shape ({expected_shape}).")
        if len(candidates) > 1:
            raise ValueError(f"Not possible to identify the input layer have the expected shape({expected_shape}).")
        input_layer_dict[key] = candidates[0]
        return input_layer

    @staticmethod
    def update_output_names(output_names_dict, layer_name, old_name):
        new_name = layer_name[: layer_name.find("/")]
        return output_names_dict.update({new_name: old_name})


class ModelExpansion(ModelOptimization):
    def is_to_be_processed(self, name, layer, custom_layer_id):
        if self.is_custom_layer(layer):
            if custom_layer_id == "all":
                return True
            else:
                return self.is_name_or_class_match(name, layer, custom_layer_id)
        return False

    def process_layer(self, layer, layer_args, layer_kwargs, **kwargs):
        """
        Capture the trace of a symbolic Keras tensor when it is propagated through the layer's call method.
        :param layer: Layer to be expanded.
        :param layer_args: Arguments needed for the call method.
        :param layer_kwargs: Other arguments needed for the call method.
        :return: Trace at the output of the call method.
        """
        layer_kwargs = self.adjust_training_flag(layer_kwargs)
        return layer.call(*layer_args, **layer_kwargs)


class ModelLayerModification(ModelOptimization):
    def __init__(self):
        self.updated_values = dict()
        super().__init__()

    @staticmethod
    def get_layer_properties_as_dict(obj):
        return obj.__dict__

    def is_to_be_processed(self, name, layer, custom_layer_id):
        if custom_layer_id == "all":
            return True
        return self.is_name_or_class_match(name, layer, custom_layer_id)

    def process_layer(self, layer, layer_args, layer_kwargs, **kwargs):
        raise NotImplementedError("This method should be implemented by each child class.")

    def recursively_search_layers(self, layer, to_update_params, update_func, **kwargs):
        logger.debug(f"Searching layer: {layer}")
        if type(layer) in to_update_params:
            if isinstance(to_update_params, (list, tuple)):
                return update_func(layer, **kwargs)
            return update_func(layer, to_update_params, **kwargs)

        elif self.is_custom_layer(layer):
            properties = self.get_layer_properties_as_dict(layer)
            for key, val in properties.items():
                if isinstance(val, tf.keras.layers.Layer):
                    properties[key] = self.update_property(val, to_update_params, update_func, parent=layer)
                elif isinstance(val, (list, tuple)):
                    for i in range(len(val)):
                        if isinstance(val[i], tf.keras.layers.Layer):
                            properties[key][i] = self.update_property(
                                val[i], to_update_params, update_func, parent=layer
                            )
                elif isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, tf.keras.layers.Layer):
                            properties[key][k] = self.update_property(
                                val[k], to_update_params, update_func, parent=layer
                            )
        return layer

    def search_layers(self, layer, layer_args, layer_kwargs, to_update_params, update_func, **kwargs):
        new_layer = self.recursively_search_layers(layer, to_update_params, update_func, **kwargs)
        new_output = self.get_layer_output(new_layer, layer_args, layer_kwargs)
        return new_output

    def update_property(self, val, to_update_params, update_func, **kwargs):
        if val in self.updated_values:
            return self.updated_values[val]
        else:
            new_layer = self.recursively_search_layers(
                val,
                to_update_params,
                update_func,
                **kwargs,
            )
            self.updated_values[val] = new_layer
            return new_layer


def copy_weights_dense(layer, new_layer):
    input_shape = (None, layer.weights[0].shape[0])
    new_layer.build(input_shape)
    new_layer.set_weights(layer.get_weights())
    return None


def get_input_shape_layer_normalization(layer):
    axis = layer.axis
    input_shape = [None for _ in range(max(axis) + 1)]
    for i, ax in enumerate(axis):
        input_shape[ax] = layer.weights[0].shape[i]
    return input_shape


def copy_weights_layer_normalization(layer, new_layer):
    assert len(layer.axis) == 1, (
        f"Current copy weights implementation only works when normalising along a single "
        f"axis, but provided `layer.axis={layer.axis}`."
    )
    input_shape = get_input_shape_layer_normalization(layer)
    new_layer.build(input_shape)
    new_layer.set_weights(layer.get_weights())
    return None


def default_copy_weights_func(layer, new_layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return copy_weights_dense(layer, new_layer)
    if isinstance(layer, tf.keras.layers.LayerNormalization):
        return copy_weights_layer_normalization(layer, new_layer)
    raise ValueError(f"Not implemented method for layers of type {type(layer)}.")


class ModelReplacing(ModelLayerModification):
    def __init__(self, to_replace_dict=None):
        self.to_replace_dict = to_replace_dict
        super().__init__()

    @staticmethod
    def has_valid_get_config_method(layer):
        all_args = tf_inspect.getfullargspec(layer.__init__).args
        expected_args = ("trainable", "name", "dtype", "dynamic", "_batch_input_shape", "self")
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either there are no new arguments or that `get_config` has been overridden:
        if len(extra_args) > 0 and hasattr(layer.get_config, "_is_default"):
            return False
        return True

    @staticmethod
    def replace_trackable_layer(parent, layer, new_layer):
        logging.debug(f"Ensuring new layer {new_layer}" f"is trackable from parent {parent}.")
        # Ensure parent's attributes associated with tracking are
        # initialized. For example, if the model hasn't yet been
        # compiled.
        parent._maybe_initialize_trackable()
        # Remove the layer being replaced from the checkpoint dependencies
        # of the parent
        parent._self_unconditional_checkpoint_dependencies[:] = [
            t_ref for t_ref in parent._self_unconditional_checkpoint_dependencies[:] if t_ref.ref is not layer
        ]
        # Turn off _self_setattr_tracking when removing the layer from the
        # _self_unconditional_dependency_names to prevent tracking itself
        parent._self_setattr_tracking = False
        # Remove the layer being replaced from the parent's dependency
        # names to ensure it is not expected when doing dependency
        # search in the model
        parent._self_unconditional_dependency_names = {
            k: v for k, v in parent._self_unconditional_dependency_names.items() if v is not layer
        }
        # Turn on parent _self_setattr_tracking as it was
        parent._self_setattr_tracking = True
        # Add the new layer as a trackable object from the parent
        parent._track_trackable(new_layer, new_layer.name, overwrite=True)

    def get_layer_params(self, layer):
        if not self.has_valid_get_config_method(layer):
            args = tf_inspect.getcallargs(layer.__init__)
            args.pop("self")
            params = dict()
            for key in args.keys():
                if not hasattr(layer, key):
                    params[key] = getattr(layer, key)
                else:
                    default_arg = args[key]
                    params[key] = default_arg
                    logger.debug(f"Replacing layer {layer.name} with default value `{key}={default_arg}`")
            return params
        return layer.get_config()

    def get_new_layer_params(self, layer, to_replace_dict):
        params = self.get_layer_params(layer)
        if "new_params" in to_replace_dict[type(layer)]:
            new_layer_params = to_replace_dict[type(layer)]["new_params"]

            # Resolve any callable values
            for key, val in new_layer_params.items():
                if callable(val):
                    new_layer_params[key] = val()
        else:
            new_layer_params = dict()

        new_layer_class = to_replace_dict[type(layer)]["new_class"]
        valid_args = inspect.getfullargspec(new_layer_class.__init__)
        if valid_args.varkw:
            params.update(new_layer_params)
            return params

        for key, val in params.items():
            if key in valid_args.args and key not in new_layer_params:
                new_layer_params[key] = val
            else:
                logger.debug(f"Arg `{key}` from {type(layer)} is not accepted by {new_layer_class}.")
        return new_layer_params

    def process_layer(self, layer, layer_args, layer_kwargs, **kwargs):
        return self.search_layers(
            layer,
            layer_args,
            layer_kwargs,
            self.to_replace_dict,
            self.replace_layer,
            **kwargs,
        )

    def replace_layer(self, layer, to_replace_dict, parent, *args, **kwargs):
        to_replace_layer = to_replace_dict[type(layer)]
        new_layer_class = to_replace_layer["new_class"]

        params = self.get_new_layer_params(layer, to_replace_dict)
        logger.debug(f"Replacing layer {layer} with {new_layer_class}" f" and using additional params {params.keys()}")
        new_layer = new_layer_class(**params)

        if to_replace_layer.get("copy_weights", False):
            copy_weights_func = to_replace_layer.get("copy_weights_func", default_copy_weights_func)
            copy_weights_func(layer, new_layer)

        # Manually make new layer trackable from parent layer / model
        self.replace_trackable_layer(parent, layer, new_layer)

        return new_layer


class ModelOutlining(ModelLayerModification):
    def __init__(self, to_outline_dict):
        self.to_outline_dict = to_outline_dict
        super().__init__()

    @staticmethod
    def outline_layer_inplace(layer, to_outline_dict, *args, **kwargs):
        outline_kwargs = to_outline_dict[type(layer)].get("outline_kwargs", {})
        logger.debug(f"Replacing call with outlined call in layer" f" {layer} using outline kwargs: {outline_kwargs}")

        layer_call = layer.call

        def call(*args, **kwargs):
            @ipu.outlined_function(**outline_kwargs)
            def inner_call():
                return layer_call(*args, **kwargs)

            return inner_call()

        layer.call = call
        return layer

    def process_layer(self, layer, layer_args, layer_kwargs, **kwargs):
        return self.search_layers(
            layer,
            layer_args,
            layer_kwargs,
            self.to_outline_dict,
            self.outline_layer_inplace,
        )


class ModelAddRecomputationCheckpoints(ModelLayerModification):
    def __init__(self, to_checkpoint):
        self.to_checkpoint = to_checkpoint
        super().__init__()

    @staticmethod
    def add_recomputation_checkpoint(layer, *args, **kwargs):
        logger.debug(f"Adding recomputation checkpoint after layer {layer}")

        layer_call = layer.call

        def call(*args, **kwargs):
            x = layer_call(*args, **kwargs)
            x = ipu.pipelining_ops.recomputation_checkpoint(x)
            return x

        layer.call = call
        return layer

    def process_layer(self, layer, layer_args, layer_kwargs, **kwargs):
        return self.search_layers(
            layer,
            layer_args,
            layer_kwargs,
            self.to_checkpoint,
            self.add_recomputation_checkpoint,
        )
