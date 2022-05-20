# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import re
import torch
import logging
from functools import partial
from typing import Callable, List, Tuple


class ModelManipulator:
    """Manipulates an existing model at op level."""
    def __init__(self, model: torch.nn.Module):
        traced_graph = torch.fx.Tracer(autowrap_functions=getattr(self.__class__, "untractable_functions", [])).trace(model)
        self.traced_model = torch.fx.GraphModule(model, traced_graph)

    @classmethod
    def add_untracable_function(cls, function: Callable) -> None:
        """Add untracable functions in the models, which must be ignored by the tracer."""
        if not hasattr(cls, "untractable_functions"):
            cls.untractable_functions = []
        cls.untractable_functions.append(function)

    def transform(self, match_fn: Callable[[torch.fx.Node], bool], transform_fn: Callable[[torch.fx.Node], None],
                  transform_name: str="") -> torch.fx.GraphModule:
        """
        Apply transformation on the model
        match_fn: the function which return true if a given node requires transformation. Signiture of the method is fn(node:Node)
        transform_fn: the transform function. The signature of the method is fn(node:Node) The input is the selected node.
        transform_name: The name of the transform going to be logged to help the debugging.
        """
        for node in self.traced_model.graph.nodes:
            if match_fn(node):
                node_name = get_node_name(node)
                logging.info(f"{transform_name} applied on {node_name}")
                transform_fn(node)
                self.traced_model.recompile()
        return self.traced_model

    def transform_pipeline(self, transform_list: List[Tuple[Callable[[torch.fx.Node], Callable[[torch.fx.Node], None]]]]) -> torch.fx.GraphModule:
        """ Apply multiple transformation"""
        for match, trans, name in transform_list:
            self.transform(match, trans, name)
        return self.traced_model

    @classmethod
    def first_match(cls, match_fn: Callable[[torch.fx.Node], bool]) -> Callable[[torch.fx.Node], bool]:
        """Return true only for the first match"""
        cls.match_nr = 0

        def fn(node):
            if match_fn(node):
                cls.match_nr += 1
                return cls.match_nr == 1
        return fn


def next_node(match_fn: Callable[[torch.fx.Node], bool]) -> Callable[[torch.fx.Node], bool]:
    """Check whether any of the following node matches with the given function"""
    def fn(node):
        for next_n in node.users.keys():
            if match_fn(next_n):
                return True
        return False
    return fn


def name_match(name_patterns: List[str]) -> Callable[[torch.fx.Node], bool]:
    """Checks whether any of the provided regex matches with the name of the node"""
    if isinstance(name_patterns, str):
        name_patterns = [name_patterns]

    def fn(node):
        name = get_node_name(node)
        for name_pattern in name_patterns:
            if re.fullmatch(name_pattern, name):
                return True
        return False
    return fn


def type_match(types: List) -> Callable[[torch.fx.Node], bool]:
    """Checks whether any of the provided type matches with the node's type."""
    if not isinstance(types, list):
        types = [types]

    def fn(node):
        node_module = get_module_from_node(node)
        for t in types:
            if isinstance(node_module, t):
                return True
        return False
    return fn


def replace_op(node2node_func: Callable[[torch.fx.Node], torch.fx.Node]) -> Callable[[torch.fx.Node], None]:
    """Replace the node to a new one. The new node defined by a function, which maps the old Node to a new Node"""
    def fn(node):
        traced_graph = node.graph
        new_op = node2node_func(node)
        with traced_graph.inserting_after(node):
            new_node = traced_graph.call_function(new_op, args=tuple(node.all_input_nodes))
            node.replace_all_uses_with(new_node)
            node.graph.erase_node(node)
    return fn


def insert_after(node2node_func: Callable[[torch.fx.Node], torch.fx.Node]) -> Callable[[torch.fx.Node], None]:
    """Insert a new node after the current one. The new node defined by a function, which uses the current Node."""
    def fn(node):
        traced_graph = node.graph
        new_op = node2node_func(node)
        with traced_graph.inserting_after(node):
            new_node = traced_graph.call_function(new_op, args=(node,))
            node.replace_all_uses_with(new_node)
            new_node.args = (node,)
    return fn


def replace_module(node2node_func: Callable[[torch.fx.Node], torch.fx.Node]) -> Callable[[torch.fx.Node], None]:
    """Replaces the module, which belongs to the current node.
        The new node defined by a function, which uses the current Node.
    """
    def fn(node):
        new_op = node2node_func(node)
        sub_module_names = node.target.split('.')
        module = node.graph.owning_module
        for name in sub_module_names[:-1]:
            module = module.get_submodule(name)
        setattr(module, sub_module_names[-1], new_op)
    return fn


def get_module_from_node(node: torch.fx.node.Node) -> torch.nn.Module:
    """ Return the asigned module for the node. If it doesn't exist return None."""
    modules = dict(node.graph.owning_module.named_modules())
    return modules.get(node.target, None)


def get_node_name(node: torch.fx.node.Node) -> str:
    """Return the name of the node"""
    if get_module_from_node(node) is None:
        name = str(node)
    else:
        name = str(node.target).replace('.', '/')
    return name


def get_norm_layer(args) -> torch.nn.Module:
    """Provides the required norm layer"""
    if args.norm_type == "none":
        return torch.nn.Identity
    elif args.norm_type == "batch":
        return partial(torch.nn.BatchNorm2d, momentum=args.batchnorm_momentum, eps=args.norm_eps)
    elif args.norm_type == "group":
        return partial(torch.nn.GroupNorm, args.norm_num_groups, eps=args.norm_eps)
