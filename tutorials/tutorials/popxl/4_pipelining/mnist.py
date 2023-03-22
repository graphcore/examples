# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
from functools import partial
from telnetlib import FORWARD_X
from typing import Mapping, Optional
from typing_extensions import Required
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from time import time
from dataclasses import dataclass, field

import popxl
import popxl_addons as addons
import popxl.ops as ops
from typing import Union, Dict
from popxl_addons.graph import GraphWithNamedArgs, BoundGraph
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.variable_factory import NamedVariableFactories
from popxl.transforms import GradGraphInfo
import logging

np.random.seed(42)


def get_mnist_data(test_batch_size: int, batch_size: int):
    training_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # mean and std computed on the training set.
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "~/.torch/datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        drop_last=True,
    )
    return training_data, validation_data


def accuracy(predictions: np.ndarray, labels: np.ndarray):
    ind = np.argmax(predictions, axis=-1).flatten()
    labels = labels.detach().numpy().flatten()
    return np.mean(ind == labels) * 100.0


# includes gelu
class Linear(addons.Module):
    def __init__(self, out_features: int, bias: bool = True, gelu: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.gelu = gelu

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        # add a state variable to the module
        w = self.add_variable_input(
            "weight",
            partial(np.random.normal, 0, 0.02, (x.shape[-1], self.out_features)),
            x.dtype,
        )
        y = x @ w
        if self.bias:
            # add a state variable to the module
            b = self.add_variable_input("bias", partial(np.zeros, y.shape[-1]), x.dtype)
            y = y + b
        if self.gelu:
            y = ops.gelu(y)
        return y


# gelu included in the linear layer
class Net(addons.Module):
    def __init__(self, cache: Optional[addons.GraphCache] = None):
        super().__init__(cache=cache)
        self.fc1 = Linear(512)
        self.fc2 = Linear(512)
        self.fc3 = Linear(512)
        self.fc4 = Linear(10, gelu=False)

    def build(self, x: popxl.Tensor):
        x = x.reshape((-1, 28 * 28))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


"""
Adam optimizer.
Defines adam update step for a single variable
"""


class Adam(addons.Module):
    # we need to specify in_sequence because a lot of operations are in place and their order
    # shouldn't be rearranged
    @popxl.in_sequence()
    def build(
        self,
        var: popxl.TensorByRef,
        grad: popxl.Tensor,
        *,
        lr: Union[float, popxl.Tensor],
        beta1: Union[float, popxl.Tensor] = 0.9,
        beta2: Union[float, popxl.Tensor] = 0.999,
        eps: Union[float, popxl.Tensor] = 1e-5,
        weight_decay: Union[float, popxl.Tensor] = 1e-2,
        first_order_dtype: popxl.dtype = popxl.float16,
        bias_correction: bool = True,
    ):

        # gradient estimators for the variable var - same shape as the variable
        first_order = self.add_variable_input(
            "first_order", partial(np.zeros, var.shape), first_order_dtype, by_ref=True
        )
        ops.var_updates.accumulate_moving_average_(first_order, grad, f=beta1)

        # variance estimators for the variable var - same shape as the variable
        second_order = self.add_variable_input("second_order", partial(np.zeros, var.shape), popxl.float32, by_ref=True)
        ops.var_updates.accumulate_moving_average_square_(second_order, grad, f=beta2)

        # adam is a biased estimator: provide the step to correct bias
        step = None
        if bias_correction:
            step = self.add_variable_input("step", partial(np.zeros, ()), popxl.float32, by_ref=True)

        # calculate the weight increment with adam heuristic
        updater = ops.var_updates.adam_updater(
            first_order,
            second_order,
            weight=var,
            weight_decay=weight_decay,
            time_step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=eps,
        )

        # in place weight update: w += (-lr)*dw
        ops.scaled_add_(var, updater, b=-lr)


"""
Groups together the forward and backward graphs of a layer for easy access and handling.
"""


class Graphs:
    def __init__(
        self,
        layer_name: str,
        fwd: GraphWithNamedArgs,
        bwd: GraphWithNamedArgs,
        facts: NamedVariableFactories,
        optimizer: addons.Module,
    ):
        self.layer_name = layer_name
        self.fwd = fwd
        self.bwd = bwd
        self.facts = facts
        self.optimizer = optimizer
        self.vars = NamedTensors()

    def init_and_bind_fwd(self):
        self.vars.insert("fwd", self.facts.fwd.init(self.layer_name))
        return self.fwd.bind(self.vars.fwd)

    def init_and_bind_bwd(self):
        self.vars.insert("bwd", self.facts.bwd.init(self.layer_name))
        return self.bwd.bind(self.vars.bwd)

    def recompute_graph(self):
        self.bwd = addons.transforms.recompute_graph(self.bwd)

    def replicated_all_reduce(self):
        for g in self.vars.bwd.tensors[:-1]:
            g = ops.collectives.replicated_all_reduce_(g, op="mean")

    def optimizer_step(self, lr: Union[float, popxl.Tensor]):
        var_dict = self.vars.fwd.named_tensors
        grad_dict = self.vars.bwd.accum.to_dict()
        for name, var in var_dict.items():
            opt_facts, opt = self.optimizer.create_graph(var, var.spec, lr=lr, weight_decay=0.0, bias_correction=True)
            state = opt_facts.init()
            opt.bind(state).call(var, grad_dict[name])

        ops.var_updates.accumulator_scale_(self.vars.bwd.mean_accum_counter, 0.0)

    def reset_vars(self):
        self.vars._clear()


def create_graphs(
    layer_name: str,
    layer: addons.Module,
    optimizer: addons.module,
    opts,
    require_input0: bool,
    *args,
    **kwargs,
):
    facts, graph = layer.create_graph(*args)
    # tensors_to_accumulate_grads = graph.args.tensors : accumulate gradients of the weights
    # grads_required = (graph.graph.inputs[0],): we need to return the gradient of the first input of the layer, since it
    # will be starting value for backpropagation in the other layers
    req_grads = (graph.graph.inputs[0],) if require_input0 else ()
    bwd_facts, bwd_graph = addons.transforms.autodiff_with_accumulation(
        graph, tensors_to_accumulate_grads=graph.args.tensors, grads_required=req_grads
    )
    factories = NamedVariableFactories()
    factories.insert("fwd", facts)
    factories.insert("bwd", bwd_facts)

    return Graphs(layer_name, graph, bwd_graph, factories, optimizer)


def train(train_session, training_data, opts, input_streams, loss_stream):
    nr_batches = len(training_data)
    with train_session:
        for epoch in range(1, opts.epochs + 1):
            print(f"Epoch {epoch}/{opts.epochs}")
            bar = tqdm(training_data, total=nr_batches)
            for data, labels in bar:
                # reshape data accounting for replication and num hosts transfers
                data = data.reshape(
                    opts.gradient_accumulation,
                    opts.data_parallel,
                    opts.train_micro_batch_size,
                    28 * 28,
                ).squeeze()
                labels = labels.reshape(
                    opts.gradient_accumulation,
                    opts.data_parallel,
                    opts.train_micro_batch_size,
                ).squeeze()

                inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                    zip(input_streams, [data.squeeze().float(), labels.int()])
                )
                loss = train_session.run(inputs)
                losses_np = loss[loss_stream]  # shape(ir.num_host_transfers, ir.replication_factor, )
                avg_loss = np.mean(losses_np)
                bar.set_description(f"Loss:{avg_loss:0.4f}")


def evaluate_throughput(session, samples_per_step, epochs: int = 5):
    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    durations = []
    with session:
        for i in range(epochs):
            start = time()
            session.run(inputs)
            dur = time() - start
            durations.append(dur)

    duration = np.mean(durations)

    result_str = f"Mean duration: {duration} s " f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    print(result_str)


def test(test_session, test_data, input_streams, out_stream):
    nr_batches = len(test_data)
    sum_acc = 0.0
    with test_session:
        for data, labels in tqdm(test_data, total=nr_batches):
            inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                zip(input_streams, [data.squeeze().float(), labels.int()])
            )
            output = test_session.run(inputs)
            sum_acc += accuracy(output[out_stream], labels)
    print(f"Accuracy on test set: {sum_acc / len(test_data):0.2f}%")


def train_program(opts):
    ir = popxl.Ir()
    ir.replication_factor = opts.data_parallel

    with ir.main_graph:
        # -----  Define input and output streams -----
        img_spec = popxl.TensorSpec((opts.train_micro_batch_size, 28 * 28), popxl.float32)
        inner_spec = popxl.TensorSpec((opts.train_micro_batch_size, 512), popxl.float32)

        img_stream = popxl.h2d_stream(img_spec.shape, popxl.float32, "image")
        label_stream = popxl.h2d_stream((opts.train_micro_batch_size,), popxl.int32, "labels")
        loss_stream = popxl.d2h_stream((), popxl.float32, "loss")

        optimizer = Adam(cache=True)
        steps = opts.gradient_accumulation

        # ----- Create graphs -----

        # create graphs in the appropriate ipu context
        with popxl.ipu(0):
            fc1 = create_graphs("fc1", Linear(512), optimizer, opts, False, img_spec)
            fc2 = create_graphs("fc2", Linear(512), optimizer, opts, True, inner_spec)
        with popxl.ipu(1):
            fc3 = create_graphs("fc3", Linear(512), optimizer, opts, True, inner_spec)
            fc4 = create_graphs("fc4", Linear(10, gelu=False), optimizer, opts, True, inner_spec)

        # ----- Transform graphs -----
        # for example, add recomputation
        if opts.recomputation:
            fc1.recompute_graph()
            fc2.recompute_graph()
            fc3.recompute_graph()
            fc4.recompute_graph()

        # ----- Construct Execution Scheme -----
        #
        #  Pipeline
        #   stage0, ipu0: fc1 forward, fc2 forward
        #   stage1, ipu1: fc3 forward, fc4 forward, loss, fc4 backward, fc3 backward
        #   stage2, ipu0: fc2 backward, fc1 backward
        #
        #  Replica Reduction
        #  Optimizer
        #
        # --------------------------------------

        # ----- Pipeline -----
        with popxl.in_sequence(True):
            # context for pipeline: when the context closes the pipeline transformation is applied  to the graph
            with addons.pipelined_execution(steps) as pipeline:

                with pipeline.stage(0), popxl.ipu(0):
                    # fc1 fc2 forward
                    img_t = ops.host_load(img_stream)
                    x: popxl.Tensor
                    fc1_info = fc1.init_and_bind_fwd().call_with_info(img_t)
                    x = fc1_info.outputs[0]
                    fc2_info = fc2.init_and_bind_fwd().call_with_info(x)
                    x = fc2_info.outputs[0]
                    x = x.copy_to_ipu(1)

                with pipeline.stage(1), popxl.ipu(1):
                    # fc3 fc4 forward
                    labels = ops.host_load(label_stream, "labels")
                    fc3_info = fc3.init_and_bind_fwd().call_with_info(x)
                    x = fc3_info.outputs[0]
                    fc4_info = fc4.init_and_bind_fwd().call_with_info(x)
                    x = fc4_info.outputs[0]

                    # loss
                    loss, dx = addons.ops.cross_entropy_with_grad(x, labels)
                    ops.host_store(loss_stream, loss)

                    # grads
                    fc3_activations = fc3.bwd.grad_graph_info.inputs_dict(fc3_info)
                    fc4_activations = fc4.bwd.grad_graph_info.inputs_dict(fc4_info)
                    (dx,) = fc4.init_and_bind_bwd().call(dx, args=fc4_activations)  # provide fc4 activations
                    (dx,) = fc3.init_and_bind_bwd().call(dx, args=fc3_activations)  # provide fc3 activations

                    dx = dx.copy_to_ipu(0)

                with pipeline.stage(2), popxl.ipu(0):
                    # using stash_and_restore_activations ensure that when the pipeline graph is created
                    # activations are stashed during forward and retrieved from the FIFO stash during backward.
                    # Needs to be called inside a stage

                    # grads
                    fc2_activ = pipeline.stash_and_restore_activations(fc2_info, fc2.bwd.grad_graph_info)
                    (dx,) = fc2.init_and_bind_bwd().call(dx, args=fc2_activ)

                    fc1_activ = pipeline.stash_and_restore_activations(fc1_info, fc1.bwd.grad_graph_info)
                    fc1.init_and_bind_bwd().call(dx, args=fc1_activ)

            # -----/ Pipeline -----

            # ----- Replica Reduction -----
            if opts.data_parallel > 1:
                with popxl.ipu(0):
                    fc1.replicated_all_reduce()
                    fc2.replicated_all_reduce()

                with popxl.ipu(1):
                    fc3.replicated_all_reduce()
                    fc4.replicated_all_reduce()

            # ----- Optimizer -----
            with popxl.ipu(0):
                fc1.optimizer_step(opts.lr)
                fc2.optimizer_step(opts.lr)

            with popxl.ipu(1):
                fc3.optimizer_step(opts.lr)
                fc4.optimizer_step(opts.lr)

    # we have a for loop, the number of host loads is equal to gradient_accumulation
    ir.num_host_transfers = opts.gradient_accumulation

    # group all the variables to be able to copy weights to the test session
    vars = NamedTensors()
    vars.insert("fc1", fc1.vars.fwd)
    vars.insert("fc2", fc2.vars.fwd)
    vars.insert("fc3", fc3.vars.fwd)
    vars.insert("fc4", fc4.vars.fwd)

    return popxl.Session(ir, "ipu_hw"), [img_stream, label_stream], vars, loss_stream


def test_program(opts):
    ir = popxl.Ir(replication=1)

    with ir.main_graph:
        # Inputs
        in_stream = popxl.h2d_stream((opts.test_batch_size, 28, 28), popxl.float32, "image")
        in_t = ops.host_load(in_stream)

        # Create graphs
        facts, graph = Net().create_graph(in_t)

        # Initialise variables
        variables = facts.init()

        # Forward
        (outputs,) = graph.bind(variables).call(in_t)
        out_stream = popxl.d2h_stream(outputs.shape, outputs.dtype, "outputs")
        ops.host_store(out_stream, outputs)

    ir.num_host_transfers = 1
    return popxl.Session(ir, "ipu_hw"), [in_stream], variables, out_stream


def main():
    parser = argparse.ArgumentParser(description="MNIST training in popxl.addons")
    parser.add_argument(
        "--train-micro-batch-size",
        type=int,
        default=5,
        help="batch size for training (default: 3)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=80,
        help="batch size for testing (default: 80)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--data-parallel", type=int, default=1, help="data parallelism (default: 1)")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=6,
        help="gradient accumulation (default: 6)",
    )
    parser.add_argument(
        "--recomputation",
        type=bool,
        default=True,
        help="use recomputation for activations",
    )

    opts = parser.parse_args()

    train_global_batch_size = opts.train_micro_batch_size * opts.gradient_accumulation * opts.data_parallel

    training_data, test_data = get_mnist_data(opts.test_batch_size, train_global_batch_size)

    train_session, train_input_streams, train_variables, loss_stream = train_program(opts)

    print("train session")
    train(train_session, training_data, opts, train_input_streams, loss_stream)
    # get weights data : dictionary { train_session variables : tensor data (numpy) }
    train_vars_to_data = train_session.get_tensors_data(train_variables.tensors)

    # create test session
    test_session, test_input_streams, test_variables, out_stream = test_program(opts)

    # dictionary { train_session variables : test_session variables }
    train_vars_to_test_vars = train_variables.to_mapping(test_variables)
    # Create a dictionary { test_session variables : tensor data (numpy) }
    test_vars_to_data = {
        test_var: train_vars_to_data[train_var].copy() for train_var, test_var in train_vars_to_test_vars.items()
    }
    # Copy trained weights to the program, with a single host to device transfer at the end
    test_session.write_variables_data(test_vars_to_data)

    # throughput for training
    samples_per_step = opts.train_micro_batch_size * opts.gradient_accumulation * opts.data_parallel
    evaluate_throughput(train_session, samples_per_step)

    # run inference on validation dataset
    print("test session")
    test(test_session, test_data, test_input_streams, out_stream)
    # throughput for inference
    samples_per_step = opts.test_batch_size
    evaluate_throughput(test_session, samples_per_step)


if __name__ == "__main__":
    main()
