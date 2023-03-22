# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from typing import Dict, Mapping, Optional, Union
import argparse
from functools import partial

import numpy as np
import torch
import torchvision
from tqdm import tqdm

import popxl
import popxl_addons as addons
import popxl.ops as ops

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
                    # mean and std computed on the training set.
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
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


class Linear(addons.Module):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias

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

        return y


class Net(addons.Module):
    def __init__(self, cache: Optional[addons.GraphCache] = None):
        super().__init__(cache=cache)
        self.fc1 = Linear(512)
        self.fc2 = Linear(512)
        self.fc3 = Linear(512)
        self.fc4 = Linear(10)

    def build(self, x: popxl.Tensor):
        x = x.reshape((-1, 28 * 28))
        x = ops.gelu(self.fc1(x))
        x = ops.gelu(self.fc2(x))
        x = ops.gelu(self.fc3(x))
        x = self.fc4(x)
        return x


"""
Adam optimiser.
Defines Adam update step for a single variable
"""


class Adam(addons.Module):
    # We need to specify `in_sequence` because a lot of operations are in-place
    # and their order shouldn't be rearranged
    @popxl.in_sequence()
    def build(
        self,
        weight: popxl.TensorByRef,
        grad: popxl.Tensor,
        *,
        lr: Union[float, popxl.Tensor],
        beta1: Union[float, popxl.Tensor] = 0.9,
        beta2: Union[float, popxl.Tensor] = 0.999,
        eps: Union[float, popxl.Tensor] = 1e-5,
        weight_decay: Union[float, popxl.Tensor] = 0.0,
        first_order_dtype: popxl.dtype = popxl.float16,
        bias_correction: bool = True,
    ):
        # Gradient estimator for the variable `weight` - same shape as the variable
        first_order = self.add_variable_input(
            "first_order",
            partial(np.zeros, weight.shape),
            first_order_dtype,
            by_ref=True,
        )
        ops.var_updates.accumulate_moving_average_(first_order, grad, f=beta1)

        # Variance estimator for the variable `weight` - same shape as the variable
        second_order = self.add_variable_input(
            "second_order", partial(np.zeros, weight.shape), popxl.float32, by_ref=True
        )
        ops.var_updates.accumulate_moving_average_square_(second_order, grad, f=beta2)

        # Adam is a biased estimator: provide the step to correct bias
        step = None
        if bias_correction:
            step = self.add_variable_input("step", partial(np.zeros, ()), popxl.float32, by_ref=True)

        # Calculate the weight increment with Adam heuristic
        # Here we use the built-in `adam_updater`, but you can write your own.
        dw = ops.var_updates.adam_updater(
            first_order,
            second_order,
            weight=weight,
            weight_decay=weight_decay,
            time_step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=eps,
        )

        # in-place weight update: weight += (-lr)*dw
        ops.scaled_add_(weight, dw, b=-lr)


"""
Update all variables creating per-variable optimisers.
"""


def optimiser_step(
    variables,
    grads: Dict[popxl.Tensor, popxl.Tensor],
    optimiser: addons.Module,
    lr: popxl.float32 = 1e-3,
):
    for name, var in variables.named_tensors.items():
        # Create optimiser and state factories for the variable
        opt_facts, opt_graph = optimiser.create_graph(var, var.spec, lr=lr, weight_decay=0.0, bias_correction=True)
        state = opt_facts.init()
        # bind the graph to its state and call it.
        # Both the state and the variables are updated in-place and are passed by ref,
        # hence after the graph is called they are updated.
        opt_graph.bind(state).call(var, grads[var])


def train(train_session, training_data, opts, input_streams, loss_stream):
    nr_batches = len(training_data)
    with train_session:
        for epoch in range(1, opts.epochs + 1):
            print(f"Epoch {epoch}/{opts.epochs}")
            bar = tqdm(training_data, total=nr_batches)
            for data, labels in bar:
                inputs: Mapping[popxl.HostToDeviceStream, np.ndarray] = dict(
                    zip(input_streams, [data.squeeze().float(), labels.int()])
                )
                loss = train_session.run(inputs)
                bar.set_description(f"Loss:{loss[loss_stream]:0.4f}")


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

    test_set_accuracy = sum_acc / len(test_data)
    print(f"Accuracy on test set: {test_set_accuracy:0.2f}%")


def train_program(opts):
    ir = popxl.Ir(replication=1)

    with ir.main_graph:
        # Create input streams from host to device
        img_stream = popxl.h2d_stream((opts.batch_size, 28, 28), popxl.float32, "image")
        img_t = ops.host_load(img_stream)  # load data
        label_stream = popxl.h2d_stream((opts.batch_size,), popxl.int32, "labels")
        labels = ops.host_load(label_stream, "labels")

        # Create forward graph
        facts, fwd_graph = Net().create_graph(img_t)

        # Create backward graph via autodiff transform
        bwd_graph = addons.autodiff(fwd_graph)

        # Initialise variables (weights)
        variables = facts.init()

        # Call the forward graph with call_with_info because we want to retrieve
        # information from the call site
        fwd_info = fwd_graph.bind(variables).call_with_info(img_t)
        x = fwd_info.outputs[0]  # forward output

        # Compute loss and starting gradient for backpropagation
        loss, dx = addons.ops.cross_entropy_with_grad(x, labels)

        # Setup a stream to retrieve loss values from the host
        loss_stream = popxl.d2h_stream(loss.shape, loss.dtype, "loss")
        ops.host_store(loss_stream, loss)

        # Retrieve activations from the forward graph
        activations = bwd_graph.grad_graph_info.inputs_dict(fwd_info)

        # Call the backward graph providing the starting value for
        # backpropagation and activations
        bwd_info = bwd_graph.call_with_info(dx, args=activations)

        # Adam optimiser, with cache
        grads_dict = bwd_graph.grad_graph_info.fwd_parent_ins_to_grad_parent_outs(fwd_info, bwd_info)
        optimiser = Adam(cache=True)
        optimiser_step(variables, grads_dict, optimiser, opts.lr)

    ir.num_host_transfers = 1
    return (
        popxl.Session(ir, "ipu_hw"),
        [img_stream, label_stream],
        variables,
        loss_stream,
    )


def test_program(opts):
    ir = popxl.Ir()
    ir.replication_factor = 1

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
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for training (default: 8)")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=80,
        help="batch size for testing (default: 80)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    opts = parser.parse_args()

    training_data, test_data = get_mnist_data(opts.test_batch_size, opts.batch_size)

    train_session, train_input_streams, train_variables, loss_stream = train_program(opts)

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

    test(test_session, test_data, test_input_streams, out_stream)


if __name__ == "__main__":
    main()
