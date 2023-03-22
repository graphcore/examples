# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
from functools import partial
from typing import Mapping, Optional
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from time import time

import popxl
import popxl_addons as addons
import popxl.ops as ops
from typing import Union, Dict
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.variable_factory import NamedVariableFactories

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
Update all variables creating per-variable optimizers.
"""


def optimizer_step(
    variables: NamedTensors,
    grads: Dict[popxl.Tensor, popxl.Tensor],
    optimizer: addons.Module,
    accum_counter: popxl.Tensor,
    lr: popxl.float32 = 1e-3,
):
    for name, var in variables.named_tensors.items():
        # create optimizer and state factories for the variable
        opt_facts, opt_graph = optimizer.create_graph(var, var.spec, lr=lr, weight_decay=0.0, bias_correction=True)
        state = opt_facts.init()
        # bind the graph to its state and call it.
        opt_graph.bind(state).call(var, grads[var])

    if accum_counter is not None:
        # Reset accumulators.
        # Resetting the counter for mean gradient accumulation is sufficient to zero the accumulators
        # in the next call to ops.accumulate_mean_
        ops.var_updates.accumulator_scale_(accum_counter, 0.0)


def train(train_session, training_data, opts, input_streams, loss_stream):
    nr_batches = len(training_data)
    with train_session:
        for epoch in range(1, opts.epochs + 1):
            print(f"Epoch {epoch}/{opts.epochs}")
            bar = tqdm(training_data, total=nr_batches)
            for data, labels in bar:
                # reshape data accounting for replication and num hosts transfers
                data = data.reshape(
                    train_session.ir.num_host_transfers,
                    train_session.ir.replication_factor,
                    opts.train_micro_batch_size,
                    28,
                    28,
                ).squeeze()
                labels = labels.reshape(
                    train_session.ir.num_host_transfers,
                    train_session.ir.replication_factor,
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
    ir = popxl.Ir(replication=opts.data_parallel)
    # total number of replicas used in the program, regardeless their use
    # here, we are using them to implement data parallelism, and no other
    # use of replication is involved.

    with ir.main_graph:
        # Create input streams from host to device
        img_spec = popxl.TensorSpec((opts.train_micro_batch_size, 28, 28), popxl.float32)

        img_stream = popxl.h2d_stream(img_spec.shape, popxl.float32, "image")
        label_stream = popxl.h2d_stream((opts.train_micro_batch_size,), popxl.int32, "labels")
        loss_stream = popxl.d2h_stream((), popxl.float32, "loss")

        # Create forward graph
        facts, fwd_graph = Net().create_graph(img_spec)
        variables = facts.init()
        bound_fwd = fwd_graph.bind(variables)

        counter = None
        required_grads = fwd_graph.args.tensors

        if opts.gradient_accumulation > 1:
            bwd_facts, bwd_graph = addons.autodiff_with_accumulation(fwd_graph, required_grads)
            accumulated_grads = bwd_facts.init()
            counter = accumulated_grads.mean_accum_counter
            bound_bwd = bwd_graph.bind(accumulated_grads)
        else:
            # standard autodiff, avoid extra memory
            bwd_graph = addons.autodiff(fwd_graph, grads_required=required_grads)

        # in sequence needed for in place ops
        with popxl.in_sequence(True):
            # gradient accumulation loop
            for ga_step in range(opts.gradient_accumulation):
                # load data
                img_t = ops.host_load(img_stream)
                labels = ops.host_load(label_stream, "labels")

                # fwd
                fwd_info = bound_fwd.call_with_info(img_t)
                x = fwd_info.outputs[0]

                # loss
                loss, dx = addons.ops.cross_entropy_with_grad(x, labels)
                ops.host_store(loss_stream, loss)

                # bwd
                activations = bwd_graph.grad_graph_info.inputs_dict(fwd_info)

                if opts.gradient_accumulation > 1:
                    bound_bwd.call(dx, args=activations)
                    grads = accumulated_grads.tensors[:-1]  # exclude the counter

                else:
                    grads = bwd_graph.call(dx, args=activations)

            # reduce gradients across replicas with mean op
            if opts.data_parallel > 1:
                for g in grads:
                    g = ops.collectives.replicated_all_reduce_(g, op="mean")

            # optimizer step: the optimizer resets the accumulators
            grads_dict = dict(zip(variables.tensors, grads))
            optimizer = Adam(cache=True)
            optimizer_step(variables, grads_dict, optimizer, counter, opts.lr)

    # we have a for loop, the number of host loads is equal to gradient_accumulation
    ir.num_host_transfers = opts.gradient_accumulation

    return (
        popxl.Session(ir, "ipu_hw"),
        [img_stream, label_stream],
        variables,
        loss_stream,
    )


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
        default=8,
        help="batch size for training (default: 8)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=80,
        help="batch size for testing (default: 80)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--data-parallel", type=int, default=2, help="data parallelism (default: 2)")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="gradient accumulation (default: 4)",
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

    print("test session")
    # run inference on validation dataset
    samples_per_step = opts.test_batch_size  # no replication for inference in this program
    test(test_session, test_data, test_input_streams, out_stream)
    # throughput for inference
    evaluate_throughput(test_session, samples_per_step)


if __name__ == "__main__":
    main()
