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
from popxl import ReplicaGrouping
import popxl_addons as addons
import popxl.ops as ops
from typing import Union
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.named_replica_grouping import NamedReplicaGrouping
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons import (
    batch_serialise,
    batch_serialise_fwd_and_grad,
    batch_serial_buffer,
)
from popxl_addons.rts import (
    all_gather_replica_sharded_graph,
    replica_sharded_spec,
)
from popxl_addons.rts import reduce_replica_sharded_graph

from popxl_addons.remote import (
    NamedRemoteBuffers,
    named_variable_buffers,
    load_remote_graph,
    store_remote_graph,
)

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
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        gelu: bool = True,
        replica_grouping: Optional[ReplicaGrouping] = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self.gelu = gelu
        self.rg = replica_grouping

    def build(self, x: popxl.Tensor) -> popxl.Tensor:
        # add a state variable to the module
        w = self.add_variable_input(
            "weight",
            partial(np.random.normal, 0, 0.02, (x.shape[-1], self.out_features)),
            x.dtype,
            replica_grouping=self.rg,
        )
        y = x @ w
        if self.bias:
            # add a state variable to the module
            b = self.add_variable_input(
                "bias",
                partial(np.zeros, y.shape[-1]),
                x.dtype,
                replica_grouping=self.rg,
            )
            y = y + b
        if self.gelu:
            y = ops.gelu(y)
        return y


class OutputLayerWithBwd(addons.Module):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        gelu: bool = True,
        replica_grouping: Optional[ReplicaGrouping] = None,
    ):
        super().__init__()
        self.linear = Linear(
            out_features=out_features,
            bias=bias,
            gelu=gelu,
            replica_grouping=replica_grouping,
        )

    def build(self, x: popxl.Tensor, labels=popxl.Tensor) -> popxl.Tensor:

        fwd_facts, fwd_graph = self.linear.create_graph(x.spec)
        bwd_facts, bwd_graph = addons.transforms.autodiff_with_accumulation(
            fwd_graph,
            tensors_to_accumulate_grads=fwd_graph.args.tensors,
            grads_required=[fwd_graph.graph.inputs[0]],
            replica_groupings=fwd_facts.replica_groupings,
        )

        # outline forward
        vars = self.add_variable_inputs("fwd", fwd_facts)
        fwd_info = fwd_graph.bind(vars).call_with_info(x)
        x = fwd_info.parent_output(0)

        loss, dx = addons.ops.cross_entropy_with_grad(x, labels)

        # outline backward
        bwd_vars = self.add_variable_inputs("bwd", bwd_facts)
        (dx,) = bwd_graph.bind(bwd_vars).call(dx, args=bwd_graph.grad_graph_info.inputs_dict(fwd_info))

        return dx, loss


# gelu included in the linear layer
class Net(addons.Module):
    def __init__(self, rg: ReplicaGrouping, cache: Optional[addons.GraphCache] = None):
        super().__init__(cache=cache)
        self.fc1 = Linear(512, rg)
        self.fc2 = Linear(512, rg)
        self.fc3 = Linear(512, rg)
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
        replica_grouping: Optional[popxl.ReplicaGrouping] = None,
        *,
        lr: Union[float, popxl.Tensor],
        beta1: Union[float, popxl.Tensor] = 0.9,
        beta2: Union[float, popxl.Tensor] = 0.999,
        eps: Union[float, popxl.Tensor] = 1e-5,
        weight_decay: Union[float, popxl.Tensor] = 0.0,
        first_order_dtype: popxl.dtype = popxl.float16,
        bias_correction: bool = True,
    ):

        # gradient estimators for the variable var - same shape as the variable

        # Sharded inputs must be added with add_replica_sharded_variable_input
        if var.meta_shape:
            # shard over factor can be automatically computed from the variable
            shard_over = np.prod(var.meta_shape) // np.prod(var.shape)
            first_order = self.add_replica_sharded_variable_input(
                "first_order",
                partial(np.zeros, var.meta_shape),
                first_order_dtype,
                replica_grouping=replica_grouping,
                shard_over=shard_over,
                by_ref=True,
            )
            second_order = self.add_replica_sharded_variable_input(
                "second_order",
                partial(np.zeros, var.meta_shape),
                popxl.float32,
                replica_grouping=replica_grouping,
                shard_over=shard_over,
                by_ref=True,
            )

        else:
            first_order = self.add_variable_input(
                "first_order",
                partial(np.zeros, var.shape),
                first_order_dtype,
                by_ref=True,
                replica_grouping=replica_grouping,
            )
            second_order = self.add_variable_input(
                "second_order",
                partial(np.zeros, var.shape),
                popxl.float32,
                by_ref=True,
                replica_grouping=replica_grouping,
            )

        ops.var_updates.accumulate_moving_average_(first_order, grad, f=beta1)
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
Build the replica groupings to be used for replicated tensor sharding.
If the tensor has less elements than the threshold, the group_size will be 1
so that no sharding happens. Otherwise, tensors will be sharded across the data
parallel replicas.
"""


def get_shard_groups(opts, facts: NamedVariableFactories) -> NamedReplicaGrouping:
    ir = popxl.gcg().ir

    rts_groups = {}
    for k, f in facts.to_dict().items():
        size = np.prod(f.shape)
        rg = f.replica_grouping
        if size >= opts.sharded_threshold and size % rg.group_size == 0:
            rts_groups[k] = rg
        else:
            rts_groups[k] = ir.replica_grouping(group_size=1)
    # it is important to sort the tensor names.
    return dict(sorted(rts_groups.items()))


"""
Groups together the forward, backward and optimizers graphs of a layer for easy access and handling.
"""


class Graphs:
    def __init__(
        self,
        opts,
        layer: addons.Module,
        optimizer: addons.Module,
        entries: int,
        require_dx_0: bool,
        rg: ReplicaGrouping,
        *args,
        **kwargs,
    ):
        self.rg = rg
        # Create Graphs for computing forward, gradient and optimizer
        fwd_facts, self.fwd = layer.create_graph(*args, **kwargs)
        # variables and accumulators will be sharded according to the selected threshold
        self.shard_groups = get_shard_groups(opts, fwd_facts)

        required_grads = (self.fwd.graph.inputs[0],) if require_dx_0 else ()
        grad_facts, self.bwd = addons.autodiff_with_accumulation(
            self.fwd,
            tensors_to_accumulate_grads=self.fwd.args.tensors,
            grads_required=required_grads,
        )

        optim_facts = self._setup_optim(optimizer, self.fwd.args, opts)
        self._set_factories(fwd_facts, optim_facts, grad_facts)
        self._setup_graphs(opts, entries)

    @classmethod
    def empty(cls):
        return super().__new__(cls)

    def from_fwd_and_bwd(
        opts,
        fwd_and_bwd: addons.Module,
        optimizer: addons.Module,
        entries: int,
        rg: ReplicaGrouping,
        *args,
        **kwargs,
    ):
        graphs = Graphs.empty()
        graphs.bwd = None
        graphs.rg = rg
        facts, graphs.fwd = fwd_and_bwd.create_graph(*args, **kwargs)
        graphs.shard_groups = get_shard_groups(opts, facts.fwd)
        optim_facts = graphs._setup_optim(optimizer, graphs.fwd.args.fwd, opts)
        graphs._set_factories(facts.fwd, optim_facts, facts.pop("bwd"))
        graphs._setup_graphs(opts, entries)
        return graphs

    def _setup_optim(self, optimizer: addons.Module, fwd_vars: NamedTensors, opts):
        optim_facts = {}
        self.optim = {}

        for name, var in fwd_vars.to_dict().items():
            optim_facts[name], self.optim[name] = optimizer.create_graph(
                replica_sharded_spec(var, shard_over=self.shard_groups[name]),
                replica_sharded_spec(var, shard_over=self.shard_groups[name]),
                lr=opts.lr,
                bias_correction=False,
                replica_grouping=popxl.gcg().ir.replica_grouping(group_size=opts.data_parallel),
            )
        return optim_facts

    def _set_factories(self, fwd_facts, optim_facts, grad_facts):
        self.facts = NamedVariableFactories(fwd=fwd_facts, optim=NamedVariableFactories.from_dict(optim_facts))
        self.grad_facts = grad_facts

    def _setup_graphs(self, opts, entries: int):
        # Create remote buffers for fwd vars and optimiser state.

        # only require the group size
        shard_dict = {n: g.group_size for n, g in self.shard_groups.items()}
        optim_shard_dict = {n: g.group_size for n, g in get_shard_groups(opts, self.facts.optim).items()}
        self.buffers = NamedRemoteBuffers(
            fwd=named_variable_buffers(self.facts.fwd, entries, shard_over_dict=shard_dict),
            optim=named_variable_buffers(self.facts.optim, entries, shard_over_dict=optim_shard_dict),
        )
        # Create Graphs for loading/gathering/storing/reducing
        self._fwd_load, self._fwd_load_names = load_remote_graph(self.buffers.fwd, entries)
        self._optim_fwd_load, self._optim_fwd_load_names = load_remote_graph(self.buffers, entries)
        self._optim_fwd_store = store_remote_graph(self.buffers, entries)
        (self._fwd_all_gather, self._fwd_all_gather_names,) = all_gather_replica_sharded_graph(
            NamedTensors.pack(self._fwd_load_names, self._fwd_load.graph.outputs),
            replica_groups=NamedReplicaGrouping.from_dict(dict(zip(self._fwd_load_names, self.shard_groups.values()))),
        )
        grad_accums = self.bwd.args.copy() if self.bwd else self.fwd.args.bwd.copy()
        grad_accums.pop("mean_accum_counter")
        self._grad_reduce, self._grad_reduce_names = reduce_replica_sharded_graph(
            grad_accums,
            "mean",
            shard_groups=NamedReplicaGrouping.from_dict(get_shard_groups(opts, self.grad_facts)),
            replica_group=self.rg,
        )

    # load forward variables
    def fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._fwd_load_names, self._fwd_load.call(i))

    # load forward variables and optimizer state
    def optim_fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._optim_fwd_load_names, self._optim_fwd_load.call(i))

    # store forward variables and optimizer state
    def optim_fwd_store(self, args: NamedTensors, i: Union[int, popxl.Tensor]):
        return self._optim_fwd_store.bind(args).call(i)

    # gathers replica sharded forward variables
    def fwd_all_gather(self, args: NamedTensors):
        return NamedTensors.pack(self._fwd_all_gather_names, self._fwd_all_gather.bind(args).call())

    # reduce scatter gradients
    def grad_reduce(self, args: NamedTensors):
        return NamedTensors.pack(self._grad_reduce_names, self._grad_reduce.bind(args).call())

    # update forward variables
    def optimizer_remote_step(
        self,
        i: int,
        vars_and_state: NamedTensors,
        grads: NamedTensors,
        accum_counter: popxl.Tensor,
    ):
        _variables = vars_and_state.fwd.to_dict()
        _state = vars_and_state.optim
        _grads = grads.accum.to_dict()
        for name, graph in self.optim.items():
            state_clean_names = self._get_optimizer_state(name, _state)
            self.optim[name].bind(state_clean_names).call(_variables[name], _grads[name])
        ops.var_updates.accumulator_scale_(accum_counter, 0.0)

    def _get_optimizer_state(self, name: str, state: NamedTensors) -> NamedTensors:
        attrs = name.split(".")
        for attr in attrs:
            state = getattr(state, attr)
        return state


def input_layer_batch_serialise(
    opts,
    layer_graphs: Graphs,
    x_buffer: popxl.RemoteBuffer,
    dx_buffer: popxl.RemoteBuffer,
    input_stream: popxl.HostToDeviceStream,
):
    fwd_bs, bwd_bs = batch_serialise_fwd_and_grad(
        layer_graphs.fwd,
        layer_graphs.bwd,
        layer_graphs.fwd.args,
        opts.gradient_accumulation,
        load_handles={
            layer_graphs.fwd.graph.inputs[0]: input_stream,
            layer_graphs.bwd.graph.inputs[0]: (dx_buffer, 0),
        },
        store_streams={},
        store_buffers={
            layer_graphs.fwd.graph.outputs[0]: (x_buffer, 0),
        },
        rows=1,
        io_mode=opts.io_mode,
    )
    layer_graphs.fwd = fwd_bs.graph
    layer_graphs.bwd = bwd_bs.graph


def layer_batch_serialise(
    opts,
    layer_graphs: Graphs,
    x_buffer: popxl.RemoteBuffer,
    dx_buffer: popxl.RemoteBuffer,
    rows: int,
):
    fwd_bs, bwd_bs = batch_serialise_fwd_and_grad(
        layer_graphs.fwd,
        layer_graphs.bwd,
        layer_graphs.fwd.args,
        opts.gradient_accumulation,
        load_handles={
            layer_graphs.fwd.graph.inputs[0]: (x_buffer, 0),
            layer_graphs.bwd.graph.inputs[0]: (
                dx_buffer,
                1,
            ),  # load dx from next layer row
        },
        store_streams={},
        store_buffers={
            layer_graphs.fwd.graph.outputs[0]: (
                x_buffer,
                1,
            ),  # store x in next layer row
            layer_graphs.bwd.graph.outputs[0]: (
                dx_buffer,
                0,
            ),  # store dx in previous layer row
        },
        rows=2,
        io_mode=opts.io_mode,
    )
    layer_graphs.fwd = fwd_bs.graph
    layer_graphs.bwd = bwd_bs.graph


def output_layer_batch_serialise(
    opts,
    layer_graphs: Graphs,
    x_buffer: popxl.RemoteBuffer,
    dx_buffer: popxl.RemoteBuffer,
    label_stream: popxl.h2d_stream,
    output_stream: popxl.d2h_stream,
):
    fwd_bs = batch_serialise(
        layer_graphs.fwd,
        opts.gradient_accumulation,
        load_handles={
            layer_graphs.fwd.graph.inputs[0]: (x_buffer, 2),
            layer_graphs.fwd.graph.inputs[1]: label_stream,
        },
        store_streams={layer_graphs.fwd.graph.outputs[1]: output_stream},
        store_buffers={layer_graphs.fwd.graph.outputs[0]: (dx_buffer, 2)},
        rows=1,
        io_mode=opts.io_mode,
    )
    layer_graphs.fwd = fwd_bs.graph


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
                outputs = train_session.run(inputs)
                losses_np = outputs[loss_stream]  # shape(ir.num_host_transfers, ir.replication_factor, )
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
    assert opts.gradient_accumulation > 1
    assert opts.data_parallel > 1

    ir = popxl.Ir()
    ir.replication_factor = opts.data_parallel
    num_inner_layers = 2
    rg = ir.replica_grouping(group_size=opts.data_parallel)
    if opts.io_mode != "compute":
        session_opts = ir._pb_ir.getSessionOptions()
        session_opts.numIOTiles = 32

    with ir.main_graph:
        # -----  Define input and output streams -----
        img_spec = popxl.TensorSpec((opts.train_micro_batch_size, 28 * 28), popxl.float32)
        inner_spec = popxl.TensorSpec((opts.train_micro_batch_size, 512), popxl.float32)

        img_stream = popxl.h2d_stream(img_spec.shape, popxl.float32, "image")
        label_stream = popxl.h2d_stream((opts.train_micro_batch_size,), popxl.int32, "labels")
        loss_stream = popxl.d2h_stream((), popxl.float32, "loss")
        optimizer = Adam(cache=False)

        # ----- Create graphs -----
        fc1 = Graphs(opts, Linear(512, replica_grouping=rg), optimizer, 1, False, rg, img_spec)

        inner_layer = Graphs(
            opts,
            Linear(512, replica_grouping=rg),
            optimizer,
            num_inner_layers,
            True,
            rg,
            inner_spec,
        )

        fc4_fwd_bwd = Graphs.from_fwd_and_bwd(
            opts,
            OutputLayerWithBwd(10, gelu=False, replica_grouping=rg),
            optimizer,
            1,
            rg,
            inner_spec,
            label_stream.spec,
        )

        x_buffer = batch_serial_buffer(
            fc1.fwd.graph.outputs[0],
            steps=opts.gradient_accumulation,
            rows=num_inner_layers + 1,
        )
        dx_buffer = batch_serial_buffer(
            fc1.bwd.graph.inputs[0],
            steps=opts.gradient_accumulation,
            rows=num_inner_layers + 1,
        )

        # ----- Transform graphs -----

        # apply batch serialisation
        input_layer_batch_serialise(opts, fc1, x_buffer, dx_buffer, img_stream)
        layer_batch_serialise(
            opts, inner_layer, x_buffer, dx_buffer, num_inner_layers
        )  # use a buffer with two rows because the inner layer is duplicated two times. row 0 is for fc2 and row 1 for fc3
        output_layer_batch_serialise(opts, fc4_fwd_bwd, x_buffer, dx_buffer, label_stream, loss_stream)

        # ----- Create Variables -----
        variables = NamedTensors()
        variables.insert("fc1", fc1.facts.init_remote(fc1.buffers, 0, "fc1"))
        variables.insert("fc2", inner_layer.facts.init_remote(inner_layer.buffers, 0, "fc2"))
        variables.insert("fc3", inner_layer.facts.init_remote(inner_layer.buffers, 1, "fc3"))
        variables.insert("fc4", fc4_fwd_bwd.facts.init_remote(fc4_fwd_bwd.buffers, 0, "fc4"))

        # ----- Construct Execution Scheme -----

        # Phased Execution (with remote fwd variables and optimizer state). N layers executing separately
        # phase 1: fwd for layer 1:
        #   load fwd variables.
        #   for i in range(gradient_accumulation_steps):
        #        load inputs (xs)
        #        execute fwd
        #        store outputs & activations
        #
        # phase 2: fwd for layer 2:
        # ...
        # phase N: fwd + bwd + optimizer for layer N:
        #   load fwd variables and optimizer state.
        #   for i in range(gradient_accumulation_steps):
        #       load fwd and bwd inputs (xs)
        #       execute fwd
        #       compute loss
        #       execute bwd
        #       store outputs & activations
        #   call optimizer
        #   store updated fwd variables and optimizer state
        #
        # phase N+1: bwd for layer N-1 + optimizer
        #   load fwd variables and optimizer state. both needed, fwd vars are needed from the backward.
        #   for i in range(gradient_accumulation_steps):
        #       load bwd inputs (xs)
        #       execute bwd
        #       store outputs & activations
        #   call optimizer
        #   store updated fwd variables and optimizer state
        # ...
        # phase 2N-1: bwd for layer 1 + optimizer
        #

        # ----- Phased Execution -----
        with popxl.in_sequence(True):

            def forward_phase(graphs: Graphs, row_offset: int):
                vars = graphs.fwd_load(row_offset)  # load forward remote variables, which are sharded
                vars = graphs.fwd_all_gather(vars)  # gathered variables: graph must be bound to gathered vars
                # calling the graph executes the GA loop for the phase: repeat ( load xs - execute - store )
                graphs.fwd.bind(vars).call(row_offset)

            def backward_phase(graphs: Graphs, row_offset: int):
                is_joint_fwd_bwd = graphs.bwd is None
                # forward vars and optimizer state are needed in the backward.
                # loading them together is convenient
                fwd_vars_and_state = graphs.optim_fwd_load(row_offset)  # sharded
                vars: NamedTensors  # gathered variables comprising forward and backward named inputs
                reduced_grads: NamedTensors  # scattered gradient accumulators
                mean_accum_counter: popxl.Tensor
                if is_joint_fwd_bwd:
                    vars = NamedTensors(
                        fwd=graphs.fwd_all_gather(fwd_vars_and_state.fwd),  # gathered forward variables
                        bwd=graphs.grad_facts.init_zero(),  # gradient accumulators
                    )
                    # calling the graph executes the GA loop for the phase: repeat ( load xs - execute fwd compute loss execute bwd - store )
                    graphs.fwd.bind(vars).call(row_offset)  # the fwd graph includes everything
                    reduced_grads = graphs.grad_reduce(vars.bwd)
                    mean_accum_counter = vars.bwd.mean_accum_counter
                else:
                    vars = graphs.fwd_all_gather(fwd_vars_and_state.fwd)
                    grad_accums = graphs.grad_facts.init_zero()  # gradient accumulators
                    vars.update(grad_accums.copy())
                    # calling the graph executes the GA loop for the phase: repeat ( load xs - execute bwd - store )
                    graphs.bwd.bind(vars).call(row_offset)  # just call the batch serialised bwd
                    reduced_grads = graphs.grad_reduce(grad_accums)
                    mean_accum_counter = vars.mean_accum_counter
                # optimizer
                graphs.optimizer_remote_step(row_offset, fwd_vars_and_state, reduced_grads, mean_accum_counter)
                graphs.optim_fwd_store(fwd_vars_and_state, row_offset)  # store updated vars

            # ----- Phase 1 (fwd): fc1 Fwd -----
            forward_phase(fc1, 0)
            # ----- Phase 2 (fwd): fc2 Fwd-----
            forward_phase(inner_layer, 0)
            # ----- Phase 3 (fwd): fc2 Fwd-----
            forward_phase(inner_layer, 1)
            # ----- Phase 4 (merged fwd-bwd): Fwd for output layer, loss,  Bwd for output layer - Optimizer for output layer -----
            backward_phase(fc4_fwd_bwd, 0)
            # ----- Phase 5 (bwd): fc3 bwd - Optimizer for fc3 -----
            backward_phase(inner_layer, 1)
            # ----- Phase 6 (bwd): fc2 bwd - Optimizer for fc2 -----
            backward_phase(inner_layer, 0)
            # ----- Phase 7 (bwd): fc1 bwd - Optimizer for fc1 -----
            backward_phase(fc1, 0)

    # we have a for loop, the number of host loads is equal to gradient_accumulation
    ir.num_host_transfers = opts.gradient_accumulation
    # weights we need to retrieve and copy to the test session. They need to be in the same names as the full model (fc1-fc2-fc3-fc4).
    vars = NamedTensors(
        fc1=variables.fc1.fwd,
        fc2=variables.fc2.fwd,
        fc3=variables.fc3.fwd,
        fc4=variables.fc4.fwd,
    )

    return popxl.Session(ir, "ipu_hw"), [img_stream, label_stream], vars, loss_stream


def test_program(opts):
    ir = popxl.Ir(replication=1)

    with ir.main_graph:
        # Inputs
        in_stream = popxl.h2d_stream((opts.test_batch_size, 28, 28), popxl.float32, "image")
        in_t = ops.host_load(in_stream)

        # Create graphs
        rg = ir.replica_grouping(group_size=1)
        facts, graph = Net(rg).create_graph(in_t)

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
        help="gradient accumulation (default: 8)",
    )
    parser.add_argument(
        "--io-mode",
        type=str,
        default="io",
        help="How to load/store the Tensors during the batch serialisation loop. Can be io, io_overlapped or compute.",
    )
    parser.add_argument(
        "--sharded-threshold",
        type=int,
        default=512,
        help="if the size of a tensor exceeds sharded-threshold the tensor will be sharded",
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
