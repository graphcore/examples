# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import time
import numpy as np
from typing import Dict, Union

import popxl
from popxl import ops
import popxl_addons as addons
from config import CONFIG_DIR
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.transforms.batch_serialisation import batch_serialise_fwd_and_grad, batch_serial_buffer, batch_serialise
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.rts import (
    all_gather_replica_sharded_graph,
    reduce_replica_sharded_graph,
    replica_sharded_spec,
)

from popxl_addons.remote import (
    named_variable_buffers,
    load_remote_graph,
    store_remote_graph,
)

from config import BertConfig
from utils.setup import bert_config_setup
from modelling.embedding import BertEmbeddings
from modelling.bert_model import BertLayer
from modelling.squad import BertSquadLossAndGrad
from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons import TaskSession

__all__ = ["squad_training_phased"]

OptimGraphs = Dict[str, GraphWithNamedArgs]


def requires_weight_decay(t: popxl.Tensor):
    return not any(map(lambda exclude: exclude in t.name, ["norm", "bias"]))


class Graphs:
    def __init__(self,
                 config: BertConfig,
                 layer: addons.Module,
                 optimizer: addons.Module,
                 *args, **kwargs):
        # Create Graphs for computing forward, gradient and optimizer
        fwd_args, self.fwd = layer.create_graph(*args, **kwargs)
        required_grads = () if isinstance(
            layer, BertEmbeddings) else (self.fwd.graph.inputs[0],)
        grad_args, self.grad = addons.autodiff_with_accumulation(
            self.fwd, self.fwd.args.tensors, required_grads)

        optim_args = {}
        self.optim: OptimGraphs = {}
        for name, var in self.fwd.args.to_dict().items():
            optim_args[name], self.optim[name] = optimizer.create_graph(
                replica_sharded_spec(var),
                replica_sharded_spec(var),
                lr=popxl.TensorSpec((), popxl.float32),
                weight_decay=config.training.optimizer.weight_decay if requires_weight_decay(
                    var) else 0.0,
                beta1=config.training.optimizer.beta1,
                beta2=config.training.optimizer.beta2,
                eps=(1e-6 * config.execution.loss_scaling),
                bias_correction=False)

        # Variables required
        self.args = NamedVariableFactories(
            fwd=fwd_args,
            optim=NamedVariableFactories.from_dict(optim_args))
        self.grad_args = grad_args

        # Create remote buffers for fwd vars and optimiser state.
        entries = config.model.layers if isinstance(layer, BertLayer) else 1
        self.buffers = named_variable_buffers(self.args, entries)

        # Create Graphs for loading/gathering/storing/reducing
        self._fwd_load, self._fwd_load_names = load_remote_graph(
            self.buffers.fwd, entries)
        self._optim_load, self._optim_load_names = load_remote_graph(
            self.buffers, entries)
        self._optim_store = store_remote_graph(self.buffers, entries)

        self._fwd_all_gather, self._fwd_all_gather_names = all_gather_replica_sharded_graph(
            NamedTensors.pack(self._fwd_load_names, self._fwd_load.graph.outputs))

        grad_accums = self.grad.args.copy()
        grad_accums.pop("mean_accum_counter")
        self._grad_reduce, self._grad_reduce_names = reduce_replica_sharded_graph(
            grad_accums, 'mean')

    @classmethod
    def empty(cls):
        return super().__new__(cls)

    def fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._fwd_load_names, self._fwd_load.call(i))

    def optim_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._optim_load_names, self._optim_load.call(i))

    def optim_store(self, args: NamedTensors, i: Union[int, popxl.Tensor]):
        return self._optim_store.bind(args).call(i)

    def fwd_all_gather(self, args: NamedTensors):
        return NamedTensors.pack(self._fwd_all_gather_names, self._fwd_all_gather.bind(args).call())

    def grad_reduce(self, args: NamedTensors):
        return NamedTensors.pack(self._grad_reduce_names, self._grad_reduce.bind(args).call())


def create_squad_graph(config: BertConfig, optimizer: addons.Module, *args, **kwargs):
    """Squad combines the forward, loss and grad into a single Module."""
    args, graph = BertSquadLossAndGrad(config).create_graph(*args, **kwargs)

    optim_args: Dict[str, NamedVariableFactories] = {}
    optim_graphs: OptimGraphs = {}
    for name, var in graph.args.fwd.to_dict().items():
        optim_args[name], optim_graphs[name] = optimizer.create_graph(
            replica_sharded_spec(var),
            replica_sharded_spec(var),
            lr=popxl.constant(1e-3),  # TODO: Replace with TensorSpec
            weight_decay=config.training.optimizer.weight_decay if requires_weight_decay(
                var) else 0.0,
            beta1=config.training.optimizer.beta1,
            beta2=config.training.optimizer.beta2,
            eps=(1e-6 * config.execution.loss_scaling),
            bias_correction=False)

    args.insert("optim", NamedVariableFactories.from_dict(optim_args))
    grad_args = args.pop("grad")

    squad = Graphs.empty()
    squad.fwd = graph
    squad.optim = optim_graphs
    squad.args = args
    squad.grad_args = grad_args
    # Create remote buffers for fwd vars and optimiser state.
    squad.buffers = named_variable_buffers(args)
    squad._fwd_load, squad._fwd_load_names = load_remote_graph(
        squad.buffers.fwd)
    squad._optim_load, squad._optim_load_names = load_remote_graph(
        squad.buffers)
    squad._optim_store = store_remote_graph(squad.buffers)
    squad._fwd_all_gather, squad._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(squad._fwd_load_names, squad._fwd_load.graph.outputs))
    grad_accums = graph.args.grad.copy()
    grad_accums.pop("mean_accum_counter")
    squad._grad_reduce, squad._grad_reduce_names = reduce_replica_sharded_graph(
        grad_accums, 'mean')

    return squad


def get_optimizer_state(name: str, state: NamedTensors) -> NamedTensors:
    attrs = name.split(".")
    for attr in attrs:
        state = getattr(state, attr)
    return state


def optimizer_step(optim_graphs: OptimGraphs,
                   vars_and_state: NamedTensors,
                   grads: NamedTensors,
                   lr: popxl.Tensor):
    _variables = vars_and_state.fwd.to_dict()
    _state = vars_and_state.optim
    _grads = grads.accum.to_dict()
    for name, graph in optim_graphs.items():
        graph.bind(get_optimizer_state(name, _state)).call(
            _variables[name], _grads[name], lr)


def embeddings_batch_serialise(config: BertConfig,
                               embeddings: Graphs,
                               inputs: addons.InputStreams,
                               x_buffer: popxl.RemoteBuffer,
                               dx_buffer: popxl.RemoteBuffer):
    fwd, grad = batch_serialise_fwd_and_grad(
        embeddings.fwd,
        embeddings.grad,
        embeddings.fwd.args,
        config.gradient_accumulation,
        load_handles={
            embeddings.fwd.graph.inputs[0]: inputs.words,
            embeddings.fwd.graph.inputs[1]: inputs.token_type,
            embeddings.grad.graph.inputs[0]: (dx_buffer, 0)
        },
        store_streams={},
        store_buffers={
            embeddings.fwd.graph.outputs[0]: (x_buffer, 0)
        },
        seed_input=embeddings.fwd.graph.inputs[2],
        rows=1,
        io_mode='io')
    embeddings.fwd = fwd.graph
    embeddings.grad = grad.graph


def layer_batch_serialise(config: BertConfig,
                          layer: Graphs,
                          x_buffer: popxl.RemoteBuffer,
                          dx_buffer: popxl.RemoteBuffer,
                          mask_buffer: popxl.RemoteBuffer):
    fwd, grad = batch_serialise_fwd_and_grad(
        layer.fwd,
        layer.grad,
        layer.fwd.args,
        config.gradient_accumulation,
        load_handles={
            layer.fwd.graph.inputs[0]: (x_buffer, 0),
            layer.fwd.graph.inputs[1]: (mask_buffer, None),
            layer.grad.graph.inputs[0]: (dx_buffer, 1)
        },
        store_streams={},
        store_buffers={
            layer.fwd.graph.outputs[0]: (x_buffer, 1),
            layer.grad.graph.outputs[0]: (dx_buffer, 0)
        },
        seed_input=layer.fwd.graph.inputs[2],
        rows=config.model.layers,
        io_mode='io')
    layer.fwd = fwd.graph
    layer.grad = grad.graph


def squad_batch_serialise(config: BertConfig,
                          squad: Graphs,
                          inputs: addons.InputStreams,
                          outputs: addons.OutputStreams,
                          x_buffer: popxl.RemoteBuffer,
                          dx_buffer: popxl.RemoteBuffer):
    bs_squad = batch_serialise(
        squad.fwd,
        config.gradient_accumulation,
        load_handles={
            squad.fwd.graph.inputs[0]: (x_buffer, config.model.layers),
            squad.fwd.graph.inputs[1]: inputs.labels
        },
        store_streams={squad.fwd.graph.outputs[0]: outputs.loss},
        store_buffers={squad.fwd.graph.outputs[1]: (dx_buffer, config.model.layers)},
        rows=1,
        io_mode='io')
    squad.fwd = bs_squad.graph


def squad_training_phased(config: BertConfig) -> TaskSession:
    assert config.execution.data_parallel > 1

    ir = popxl.Ir()
    ir.replication_factor = config.execution.data_parallel
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"

    t = time.time()
    main = ir.main_graph

    with main:
        # -----  Define input and output streams -----
        input_shape = (
            config.execution.micro_batch_size * config.model.sequence_length,)
        inputs = addons.InputStreams(
            words=(input_shape, popxl.uint32),
            token_type=(input_shape, popxl.uint32),
            mask=(input_shape, config.model.dtype),
            labels=((config.execution.micro_batch_size, 2), popxl.uint32),
            seed=((2,), popxl.uint32),
            lr=((), popxl.float32)
        )
        outputs = addons.OutputStreams(
            loss=((), config.model.dtype)
        )

        # ----- Build compute graphs -----
        optimizer = AdamOptimizerStep(cache=True)

        embeddings = Graphs(
            config, BertEmbeddings(config), optimizer, inputs.words.spec, inputs.token_type.spec, seed=inputs.seed.spec)
        layer = Graphs(
            config, BertLayer(config), optimizer, embeddings.fwd.graph.outputs[0].spec, inputs.mask.spec, seed=inputs.seed.spec)

        squad = create_squad_graph(
            config,
            optimizer,
            layer.fwd.graph.outputs[0].spec,
            inputs.labels.spec)

        # ---- Transform graphs ----

        # Recomputation
        embeddings.grad = addons.recompute_graph(embeddings.grad)
        layer.grad = addons.recompute_graph(layer.grad)

        # Batch Serialisation
        #   Buffers
        x_buffer = batch_serial_buffer(embeddings.fwd.graph.outputs[0], steps=config.gradient_accumulation, rows=config.model.layers+1)
        dx_buffer = batch_serial_buffer(embeddings.grad.graph.inputs[0], steps=config.gradient_accumulation, rows=config.model.layers+1)
        mask_buffer = batch_serial_buffer(layer.fwd.graph.inputs[1], steps=config.gradient_accumulation)
        #   Graphs
        embeddings_batch_serialise(config, embeddings, inputs, x_buffer, dx_buffer)
        layer_batch_serialise(config, layer, x_buffer, dx_buffer, mask_buffer)
        squad_batch_serialise(config, squad, inputs, outputs, x_buffer, dx_buffer)

        # Available Memory Proportion
        addons.set_available_memory_proportion_by_ipu(
            ir, config.execution.available_memory_proportion)

        # ----- Create Variables -----

        variables = NamedTensors()
        variables.insert("embeddings", embeddings.args.init_remote(
            embeddings.buffers, 0, "embeddings"))
        variables.insert("squad", squad.args.init_remote(
            squad.buffers, 0, "squad"))
        variables.insert("layer", NamedTensors.from_dict({
            n: layer.args.init_remote(layer.buffers, n, f"layer.{n}")
            for n in range(config.model.layers)
        }))

        # ---- Execute ----

        with popxl.in_sequence():
            # Load current learning rate
            with popxl.transforms.merge_exchange(), popxl.in_sequence(False):
                seed = ops.host_load(inputs.seed)
                lr = ops.host_load(inputs.lr)

            @popxl.io_tiles()
            def fill_buffer_from_host(i: popxl.Tensor, stream: popxl.HostToDeviceStream, buffer: popxl.RemoteBuffer):
                t = ops.host_load(stream)
                ops.remote_store(buffer, i, t)

            # Load from host then store all masks TODO: Move this into the embedding loop
            mask_fill_graph = ir.create_graph(
                fill_buffer_from_host, popxl.constant(0, popxl.uint32), inputs.mask, mask_buffer)
            for i in range(config.gradient_accumulation):
                ops.call(mask_fill_graph, i)

            def embedding_fwd_phase(seed):
                # Load Embedding layer
                embeddings_vars = embeddings.fwd_load(0)
                embeddings_vars = embeddings.fwd_all_gather(embeddings_vars)
                # Forward
                seed, embed_seed = ops.split_random_seed(seed)
                embeddings.fwd.bind(embeddings_vars).call(0, embed_seed)
                return seed

            seed = embedding_fwd_phase(seed)

            def single_bert_layer_fwd_phase(n: popxl.Tensor, seed: popxl.Tensor):
                # Load Encoder layers
                layer_vars = layer.fwd_load(n)
                layer_vars = layer.fwd_all_gather(layer_vars)
                # Forward
                seed, layer_seed = ops.split_random_seed(seed)
                layer.fwd.bind(layer_vars).call(n, layer_seed)
                return n + 1, seed

            i = popxl.constant(0)
            bwd_graph = ir.create_graph(single_bert_layer_fwd_phase, i, seed)
            ops.repeat(bwd_graph, config.model.layers, i, seed)

            def squad_fwd_grad_optimizer_phase():
                # Load Squad layer
                squad_vars = squad.optim_load(0)
                squad_fwd_vars = NamedTensors(
                    fwd=squad.fwd_all_gather(squad_vars.fwd),
                    grad=squad.grad_args.init_zero())
                # Forward + Gradient
                squad.fwd.bind(squad_fwd_vars).call(0)
                # Optimizer
                reduced_grads = squad.grad_reduce(squad_fwd_vars.grad)
                optimizer_step(squad.optim, squad_vars, reduced_grads, lr)
                # Store
                squad.optim_store(squad_vars, 0)

            squad_fwd_grad_optimizer_phase()

            def single_bert_layer_grad_optimizer_phase(n: popxl.Tensor, lr: popxl.Tensor):
                layer_vars = layer.optim_load(n)
                layer_fwd_vars = layer.fwd_all_gather(layer_vars.fwd)
                # Gradient
                grads = layer.grad_args.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(layer_fwd_vars)
                layer.grad.bind(bwd_vars).call(n)
                # Optimizer
                reduced_grads = layer.grad_reduce(grads)
                optimizer_step(layer.optim, layer_vars, reduced_grads, lr)
                # Store
                layer.optim_store(layer_vars, n)
                return n - 1

            i = popxl.constant(config.model.layers - 1)
            bwd_graph = ir.create_graph(single_bert_layer_grad_optimizer_phase, i, lr)
            ops.repeat(bwd_graph, config.model.layers, i, lr)

            def embedding_grad_optimizer_phase():
                # Load Embeddings layer
                embeddings_vars = embeddings.optim_load(0)
                embeddings_fwd_vars = embeddings.fwd_all_gather(
                    embeddings_vars.fwd)
                # Gradient
                grads = embeddings.grad_args.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(embeddings_fwd_vars)
                embeddings.grad.bind(bwd_vars).call(0)
                # Optimizer
                reduced_grads = embeddings.grad_reduce(grads)
                optimizer_step(embeddings.optim, embeddings_vars, reduced_grads, lr)
                # Store
                embeddings.optim_store(embeddings_vars, 0)

            embedding_grad_optimizer_phase()

    repeat_graph(main, config.execution.device_iterations)

    fwd_vars = NamedTensors(
        embeddings=variables.embeddings.fwd,
        layer=NamedTensors.from_dict(
            {i: variables.layer[i].fwd for i in range(config.model.layers)}),
        squad=variables.squad.fwd,
    )

    logging.info(f"popxl IR construction duration: {(time.time() - t) / 60:.2f} mins")

    ir.num_host_transfers = config.execution.device_iterations * \
        config.gradient_accumulation
    session = TaskSession(
        inputs,
        outputs,
        fwd_vars,
        ir,
        "ipu_hw"
    )
    return session


def main():
    """Run a benchmark configuration"""
    config, _ = bert_config_setup(
        CONFIG_DIR / "squad_training.yml",
        "phased",
        "large")
    session = squad_training_phased(config)

    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()}

    with session:
        # Skip one result
        session.run(inputs)

        durations = []
        for _ in range(5):
            start = time.time()
            session.run(inputs)
            durations.append(time.time() - start)
    duration = np.mean(durations)

    samples_per_step = config.execution.device_iterations * config.training.global_batch_size
    result_str = \
        f"Duration: {duration} s " \
        f"Throughput: {samples_per_step/duration:6.1f} samples/s "
    logging.info(result_str)


if __name__ == "__main__":
    main()
