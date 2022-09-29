# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
import time
import numpy as np
from typing import Dict, List, Union

import popxl
from popxl import ops
import popxl_addons as addons
from popxl_addons import TaskSession
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.optimizers.adam import AdamOptimizerStep
from popxl_addons.transforms.repeat_graph import repeat_graph
from popxl_addons.transforms.batch_serialisation import batch_serialise_fwd_and_grad, batch_serial_buffer, batch_serialise
from popxl_addons.rts import (
    all_gather_replica_sharded_graph,
    replica_sharded_spec,
)
from popxl_addons.rts import reduce_replica_sharded_graph
from popxl_addons.remote import (named_variable_buffers, load_remote_graph, store_remote_graph, NamedRemoteBuffers)
from popxl_addons.ops.grad_reduce_square_add import grad_reduce_square_add

from config import BertConfig, CONFIG_DIR
from utils.setup import bert_config_setup
from modelling.embedding import BertEmbeddings
from modelling.bert_model import BertLayer, BertPretrainingLossAndGrad

__all__ = ["pretraining_phased"]

OptimGraphs = Dict[str, GraphWithNamedArgs]


def requires_weight_decay(t: popxl.Tensor):
    return not any(map(lambda exclude: exclude in t.name, ["norm", "bias"]))


def optimizer_graphs(config: BertConfig, optimizer: addons.Module, variables: NamedTensors):
    optim_args = {}
    optim_graphs = {}
    for name, var in variables.to_dict().items():
        optim_args[name], optim_graphs[name] = optimizer.create_graph(
            replica_sharded_spec(var),
            replica_sharded_spec(var),
            lr=popxl.TensorSpec((), popxl.float32),
            weight_decay=config.training.optimizer.weight_decay if requires_weight_decay(var) else 0.0,
            beta1=config.training.optimizer.beta1,
            beta2=config.training.optimizer.beta2,
            eps=1e-6,
            bias_correction=True,
            first_order_dtype=popxl.float32,
            loss_scaling=config.execution.loss_scaling,
            global_norm=popxl.TensorSpec((), popxl.float32),
            global_norm_max=config.training.optimizer.gradient_clipping)
    return NamedVariableFactories.from_dict(optim_args), optim_graphs


class Graphs:
    def __init__(self):
        self.fwd: GraphWithNamedArgs
        self.grad: GraphWithNamedArgs
        self.optim: OptimGraphs
        self.args: NamedVariableFactories
        self.grad_args: NamedVariableFactories
        self.buffers: NamedRemoteBuffers

        self._fwd_load: GraphWithNamedArgs
        self._fwd_load_names: List[str]
        self._grad_store: GraphWithNamedArgs
        self._optim_load: GraphWithNamedArgs
        self._optim_load_names: List[str]
        self._optim_store: GraphWithNamedArgs
        self._fwd_all_gather: GraphWithNamedArgs
        self._fwd_all_gather_names: List[str]
        self._grad_reduce: GraphWithNamedArgs
        self._grad_reduce_names: List[str]

    def fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._fwd_load_names, self._fwd_load.call(i))

    def grad_store(self, args: NamedTensors, i: Union[float, popxl.Tensor]):
        return self._grad_store.bind(args).call(i)

    def optim_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._optim_load_names, self._optim_load.call(i))

    def optim_store(self, args: NamedTensors, i: Union[int, popxl.Tensor]):
        return self._optim_store.bind(args).call(i)

    def fwd_all_gather(self, args: NamedTensors):
        return NamedTensors.pack(self._fwd_all_gather_names, self._fwd_all_gather.bind(args).call())

    def grad_reduce(self, args: NamedTensors):
        return NamedTensors.pack(self._grad_reduce_names, self._grad_reduce.bind(args).call())


def create_embeddings_graph(config: BertConfig, optimizer: addons.Module, *args, **kwargs):
    embeddings = Graphs()
    # Create Graphs for computing forward, gradient and optimizer
    fwd_args, embeddings.fwd = BertEmbeddings(config).create_graph(*args, **kwargs)
    # Embedding needs no onward gradients
    required_grads = ()
    grad_args, embeddings.grad = addons.autodiff_with_accumulation(embeddings.fwd, embeddings.fwd.args.tensors,
                                                                   required_grads)

    optim_args, embeddings.optim = optimizer_graphs(config, optimizer, embeddings.fwd.args)

    # Variables required
    embeddings.args = NamedVariableFactories(fwd=fwd_args, optim=optim_args)
    embeddings.grad_args = grad_args

    # Create remote buffers
    # Embedding optimizer step happens straight after the bwd.. so no need to store the gradient in a buffer.
    embeddings.buffers = named_variable_buffers(embeddings.args)

    # Create Graphs for loading/gathering/storing/reducing
    embeddings._optim_load, embeddings._optim_load_names = load_remote_graph(embeddings.buffers)
    embeddings._optim_store = store_remote_graph(embeddings.buffers)

    embeddings._fwd_load, embeddings._fwd_load_names = load_remote_graph(embeddings.buffers.fwd)
    embeddings._fwd_all_gather, embeddings._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(embeddings._fwd_load_names, embeddings._fwd_load.graph.outputs))

    grad_accums = embeddings.grad.args.copy()
    grad_accums.pop("mean_accum_counter")
    embeddings._grad_reduce, embeddings._grad_reduce_names = reduce_replica_sharded_graph(grad_accums, 'mean')

    return embeddings


def create_layer_graph(config: BertConfig, optimizer: addons.Module, *args, **kwargs):
    layer = Graphs()
    # Create Graphs for computing forward, gradient and optimizer
    fwd_args, layer.fwd = BertLayer(config).create_graph(*args, **kwargs)
    required_grads = (layer.fwd.graph.inputs[0], )
    grad_args, layer.grad = addons.autodiff_with_accumulation(layer.fwd, layer.fwd.args.tensors, required_grads)

    optim_args, layer.optim = optimizer_graphs(config, optimizer, layer.fwd.args)

    # Variables required
    layer.args = NamedVariableFactories(fwd=fwd_args, optim=optim_args)
    layer.grad_args = grad_args

    # Create remote buffers
    entries = config.model.layers
    buffer_args = layer.args.copy()
    buffer_args.insert("grad", grad_args.copy())
    buffer_args.grad.pop("mean_accum_counter")
    layer.buffers = named_variable_buffers(buffer_args, entries)

    # Create Graphs for loading/gathering/storing/reducing
    # Load fwd, grad and optim
    layer._optim_load, layer._optim_load_names = load_remote_graph(layer.buffers, entries)

    buffers = layer.buffers.copy()
    buffers_grad = buffers.pop("grad")
    # Store fwd and optim
    layer._optim_store = store_remote_graph(buffers, entries)
    # Store grad
    layer._grad_store = store_remote_graph(buffers_grad, entries)

    layer._fwd_load, layer._fwd_load_names = load_remote_graph(layer.buffers.fwd, entries)
    layer._fwd_all_gather, layer._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(layer._fwd_load_names, layer._fwd_load.graph.outputs))

    grad_accums = layer.grad.args.copy()
    grad_accums.pop("mean_accum_counter")
    layer._grad_reduce, layer._grad_reduce_names = reduce_replica_sharded_graph(grad_accums, 'mean')

    return layer


def create_task_head_graph(config: BertConfig, optimizer: addons.Module, embeddings: Graphs, *args, **kwargs):
    """Combines the MLM+NSP forward, loss and grad into a single Module."""
    head = Graphs()

    args, graph = BertPretrainingLossAndGrad(config).create_graph(*args, **kwargs)

    # We're gonna tie this weight from the Embedding.
    optim_ts = graph.args.fwd.copy()
    optim_args, optim_graphs = optimizer_graphs(config, optimizer, optim_ts)

    args.insert("optim", optim_args)
    grad_args = args.pop("grad")

    head.fwd = graph
    head.optim = optim_graphs
    head.args = args
    head.grad_args = grad_args

    # Create remote buffers
    buffer_args = head.args.copy()
    buffer_args.insert("grad", head.grad_args.copy())

    # 3. Don't create a remote buffer for the tied weight grad
    buffer_args.grad.mlm.pop("mean_accum_counter")
    buffer_args.grad.nsp.pop("mean_accum_counter")

    head.buffers = named_variable_buffers(buffer_args)

    head._optim_load, head._optim_load_names = load_remote_graph(head.buffers)
    buffers = head.buffers.copy()
    buffers_grad = buffers.pop("grad")
    head._optim_store = store_remote_graph(head.buffers)
    head._grad_store = store_remote_graph(buffers_grad)

    # 3. Add the tied weight buffer to the buffers to the fwd load
    head.buffers.fwd.mlm.insert("weight", embeddings.buffers.fwd.word.weight)

    head._fwd_load, head._fwd_load_names = load_remote_graph(head.buffers.fwd)
    head._fwd_all_gather, head._fwd_all_gather_names = all_gather_replica_sharded_graph(
        NamedTensors.pack(head._fwd_load_names, head._fwd_load.graph.outputs))

    grad_accums = graph.args.grad.copy()
    # 6. All reduce the tied weight separately
    grad_accums.mlm.pop("mean_accum_counter")
    grad_accums.nsp.pop("mean_accum_counter")
    head._grad_reduce, head._grad_reduce_names = reduce_replica_sharded_graph(grad_accums, 'mean')

    return head


def embeddings_batch_serialise(config: BertConfig, embeddings: Graphs, input_streams: addons.InputStreams,
                               x_buffer: popxl.RemoteBuffer, dx_buffer: popxl.RemoteBuffer):
    fwd, grad = batch_serialise_fwd_and_grad(embeddings.fwd,
                                             embeddings.grad,
                                             embeddings.fwd.args,
                                             config.gradient_accumulation,
                                             load_handles={
                                                 embeddings.fwd.graph.inputs[0]: input_streams.words,
                                                 embeddings.fwd.graph.inputs[1]: input_streams.token_type,
                                                 embeddings.grad.graph.inputs[0]: (dx_buffer, 0)
                                             },
                                             store_streams={},
                                             store_buffers={embeddings.fwd.graph.outputs[0]: (x_buffer, 0)},
                                             seed_input=embeddings.fwd.graph.inputs[2],
                                             rows=1,
                                             io_mode='io_overlapped')
    embeddings.fwd = fwd.graph
    embeddings.grad = grad.graph


def layer_batch_serialise(config: BertConfig, layer: Graphs, x_buffer: popxl.RemoteBuffer,
                          dx_buffer: popxl.RemoteBuffer, mask_buffer: popxl.RemoteBuffer):
    fwd, grad = batch_serialise_fwd_and_grad(layer.fwd,
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
                                             io_mode='io_overlapped')
    layer.fwd = fwd.graph
    layer.grad = grad.graph


def head_batch_serialise(config: BertConfig, head_graph: GraphWithNamedArgs, input_streams: addons.InputStreams,
                         output_streams: addons.OutputStreams, x_buffer: popxl.RemoteBuffer,
                         dx_buffer: popxl.RemoteBuffer) -> GraphWithNamedArgs:
    bs_head = batch_serialise(head_graph,
                              config.gradient_accumulation,
                              load_handles={
                                  head_graph.graph.inputs[0]: (x_buffer, config.model.layers),
                                  head_graph.graph.inputs[3]: input_streams.masked_positions,
                                  head_graph.graph.inputs[4]: input_streams.mlm_labels,
                                  head_graph.graph.inputs[5]: input_streams.nsp_labels,
                              },
                              store_streams={head_graph.graph.outputs[0]: output_streams.loss},
                              store_buffers={head_graph.graph.outputs[1]: (dx_buffer, config.model.layers)},
                              io_mode='io_overlapped')
    return bs_head.graph


def get_optimizer_state(name: str, state: NamedTensors) -> NamedTensors:
    attrs = name.split(".")
    for attr in attrs:
        state = getattr(state, attr)
    return state


def optimizer_step(optim_graphs: OptimGraphs, ts: NamedTensors, lr: popxl.Tensor, global_norm: popxl.Tensor):
    _variables = ts.fwd.to_dict()
    _state = ts.optim
    _grads = ts.grad.accum.to_dict()
    for name, graph in optim_graphs.items():
        graph.bind(get_optimizer_state(name, _state)).call(_variables[name], _grads[name], lr, global_norm)


def task_head_optimizer_step(optim_graphs: OptimGraphs, ts: NamedTensors, lr: popxl.Tensor, global_norm: popxl.Tensor):
    _variables = ts.fwd.to_dict()
    _state = ts.optim
    _grads = {name.replace(".accum", ""): t for name, t in ts.grad.to_dict().items()}
    for name, graph in optim_graphs.items():
        graph.bind(get_optimizer_state(name, _state)).call(_variables[name], _grads[name], lr, global_norm)


def global_norm_reduce(config: BertConfig, grad_norm: popxl.Tensor, grads: NamedTensors):
    for g in grads.tensors:
        ops.add_(grad_norm, grad_reduce_square_add(g, config.execution.loss_scaling))


def pretraining_phased(config: BertConfig) -> TaskSession:
    assert config.execution.data_parallel > 1

    ir = popxl.Ir()
    ir.replication_factor = config.execution.data_parallel
    opts = ir._pb_ir.getSessionOptions()
    opts.numIOTiles = config.execution.io_tiles
    opts.enableStochasticRounding = config.training.stochastic_rounding
    opts.partialsTypeMatMuls = "half"
    opts.engineOptions["target.syncReplicasIndependently"] = "true"

    t = time.time()
    main = ir.main_graph

    with main:
        # -----  Define input and output streams -----
        input_shape = (config.execution.micro_batch_size * config.model.sequence_length, )
        input_streams = addons.InputStreams(words=(input_shape, popxl.uint32),
                                            token_type=(input_shape, popxl.uint32),
                                            mask=(input_shape, config.model.dtype),
                                            masked_positions=((
                                                config.execution.micro_batch_size,
                                                config.model.mlm.mask_tokens,
                                            ), popxl.uint32),
                                            mlm_labels=((
                                                config.execution.micro_batch_size,
                                                config.model.mlm.mask_tokens,
                                            ), popxl.uint32),
                                            nsp_labels=((config.execution.micro_batch_size, ), popxl.uint32),
                                            seed=((2, ), popxl.uint32),
                                            lr=((), popxl.float32))
        output_streams = addons.OutputStreams(loss=((), config.model.dtype), grad_norm=((), popxl.float32))

        # ----- Build compute graphs -----
        optimizer = AdamOptimizerStep(cache=True)

        embeddings = create_embeddings_graph(config,
                                             optimizer,
                                             input_streams.words.spec,
                                             input_streams.token_type.spec,
                                             seed=input_streams.seed.spec)

        layer = create_layer_graph(config,
                                   optimizer,
                                   embeddings.fwd.graph.outputs[0],
                                   input_streams.mask.spec,
                                   seed=input_streams.seed.spec)

        tied_weight_spec = popxl.TensorSpec(
            (embeddings.fwd.args.word.weight.shape[1], embeddings.fwd.args.word.weight.shape[0]),
            dtype=config.model.dtype)

        head = create_task_head_graph(config, optimizer, embeddings, layer.fwd.graph.outputs[0], tied_weight_spec,
                                      tied_weight_spec, input_streams.masked_positions.spec,
                                      input_streams.mlm_labels.spec, input_streams.nsp_labels.spec)

        # ---- Transform graphs ----

        # Recomputation
        embeddings.grad = addons.recompute_graph(embeddings.grad)
        layer.grad = addons.recompute_graph(layer.grad)

        # Batch Serialisation
        #   Buffers
        x_buffer = batch_serial_buffer(embeddings.fwd.graph.outputs[0],
                                       steps=config.gradient_accumulation,
                                       rows=config.model.layers + 1)
        dx_buffer = batch_serial_buffer(embeddings.grad.graph.inputs[0],
                                        steps=config.gradient_accumulation,
                                        rows=config.model.layers + 1)
        mask_buffer = batch_serial_buffer(layer.fwd.graph.inputs[1], steps=config.gradient_accumulation)
        #   Graphs
        embeddings_batch_serialise(config, embeddings, input_streams, x_buffer, dx_buffer)
        layer_batch_serialise(config, layer, x_buffer, dx_buffer, mask_buffer)
        head.fwd = head_batch_serialise(config, head.fwd, input_streams, output_streams, x_buffer, dx_buffer)

        # Available Memory Proportion
        addons.set_available_memory_proportion_by_ipu(ir, config.execution.available_memory_proportion)

        # ----- Create Variables -----

        variables = NamedTensors()
        variables.insert("embeddings", embeddings.args.init_remote(embeddings.buffers, 0, "embeddings"))
        variables.insert("head", head.args.init_remote(head.buffers, 0, "head"))
        variables.insert(
            "layer",
            NamedTensors.from_dict(
                {n: layer.args.init_remote(layer.buffers, n, f"layer.{n}")
                 for n in range(config.model.layers)}))

        # ---- Execute ----

        with popxl.in_sequence():
            # Load current learning rate
            with popxl.transforms.merge_exchange(), popxl.in_sequence(False):
                seed = ops.host_load(input_streams.seed)
                lr = ops.host_load(input_streams.lr)

            @popxl.io_tiles()
            def fill_buffer_from_host(i: popxl.Tensor, stream: popxl.HostToDeviceStream, buffer: popxl.RemoteBuffer):
                t = ops.host_load(stream)
                ops.remote_store(buffer, i, t)
                return i + 1

            # Load from host then store all masks
            i = popxl.constant(0)
            mask_fill_graph = ir.create_graph(fill_buffer_from_host, i, input_streams.mask, mask_buffer)
            ops.repeat(mask_fill_graph, config.gradient_accumulation, i)

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

            i = popxl.constant(0, name="layer_index")
            bwd_graph = ir.create_graph(single_bert_layer_fwd_phase, i, seed)
            ops.repeat(bwd_graph, config.model.layers, i, seed)

            grad_norm = ops.init((), popxl.float32, name='grad_norm', init_type='zero')

            def task_head_fwd_grad_phase():
                # Load task head layer
                head_vars = head.fwd_load(0)
                head_vars = NamedTensors(fwd=head.fwd_all_gather(head_vars), grad=head.grad_args.init_zero())
                # Tied weight
                tied_weight_t = ops.transpose_(head_vars.fwd.mlm.pop('weight'))
                tied_weight_grad_t = ops.init(tied_weight_t.shape, tied_weight_t.dtype, 'word_embedding_grad_t', "zero")
                # Forward + Gradient
                head.fwd.bind(head_vars).call(0, tied_weight_t, tied_weight_grad_t)
                # Data parallel reduce
                reduced_grads = head.grad_reduce(head_vars.grad)
                # Global Norm calculation
                global_norm_reduce(config, grad_norm, reduced_grads.mlm)
                global_norm_reduce(config, grad_norm, reduced_grads.nsp)
                # Store Gradients
                head.grad_store(reduced_grads, 0)

                # Reduce and Store the tied gradient
                grad_t = ops.collectives.replicated_reduce_scatter(ops.transpose_(tied_weight_grad_t),
                                                                   'mean',
                                                                   configure_output_for_replicated_tensor_sharding=True)
                tied_weight_grad_buffer = popxl.remote_buffer(grad_t.shape, grad_t.dtype, 1)
                tied_weight_grad_buffer.meta_shape = grad_t.meta_shape
                ops.remote_store(tied_weight_grad_buffer, 0, grad_t)
                return tied_weight_grad_buffer

            tied_weight_grad_buffer = task_head_fwd_grad_phase()

            def single_bert_layer_grad_phase(n: popxl.Tensor, grad_norm: popxl.TensorByRef):
                # Load layer
                layer_vars = layer.fwd_load(n)
                layer_vars = layer.fwd_all_gather(layer_vars)
                # Gradient
                grads = layer.grad_args.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(layer_vars)
                layer.grad.bind(bwd_vars).call(n)
                # Data parallel reduce
                reduced_grads = layer.grad_reduce(grads)
                # Global Norm calculation
                global_norm_reduce(config, grad_norm, reduced_grads)
                # Store gradient
                layer.grad_store(reduced_grads, n)
                return n - 1

            i = popxl.constant(config.model.layers - 1, name="layer_index")
            bwd_graph = ir.create_graph(single_bert_layer_grad_phase, i, grad_norm)
            ops.repeat(bwd_graph, config.model.layers, i, grad_norm)

            def embedding_grad_optimizer_phase(tied_weight_grad_buffer, grad_norm: popxl.TensorByRef):
                # Load Embeddings layer
                embeddings_vars = embeddings.optim_load(0)
                embeddings_fwd_vars = embeddings.fwd_all_gather(embeddings_vars.fwd)
                # Gradient
                grads = embeddings.grad_args.init_zero()
                bwd_vars = grads.copy()
                bwd_vars.update(embeddings_fwd_vars)
                embeddings.grad.bind(bwd_vars).call(0)
                # Data parallel reduce
                reduced_grads = embeddings.grad_reduce(grads)

                # Add the tied gradient from the projection
                tied_weight_grad = ops.remote_load(tied_weight_grad_buffer, 0)
                ops.add_(reduced_grads.accum.word.weight, tied_weight_grad)

                # Global Norm calculation
                global_norm_reduce(config, grad_norm, reduced_grads)
                # Finalise global grad norm with an all reduce and sqrt
                grad_norm = ops.sqrt(ops.collectives.replicated_all_reduce(grad_norm, op='add'))
                ops.host_store(output_streams.grad_norm, grad_norm)

                # Optimizer Step for Embeddings.
                # Note: No need to store then load the gradient.. just use it directly
                embeddings_vars.insert("grad", reduced_grads)
                optimizer_step(embeddings.optim, embeddings_vars, lr, grad_norm)
                # Store
                embeddings.optim_store(embeddings_vars, 0)
                return grad_norm

            grad_norm = embedding_grad_optimizer_phase(tied_weight_grad_buffer, grad_norm)

            # Optimizer Step for Layers
            def layer_optim(n: popxl.Tensor, lr: popxl.Tensor, grad_norm: popxl.Tensor):
                layer_vars = layer.optim_load(n)
                optimizer_step(layer.optim, layer_vars, lr, grad_norm)
                layer.optim_store(layer_vars, n)
                return n + 1

            i = popxl.constant(0, name="layer_index")
            optim_graph = ir.create_graph(layer_optim, i, lr, grad_norm)
            ops.repeat(optim_graph, config.model.layers, i, lr, grad_norm)

            # Optimizer Step for Task Head
            head_vars = head.optim_load(0)
            task_head_optimizer_step(head.optim, head_vars, lr, grad_norm)
            # Store
            head.optim_store(head_vars, 0)

    repeat_graph(main, config.execution.device_iterations)

    fwd_vars = NamedTensors(
        embeddings=variables.embeddings.fwd,
        layer=NamedTensors.from_dict({i: variables.layer[i].fwd
                                      for i in range(config.model.layers)}),
        head=variables.head.fwd,
    )

    logging.info(f"popxl IR construction duration: {(time.time() - t) / 60:.2f} mins")

    ir.num_host_transfers = config.execution.device_iterations * \
        config.gradient_accumulation
    session = TaskSession(input_streams, output_streams, fwd_vars, ir, "ipu_hw")
    return session


def main():
    """Run a benchmark configuration"""
    config, _ = bert_config_setup(CONFIG_DIR / "pretraining.yml", "phased", "large_128")
    session = pretraining_phased(config)

    inputs = {
        stream: np.ones(session._full_input_shape(stream.shape), stream.dtype.as_numpy())
        for stream in session.expected_inputs()
    }

    with session:
        # Skip one result
        session.run(inputs)

        durations = []
        for _ in range(5):
            start = time.perf_counter()
            session.run(inputs)
            durations.append(time.perf_counter() - start)
    duration = np.mean(durations)

    samples_per_step = config.execution.device_iterations * config.training.global_batch_size
    result_str = \
        f"Duration: {duration} s " \
        f"throughput: {samples_per_step/duration:6.1f} samples/sec "
    logging.info(result_str)


if __name__ == "__main__":
    main()
