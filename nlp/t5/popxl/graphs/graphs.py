# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
from typing import Dict, List, Union

import popxl
import popxl_addons as addons
from popxl_addons.graph import GraphWithNamedArgs
from popxl_addons.variable_factory import NamedVariableFactories
from popxl_addons.named_replica_grouping import NamedReplicaGrouping, get_ild_replica_grouping
from popxl_addons.named_tensors import NamedTensors
from popxl_addons.rts import replica_sharded_spec
from popxl_addons.remote import NamedRemoteBuffers

from config import T5Config


OptimGraphs = Dict[str, GraphWithNamedArgs]
RTS_THRESHOLD = 0
RTS_ACTIVATIONS_THRESHOLD = 0
use_io_tiles = False


class Graphs:
    def __init__(self):
        self.fwd: GraphWithNamedArgs
        self.bwd: GraphWithNamedArgs
        self.optim: OptimGraphs
        self.facts: NamedVariableFactories
        self.grad_facts: NamedVariableFactories
        self.buffers: NamedRemoteBuffers
        self._fwd_load: GraphWithNamedArgs
        self._fwd_load_names: List[str]
        self._grad_store: GraphWithNamedArgs
        self._optim_fwd_load: GraphWithNamedArgs
        self._optim_fwd_load_names: List[str]
        self._optim_fwd_store: GraphWithNamedArgs
        self._fwd_all_gather: GraphWithNamedArgs
        self._fwd_all_gather_names: List[str]
        self._grad_reduce: GraphWithNamedArgs
        self._grad_reduce_names: List[str]

    def fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._fwd_load_names, self._fwd_load.call(i))

    def grad_store(self, args: NamedTensors, i: Union[float, popxl.Tensor]):
        return self._grad_store.bind(args).call(i)

    def optim_fwd_load(self, i: Union[int, popxl.Tensor]):
        return NamedTensors.pack(self._optim_fwd_load_names, self._optim_fwd_load.call(i))

    def optim_fwd_store(self, args: NamedTensors, i: Union[int, popxl.Tensor]):
        return self._optim_fwd_store.bind(args).call(i)

    def fwd_all_gather(self, args: NamedTensors):
        return NamedTensors.pack(self._fwd_all_gather_names, self._fwd_all_gather.bind(args).call())

    def grad_reduce(self, args: NamedTensors):
        return NamedTensors.pack(self._grad_reduce_names, self._grad_reduce.bind(args).call())


def get_activ_shard_group(a: popxl.Tensor, shard_group: popxl.ReplicaGrouping):
    return shard_group if a.nelms >= RTS_ACTIVATIONS_THRESHOLD else popxl.gcg().ir.replica_grouping(group_size=1)


def get_rts_groups(facts: NamedVariableFactories) -> NamedReplicaGrouping:
    ir = popxl.gcg().ir

    rts_groups = {}
    for k, f in facts.to_dict().items():
        size = np.prod(f.shape)
        rg = f.replica_grouping
        # Limit RTS to within an ILD
        rg = get_ild_replica_grouping(rg)
        if size % rg.group_size == 0 and size >= RTS_THRESHOLD:
            rts_groups[k] = rg
        else:
            rts_groups[k] = ir.replica_grouping(group_size=1)
    return NamedReplicaGrouping.from_dict(rts_groups)


def requires_weight_decay(t: popxl.Tensor):
    return not any(map(lambda exclude: exclude in t.name, ["norm", "bias"]))


def optimizer_graphs(
    config: T5Config,
    optimizer: addons.Module,
    variables: NamedTensors,
    replica_groups: NamedReplicaGrouping,
    shard_groups: NamedReplicaGrouping,
):
    optim_facts = {}
    optim_graphs = {}
    replica_groups = replica_groups.to_dict()
    shard_groups = shard_groups.to_dict()
    for name, var in variables.to_dict().items():
        # Currently assumes grads have the same replica group as their var
        input_spec = replica_sharded_spec(var, shard_over=shard_groups[name])
        optim_facts[name], optim_graphs[name] = optimizer.create_graph(
            input_spec,
            input_spec,
            lr=popxl.TensorSpec((), popxl.float32),
            replica_grouping=replica_groups[name],
            weight_decay=config.training.optimizer.weight_decay if requires_weight_decay(var) else 0.0,
            beta1=config.training.optimizer.beta1,
            beta2=config.training.optimizer.beta2,
            eps=1e-6,
            bias_correction=True,
            first_order_dtype=popxl.float32,
            loss_scaling=config.execution.loss_scaling,
            global_norm=popxl.TensorSpec((), popxl.float32),
            global_norm_max=config.training.optimizer.gradient_clipping,
        )
    return NamedVariableFactories.from_dict(optim_facts), optim_graphs
