# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Union, List, Dict, Iterable

import dataclasses
import popxl
from dataclasses import dataclass
from typing_extensions import Literal

from config import GPTConfig
from modelling.decoder import GPTDecoderBlockTP2D
from modelling.embedding import GPTEmbeddingsTP2D
from modelling.gpt_lm import GPTLMHeadLossTP2D
from popxl_addons import NamedTensors
from popxl_addons.dot_tree import DotTree


# --- CONFIG ---

RTS_THRESHOLD = 0
RTS_ACTIVATIONS_THRESHOLD = 0
USE_IO_TILES = False
ListOrGlob = Union[List[str], Literal["*"]]
RTS_ACT = True  # RTS Activations


def gen_layer_config(config: GPTConfig):
    return {
        GPTEmbeddingsTP2D: LayerConf(
            GraphConf(
                accumulate=["word.weight", "positional.weight"],
                remote_buffer_fwd=True,
                remote_buffer_bwd=False,
                remote_buffer_optim=True,
            ),
            PhaseConf(
                fwd_inputs={"input_ids": IO("stream", "words"), "seed": IO("seed", "seed")},
                bwd_inputs={0: IO("buffer", "dx", 0, RTS_ACT)},
                fwd_outputs={0: IO("buffer", "x", 0, RTS_ACT)},
                rows=1,
            ),
        ),
        GPTDecoderBlockTP2D: LayerConf(
            GraphConf(
                accumulate="*",
                grads_required=["x"],
                remote_buffer_fwd=True,
                remote_buffer_bwd=True,
                remote_buffer_optim=True,
            ),
            PhaseConf(
                fwd_inputs={"x": IO("buffer", "x", 0, RTS_ACT), "seed": IO("seed", "seed")},
                bwd_inputs={0: IO("buffer", "dx", 1, RTS_ACT)},
                fwd_outputs={0: IO("buffer", "x", 1, RTS_ACT)},
                bwd_outputs={0: IO("buffer", "dx", 0, RTS_ACT)},
                rows=config.model.layers,
            ),
        ),
        GPTLMHeadLossTP2D: LayerConf(
            GraphConf(
                accumulate=["head.word_embedding", "head.ln_f.weight", "head.ln_f.bias"],
                grads_required=["x"],
                reuse=["head.word_embedding"],
                remote_buffer_fwd=True,
                remote_buffer_bwd=True,
                remote_buffer_optim=True,
            ),
            PhaseConf(
                fwd_inputs={"x": IO("buffer", "x", config.model.layers, RTS_ACT), "labels": IO("stream", "labels")},
                fwd_outputs={0: IO("stream", "loss"), 1: IO("buffer", "dx", config.model.layers, RTS_ACT)},
                fwd_only=True,
                rows=1,
            ),
        ),
    }


# --- CONFIG UTILS ---


@dataclass
class IO:
    type: Literal["stream", "seed", "buffer"]
    name: str

    # Only used for buffers
    row_offset: int = 0
    rts: bool = False


IODict = Dict[Union[str, int], IO]


@dataclass
class GraphConf:
    accumulate: ListOrGlob = dataclasses.field(default_factory=list)
    grads_required: ListOrGlob = dataclasses.field(default_factory=list)
    reuse: List[str] = dataclasses.field(default_factory=list)
    remote_buffer_fwd: bool = False
    remote_buffer_bwd: bool = False
    remote_buffer_optim: bool = False


@dataclass
class PhaseConf:
    fwd_inputs: IODict = dataclasses.field(default_factory=dict)
    bwd_inputs: IODict = dataclasses.field(default_factory=dict)
    fwd_outputs: IODict = dataclasses.field(default_factory=dict)
    bwd_outputs: IODict = dataclasses.field(default_factory=dict)
    rows: int = 1
    fwd_only: bool = False


@dataclass
class LayerConf:
    graph_config: GraphConf
    phase_config: PhaseConf


def filter(tensors: Union[NamedTensors, Iterable[popxl.Tensor]], filter: ListOrGlob):
    if filter == "*":
        return tensors
    elif isinstance(tensors, DotTree):
        return tensors.filter_keys(filter)
    else:
        filter = set(filter)
        return [t for t in tensors if t.name in filter]
