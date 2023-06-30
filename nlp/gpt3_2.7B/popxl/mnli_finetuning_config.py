# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Union, List, Dict, Iterable

import dataclasses
import popxl
from dataclasses import dataclass
from typing_extensions import Literal

from config import GPTConfig
from modelling.decoder import GPTDecoderBlockTP
from modelling.embedding import GPTEmbeddingsTP
from modelling.mnli import GPTMnliLossHead
from popxl_addons import NamedTensors
from popxl_addons.dot_tree import DotTree

from pretraining_config import LayerConf, GraphConf, PhaseConf, IO


# --- CONFIG ---

RTS_THRESHOLD = 0
RTS_ACTIVATIONS_THRESHOLD = 0
USE_IO_TILES = False
ListOrGlob = Union[List[str], Literal["*"]]


def gen_layer_config(config: GPTConfig):
    return {
        GPTEmbeddingsTP: LayerConf(
            GraphConf(
                accumulate=["word.weight", "positional.weight"],
                grads_provided="*",
                remote_buffer_fwd=True,
                remote_buffer_bwd=False,
                remote_buffer_optim=True,
            ),
            PhaseConf(
                fwd_inputs={"input_ids": IO("stream", "words"), "seed": IO("seed", "seed")},
                bwd_inputs={0: IO("buffer", "dx", 0, config.execution.rts_activations)},
                fwd_outputs={0: IO("buffer", "x", 0, config.execution.rts_activations)},
                rows=1,
            ),
        ),
        GPTDecoderBlockTP: LayerConf(
            GraphConf(
                accumulate="*",
                grads_provided="*",
                grads_required=["x"],
                remote_buffer_fwd=True,
                remote_buffer_bwd=True,
                remote_buffer_optim=True,
            ),
            PhaseConf(
                fwd_inputs={"x": IO("buffer", "x", 0, config.execution.rts_activations), "seed": IO("seed", "seed")},
                bwd_inputs={0: IO("buffer", "dx", 1, config.execution.rts_activations)},
                fwd_outputs={0: IO("buffer", "x", 1, config.execution.rts_activations)},
                bwd_outputs={0: IO("buffer", "dx", 0, config.execution.rts_activations)},
                rows=config.model.layers,
            ),
        ),
        GPTMnliLossHead: LayerConf(
            GraphConf(
                accumulate=["head.ln_f.weight", "head.ln_f.bias", "head.score.weight", "head.score.bias"],
                grads_provided=["loss_output"],
                grads_required=["x"],
                remote_buffer_fwd=True,
                remote_buffer_bwd=True,
                remote_buffer_optim=True,
            ),
            PhaseConf(
                fwd_inputs={
                    "x": IO("buffer", "x", config.model.layers, config.execution.rts_activations),
                    "unpadded_length": IO("stream", "unpadded_length"),
                    "labels": IO("stream", "labels"),
                },
                fwd_outputs={
                    0: IO("stream", "loss"),
                    1: IO("buffer", "dx", config.model.layers, config.execution.rts_activations),
                    2: IO("stream", "logits"),
                },
                fwd_only=True,
                rows=1,
            ),
        ),
    }
