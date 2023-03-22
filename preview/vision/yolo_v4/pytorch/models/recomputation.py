# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import poptorch
import torch
import torch.nn as nn


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            return poptorch.recomputationCheckpoint(outputs)
        elif isinstance(outputs, tuple):
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    module.register_forward_hook(recompute_outputs)
