# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from dataclasses import dataclass
import torch


@dataclass
class AnchorBoxes:
    widths: torch.Tensor
    heights: torch.Tensor

    def __post_init__(self):
        if len(self.widths) != len(self.heights):
            raise ValueError(
                f"Length of 'widths' and 'heights' must be equal, got {len(self.widths)} and {len(self.heights)}."
            )

    def to_torch_tensor(self, n_dim=2):
        if n_dim == 1:
            return torch.stack((self.widths, self.heights), dim=0).view(2, 4).t().contiguous().view(2, 4)
        elif n_dim == 2:
            return torch.stack((self.widths, self.heights), axis=0)
        raise ValueError(f"Wrong number of anchors dimensions n_dim={n_dim}, supported n_dim=[1, 2].")

    def __len__(self):
        return len(self.widths)
