# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import to_fixed_size


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        u = torch.log1p(torch.exp(-x.abs()))
        v = torch.clamp_min(x, 0.0)
        return u + v - self.shift

    @staticmethod
    def replace_activation(module: torch.nn.Module):
        import torch_geometric.nn.models.schnet as pyg_schnet

        for name, child in module.named_children():
            if isinstance(child, pyg_schnet.ShiftedSoftplus):
                setattr(module, name, ShiftedSoftplus())
            else:
                ShiftedSoftplus.replace_activation(child)


class TrainingModule(torch.nn.Module):
    """
    TrainingModule for SchNet.  Assumes that each mini-batch contains a single
    padding molecule at the end and uses this to calculate the mean squared
    error (MSE) for the real molecules in each mini-batch.
    """

    def __init__(self, module, batch_size, replace_softplus=True):
        super().__init__()
        if replace_softplus:
            ShiftedSoftplus.replace_activation(module)

        self.model = to_fixed_size(module=module, batch_size=batch_size)

    def forward(self, z, pos, batch, target):
        prediction = self.model(z, pos, batch).view(-1)

        # slice off the padding molecule and calculate the mse loss
        prediction = prediction[0:-1]
        target = target[0:-1]
        loss = F.mse_loss(prediction, target)
        return prediction, loss


class KNNInteractionGraph(torch.nn.Module):
    def __init__(self, k: int, cutoff: float = 10.0):
        super().__init__()
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        """
        k-nearest neighbors without dynamic tensor shapes

        :param pos (Tensor): Coordinates of each atom with shape
            [num_atoms, 3].
        :param batch (LongTensor): Batch indices assigning each atom to
                a separate molecule with shape [num_atoms]

        This method calculates the full num_atoms x num_atoms pairwise distance
        matrix. Masking is used to remove:
            * self-interaction (the diagonal elements)
            * cross-terms (atoms interacting with atoms in different molecules)
            * atoms that are beyond the cutoff distance

        Finally topk is used to find the k-nearest neighbors and construct the
        edge_index and edge_weight.
        """
        pdist = F.pairwise_distance(pos[:, None], pos, eps=0)
        rows = arange_like(batch.shape[0], batch).view(-1, 1)
        cols = rows.view(1, -1)
        diag = rows == cols
        cross = batch.view(-1, 1) != batch.view(1, -1)
        outer = pdist > self.cutoff
        mask = diag | cross | outer
        pdist = pdist.masked_fill(mask, self.cutoff)
        edge_weight, indices = torch.topk(-pdist, k=self.k)
        rows = rows.expand_as(indices)
        edge_index = torch.vstack([indices.flatten(), rows.flatten()])
        return edge_index, -edge_weight.flatten()


def arange_like(n: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.arange(n, device=ref.device, dtype=ref.dtype)


def optimize_popart(options):
    """Apply a number of additional PopART options to optimize performance"""
    options._Popart.set("defaultBufferingDepth", 4)
    options._Popart.set("accumulateOuterFragmentSettings.schedule", 2)
    options._Popart.set("replicatedCollectivesSettings.prepareScheduleForMergingCollectives", True)
    options._Popart.set("replicatedCollectivesSettings.mergeAllReduceCollectives", True)
    return options


def prepare_data(data, target=4):
    """
    Prepares QM9 molecules for training SchNet for HOMO-LUMO gap prediction
    task.  Outputs a data object with attributes:
        z: the atomic number as a vector of integers with length [num_atoms]
        pos: the atomic position as a [num_atoms, 3] tensor of float32 values.
        y: the training target value. By default this will be the HOMO-LUMO gap
        energy in electronvolts (eV).
    """
    return Data(z=data.z, pos=data.pos, y=data.y[0, target].view(-1))
