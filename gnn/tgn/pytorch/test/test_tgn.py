# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import poptorch
import torch
import torch_geometric as G
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parent.parent))

import tgn_modules as poptgn


class TransConvTestDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=10, n_nodes=7, channels_in=10, n_neighb=11, edge_dim=15):
        self.n_samples = n_samples
        self.seeds = np.random.randint(99999, size=n_samples)
        self.n_nodes = n_nodes
        self.channels_in = channels_in
        self.n_neighb = n_neighb
        self.edge_dim = edge_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        torch.manual_seed(self.seeds[idx])
        eg_n = torch.randn(self.n_nodes, self.channels_in)
        eg_idx = torch.randint(0, self.n_nodes, (2, self.n_neighb))
        eg_e = torch.randn(self.n_neighb, self.edge_dim)

        return eg_n, eg_idx, eg_e


class TGNMemoryTestDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=10):
        self.n_samples = n_samples
        self.seeds = np.random.randint(99999, size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        torch.manual_seed(self.seeds[idx])
        return batch.src, batch.dst, batch.t, batch.msg


def _assert_allclose(actual, expected, is_padded=False):
    if is_padded:
        assert len(actual.shape) == len(expected.shape)
        assert all(a >= b for a, b in zip(actual.shape, expected.shape))
        actual_unpadded = actual[tuple(slice(0, n) for n in expected.shape)]
    else:
        assert actual.shape == expected.shape
        actual_unpadded = actual
    np.testing.assert_allclose(actual_unpadded, expected, atol=1e-3)


def recursive_get(obj, obj_name):
    this_obj = obj
    for stub in obj_name.split('.'):
        this_obj = getattr(this_obj, stub)
    return this_obj


def copy_params(from_model, to_model):
    for name, param in from_model.named_parameters():
        splits = name.split(".")
        obj_name, param_name = ".".join(splits[:-1]), splits[-1]
        from_param = copy.deepcopy(getattr(eval("from_model." + obj_name), param_name))
        setattr(recursive_get(to_model, obj_name), param_name, from_param)
        print("Copied " + ".".join([obj_name, param_name]))


class TestTGN(unittest.TestCase):
    # TODO:
    # -softmax
    # -scatter_sum

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.opts = poptorch.Options()

    def test_time_encoder(self, run_on='IPU'):
        print("Testing TimeEncoder...")

        ref_model = G.nn.models.tgn.TimeEncoder(7)

        pop_model = poptgn.TimeEncoder(7, dtype=torch.float32)
        pop_model.lin.weight = ref_model.lin.weight
        pop_model.lin.bias = ref_model.lin.bias
        if run_on == 'IPU':
            print("Running on IPU")
            pop_model = poptorch.inferenceModel(pop_model.eval(), options=self.opts)
        else:
            print("Running on CPU")

        ds = torch.utils.data.TensorDataset(torch.rand(110).reshape(10, -1))
        popdl = poptorch.DataLoader(self.opts, ds, batch_size=1)

        for data in tqdm(popdl):
            ref_output = ref_model(data[0])
            pop_output = pop_model(data[0])

            _assert_allclose(ref_output.detach(), pop_output.detach())

        print("TimeEncoder test PASSED")

    def test_trans_conv(self,
                        n_samples=10,
                        n_nodes=7,
                        channels_in=10,
                        channels_out=20,
                        edge_dim=15,
                        n_neighb=11,
                        n_heads=2,
                        run_on='IPU',
                        ):

        print("Testing TransformerConv...")
        ref_model = G.nn.TransformerConv(channels_in, channels_out, edge_dim=edge_dim, heads=n_heads, dropout=0.0)

        pop_model = poptgn.TransformerConv(channels_in, channels_out, edge_dim, heads=n_heads, dropout=0.0)
        for layer in ['skip', 'edge', 'query', 'key', 'value']:
            assert getattr(pop_model, f"lin_{layer}").weight.shape == getattr(ref_model, f"lin_{layer}").weight.shape
            getattr(pop_model, f"lin_{layer}").weight = getattr(ref_model, f"lin_{layer}").weight
            getattr(pop_model, f"lin_{layer}").bias = getattr(ref_model, f"lin_{layer}").bias

        if run_on == 'IPU':
            print("Running on IPU")
            pop_model = poptorch.inferenceModel(pop_model.eval(), options=self.opts)
        else:
            print("Running on CPU")
        ds = TransConvTestDataset(n_samples, n_nodes, channels_in, n_neighb, edge_dim)
        popdl = poptorch.DataLoader(self.opts, ds, batch_size=1)

        for data in tqdm(popdl):
            inputs = [d[0] for d in data]
            ref_output = ref_model(*inputs)
            pop_output = pop_model(*inputs)
            _assert_allclose(ref_output.detach(), pop_output.detach())

        print("TransformerConv test PASSED")
