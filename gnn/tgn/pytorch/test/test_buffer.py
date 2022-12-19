# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import unittest

import poptorch


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self._mem_size = (10, 512)
        self.register_buffer('_mem', torch.empty(*self._mem_size, dtype=torch.float))
        self._mem.data.fill_(0)

    def detach(self):
        self._mem.detach_()

    def forward(self, ind):
        val = self._mem[ind]
        return val.long()

    def update_mem(self):
        indices = torch.tensor([0, 1]).long()
        values = torch.tensor([[1]*self._mem_size[1], [1]*self._mem_size[1]]).float()
        print(indices.shape)
        print(values.shape)
        self._mem.index_put_(indices=(indices,), values=values)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.x = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.memory = Memory()

    def forward(self):
        arr = self.memory._mem[0]
        loss = self.x * torch.mean(arr.float())
        self.memory.update_mem()
        return poptorch.identity_loss(loss, "none")


class TestRegisterBuffer(unittest.TestCase):

    def test_compare_cpu_ipu(self):
        # Test on IPU
        model = Model()
        model.train()
        optimizer = poptorch.optim.SGD(model.parameters(), lr=0.0)
        ipu_model = poptorch.trainingModel(model, optimizer=optimizer)

        loss_ipu_first_pass = ipu_model()
        assert loss_ipu_first_pass == 0.0, f'Running on IPU, should be 0: {loss_ipu_first_pass}'

        loss_ipu_second_pass = ipu_model()
        assert loss_ipu_second_pass == 1.0, f'Running on IPU, should be 1: {loss_ipu_second_pass}'

        # Test on CPU
        model = Model()
        model.train()

        optimizer.zero_grad()
        loss_cpu_first_pass = model.forward()
        loss_cpu_first_pass.backward()
        optimizer.step()
        model.memory.detach()

        assert loss_cpu_first_pass == 0.0, f'Running on CPU, should be 0: {loss_cpu_first_pass}'
        print(f'Running on CPU, should be 0: {loss_cpu_first_pass}')

        optimizer.zero_grad()
        loss_cpu_second_pass = model.forward()
        loss_cpu_second_pass.backward()
        optimizer.step()
        model.memory.detach()
        assert loss_cpu_second_pass == 1.0, f'Running on CPU, should be 1: {loss_cpu_second_pass}'
