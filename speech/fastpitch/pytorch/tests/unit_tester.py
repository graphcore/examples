# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import torch
import numpy as np
import poptorch


class Data:
    def __init__(self, column, size, data_type, max_value=None):
        assert column in ['feature', 'token', 'length', 'mask']
        self.column = column
        self.size = size
        self.max_value = max_value
        type_mapping = {'float32': torch.float32, 'float16': torch.float16, 'bool': torch.bool, 'int': torch.int}
        self.data_type = type_mapping.get(data_type, None)
        assert self.data_type
        self.__get_tensor()

    def __get_float_feature(self):
        self.tensor = torch.randn(size=self.size).to(self.data_type)

    def __get_int_id(self):
        assert self.max_value
        self.tensor = torch.randint(low=0, high=self.max_value, size=self.size).to(self.data_type)

    def __get_sequence_length(self):
        assert len(self.size) == 1
        self.__get_int_id()

    def __get_mask(self):  # TODO buid mask by sequence length
        self.tensor = torch.randint(low=0, high=2, size=self.size).to(self.data_type)

    def __get_tensor(self):
        mapping = {
            'feature': self.__get_float_feature,
            'token': self.__get_int_id,
            'length': self.__get_sequence_length,
            'mask': self.__get_mask
        }
        function = mapping.get(self.column, None)
        assert function
        function()


class BatchGenerator:

    @classmethod
    def build_data(cls, proto):
        tensor = Data(**proto).tensor
        return tensor

    @classmethod
    def build_datas(cls, proto_list):
        datas = [cls.build_data(proto) for proto in proto_list]
        return tuple(datas)


class Handler:
    def __init__(self, batch, pytorch_model, poptorch_model=None, atol=1e-5, random_seed=1):
        self.batch = batch
        self.pytorch_model = pytorch_model
        self.poptorch_model = poptorch_model if poptorch_model else copy.deepcopy(pytorch_model)
        self.atol = atol
        self.random_seed = random_seed
        self.build_option()
        self.build_optimizer()
        self.prepare_poptorch_model()

    def build_option(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        # Poptorch options
        self.ipu_option = poptorch.Options()
        self.ipu_option.autoRoundNumIPUs(True)
        self.ipu_option.deviceIterations(1)
        self.ipu_option.replicationFactor(1)
        self.ipu_option.Training.gradientAccumulation(1)
        self.ipu_option.outputMode(poptorch.OutputMode.All)
        self.ipu_option.randomSeed(self.random_seed)

    def build_optimizer(self):
        self.pytorch_optimizer = poptorch.optim.SGD(self.pytorch_model.parameters(), lr=1e-3, use_combined_accum=True, momentum=0.9, weight_decay=0, loss_scaling=1.0)
        self.poptorch_optimizer = poptorch.optim.SGD(self.poptorch_model.parameters(), lr=1e-3, use_combined_accum=True, momentum=0.9, weight_decay=0, loss_scaling=1.0)

    def prepare_poptorch_model(self):
        for name, _ in self.poptorch_model.named_parameters():
            self.ipu_option.anchorTensor(f'grad_{name}', f'Gradient___{name}')
        self.poptorch_model = poptorch.trainingModel(self.poptorch_model, self.ipu_option, self.poptorch_optimizer)

    def run_one_step(self):
        poptorch_output = self.poptorch_model(self.batch)
        self.pytorch_optimizer.zero_grad()
        pytorch_output = self.pytorch_model(self.batch)
        pytorch_output.backward()
        self.pytorch_optimizer.step()
        print(f'poptorch_output: {poptorch_output}, pytorch_output: {pytorch_output}')
        for name, param in self.pytorch_model.named_parameters():
            poptorch_grad = self.poptorch_model.getAnchoredTensor(f'grad_{name}')
            pytorch_grad = param.grad
            # print(f'name: {name}, \n poptorch_grad: \n {poptorch_grad}, \n pytorch_grad: \n {pytorch_grad}\n')
            # print(poptorch_grad.detach().numpy() - pytorch_grad.detach().numpy())
            assert np.allclose(poptorch_grad.detach().numpy(), pytorch_grad.detach().numpy(), atol=self.atol)
            # print('----')


class Wrapper(torch.nn.Module):
    def __init__(self, model, loss_index):
        super().__init__()
        self.model = model
        self.loss_index = loss_index

    def forward(self, input):
        output = self.model(*input)
        return poptorch.identity_loss(output[self.loss_index].sum(), reduction='mean')


if __name__ == '__main__':

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(3, 4)
            self.linear2 = torch.nn.Linear(4, 5)
            self.dropout = torch.nn.Dropout(0.0)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.dropout(x)
            return x

    model = Model()
    model = Wrapper(model, 0)  # wrap model, add loss for trainingModel
    proto_list = [
        dict(column='feature', size=(2, 3), data_type='float32')
    ]  # proto info for the input data
    batch = BatchGenerator.build_datas(proto_list)
    handler = Handler(batch, model)
    handler.run_one_step()
