# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import poptorch

INDEX_PREFIX = '__'
SPLIT_SYMBOL = '.'


def lfind(string, symbol=SPLIT_SYMBOL):
    index = string[::-1].rfind(symbol)
    index = len(string) - index
    return index


def split_layer_path(layer_path, symbol=SPLIT_SYMBOL):
    index = lfind(layer_path, symbol)
    current_layer_name = layer_path[: index-1]
    child_layer_path = layer_path[index:]
    return current_layer_name, child_layer_path


class PipelineWrapper(torch.nn.Module):
    """
    wrapper class for poptorch.
    pass layername and ipu_id like (model.encoder.sub_layer__1, 1)
    This wrapper is built for pipelining PyTorch model with the annotate function poptorch.BeginBlock. \
    see[[ https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#annotations | PopTorch-annotations ]]

    It marks the layers in the model by parsing the layer's path we provide like model.encoder.layers__1 \
    and we should also provide an ipu_id for where the module is placed with the function set_start_point
    like obj.set_start_point(('model.encoder.layers__1', 1)).
    Or we can pass multiple annotation point with the function set_start_point_list like obj.set_start_point_list([('model.encoder.layer__1.layer', 1),\
     ('model.decoder.sub_layer', 2)])

    There are Module and ModuleList for building a PyTorch layer, so we use __{index} to represent a ModuleList and the index.
    """
    def __init__(self, model):
        super().__init__()
        self.wrapped = model

    def forward(self, feature, feature_length, target_in, target_out, target_length):
        return self.wrapped.forward(feature, feature_length, target_in, target_out, target_length)

    def __getattr__(self, attr):
        try:
            return torch.nn.Module.__getattr__(self, attr)
        except AttributeError:
            return getattr(self.wrapped, attr)

    def set_start_point_list(self, child_layer_path_list):
        for path, ipu_id in child_layer_path_list:
            self.set_start_point(path, ipu_id)

    def set_start_point(self, child_layer_path, ipu_id=0):
        self._nested_set_start_point(layer_pointer=self, child_layer_path=child_layer_path, ipu_id=ipu_id)

    def _nested_set_start_point(self, layer_pointer, child_layer_path, ipu_id):
        """
        split function for seting start point, find the current layer name and child layer path of the current layer through the split_layer_path\
         function, then find the layer whose last current_layer_index is empty through recursive call method, jump out of the loop, \
        and finally obtain the names and indexes of all sub layers

        Args:
            layer_pointer: model layer pointer instance
            child_layer_path(str):  stores subsequent strings
            ipu_id(int): ipu id
        """
        current_layer_name, child_layer_path = split_layer_path(layer_path=child_layer_path)

        if INDEX_PREFIX in current_layer_name:
            current_layer_index = int(current_layer_name.split(INDEX_PREFIX)[-1])
            current_layer_name = current_layer_name.split(INDEX_PREFIX)[0]
        else:
            current_layer_index = None

        if current_layer_index is not None:
            layer_pointer = layer_pointer.__getattr__(current_layer_name)
            sub_layer_pointer = layer_pointer[current_layer_index]
            if child_layer_path == "":
                layer_pointer[current_layer_index] = self.warp_start_point(sub_layer_pointer, ipu_id=ipu_id)
                return
            else:
                self._nested_set_start_point(sub_layer_pointer, child_layer_path, ipu_id=ipu_id)
        else:
            sub_layer_pointer = layer_pointer.__getattr__(current_layer_name)
            if child_layer_path == "":
                layer_pointer.__setattr__(current_layer_name, self.warp_start_point(sub_layer_pointer, ipu_id=ipu_id))
                return
            else:
                self._nested_set_start_point(sub_layer_pointer, child_layer_path, ipu_id=ipu_id)

    def warp_start_point(self, layer_pointer, ipu_id=0):
        layer_pointer = poptorch.BeginBlock(layer_pointer, ipu_id=ipu_id)
        return layer_pointer
