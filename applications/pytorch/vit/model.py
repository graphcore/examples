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

import poptorch
import torch.nn as nn
import transformers


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
        recomputed"""
    def recompute_outputs(module, inputs, outputs):
        return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
    module.register_forward_hook(recompute_outputs)


class PipelinedViTForImageClassification(transformers.ViTForImageClassification):
    """
    HuggingFace ViT for Image Classification model parallelized over multiple IPUs
    to run with replicated pipeline parallelism.
    """
    def __init__(self, config):
        super().__init__(config)

        print("---------- Device Allocation -----------")
        print("Embedding  --> IPU 0")
        self.vit.embeddings = poptorch.BeginBlock(self.vit.embeddings, "Embedding", ipu_id=0)

        layer_ipu = _get_layer_ipu(config.layers_per_ipu)
        for index, layer in enumerate(self.vit.encoder.layer):
            ipu = layer_ipu[index]
            if config.recompute_checkpoint_every_layer:
                recomputation_checkpoint(layer)
            self.vit.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Encoder {index:<2} --> IPU {ipu}")

        print("Head       --> IPU 3")
        self.vit.layernorm = poptorch.BeginBlock(self.vit.layernorm, "LayerNorm", ipu_id=3)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=3)

        print("---------------------------------------")

    def forward(self, pixel_values, labels=None):
        return super().forward(pixel_values=pixel_values, labels=labels, return_dict=False)
