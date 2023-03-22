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
import torch.nn as nn
import poptorch
import transformers

from models.modules import VisionTransformer
from models.utils import get_layer_ipu, recomputation_checkpoint, weight_init


class PipelinedViTForImageClassification(transformers.ViTForImageClassification):
    """
    HuggingFace ViT for Image Classification model parallelized over multiple IPUs
    to run with replicated pipeline parallelism.
    """

    def parallelize(self):
        self._hooks = []
        self.vit.embeddings = poptorch.BeginBlock(self.vit.embeddings, "Embedding", ipu_id=0)

        layer_ipu = get_layer_ipu(self.config.layers_per_ipu)
        for index, layer in enumerate(self.vit.encoder.layer):
            if self.config.recompute_checkpoint_every_layer:
                # Put checkpoints on every encoder layer
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            ipu = layer_ipu[index]
            self.vit.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)

        self.vit.layernorm = poptorch.BeginBlock(self.vit.layernorm, "LayerNorm", ipu_id=3)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=3)
        return self

    def deparallelize(self):
        # Remove any hooks
        for h in self._hooks:
            h.remove()
        # Remove poptorch Blocks
        for m in self.modules():
            if m != self:
                poptorch.removeBlocks(m)
        return self

    def print_device_allocation(self):
        layer_ipu = get_layer_ipu(self.config.layers_per_ipu)
        print("---------- Device Allocation -----------")
        print("Embedding  --> IPU 0")
        for index in range(self.config.num_hidden_layers):
            ipu = layer_ipu[index]
            print(f"Encoder {index:<2} --> IPU {ipu}")
        print("Head       --> IPU 3")
        print("---------------------------------------")

    def forward(self, pixel_values, labels=None):
        return super().forward(pixel_values=pixel_values, labels=labels, return_dict=False)


class PipelinedViTForImageClassificationPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = VisionTransformer(
            config=config, num_labels=config.num_labels, representation_size=config.representation_size
        )
        self.model.apply(weight_init)
        if self.config.pretrain:
            nn.init.constant_(self.model.head.bias, -10)
            nn.init.normal_(self.model.embeddings.position_embeddings, std=0.02)
        layer_ipu = get_layer_ipu(config.layers_per_ipu)

        print("---------- Device Allocation -----------")
        print("Embedding  --> IPU 0")
        self.model.embeddings = poptorch.BeginBlock(self.model.embeddings, "Embedding", ipu_id=0)

        for index, layer in enumerate(self.model.encoder.layer):
            ipu = layer_ipu[index]
            if config.recompute_checkpoint_every_layer:
                recomputation_checkpoint(layer)
            print(f"Encoder {index:<2} --> IPU {ipu}")
            self.model.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{ipu}", ipu_id=ipu)

        print("Representation --> IPU 3")
        self.model.representation = poptorch.BeginBlock(self.model.representation, "Representation", ipu_id=3)

        print("Head     --> IPU 3")
        self.model.head = poptorch.BeginBlock(self.model.head, "Head", ipu_id=3)
        print("---------------------------------------")

    def forward(self, x, labels=None, lam=None):
        logits, acc = self.model(x, labels, lam)
        return logits, acc
