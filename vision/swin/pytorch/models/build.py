# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
from functools import partial
import torch.nn as nn
import poptorch
from .swin_transformer import SwinTransformer


def get_layer_ipu(pipeline):
    layer_ipu = []
    for ipu, n_layers in enumerate(pipeline):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def set_pipeline(swin, pipeline, mapping_for_21k):
    # set be True will add recompute in the middle layer's first block's
    # mlp.drop of every pipeline
    use_recompute = False
    layer_ipus = get_layer_ipu(pipeline)
    print(f'swin set pipeline : {pipeline}')
    index = 0
    recompute_layer = []
    accum_index = 0
    for layer_num in pipeline:
        recompute_layer.append(accum_index + layer_num // 2 - 1)
        accum_index += layer_num

    for i, layer in enumerate(swin.layers):
        for j, block in enumerate(layer.blocks):

            ipu_id = layer_ipus[index]

            if ipu_id == 6 and layer_ipus[index - 1] == 5 and mapping_for_21k:
                # to fit imagenet21k head into IPU memory, we need to map layers before norm2 onto IPU5
                swin.layers[i].blocks[j].norm2 = poptorch.BeginBlock(block.norm2, user_id=f'layer_{i}_block{j}', ipu_id=ipu_id)
            else:
                swin.layers[i].blocks[j] = poptorch.BeginBlock(block, user_id=f'layer_{i}_block{j}', ipu_id=ipu_id)

            if use_recompute:
                if index in recompute_layer:
                    recomputation_checkpoint(block.mlp.drop)

            index += 1
    if mapping_for_21k:
        # special mapping for imagenet 21k
        swin.head = poptorch.BeginBlock(swin.head, user_id=f'layer_last', ipu_id=7)


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
        recomputed"""
    def recompute_outputs(module, inputs, outputs):
        return poptorch.recomputationCheckpoint(outputs)
    module.register_forward_hook(recompute_outputs)


def build_pipeline(config, train_loss_fn):
    if config.PRECISION != ['float', 'float'] and config.PRECISION != ['half', 'half']:
        print("Only support half_half and float_float!")
        exit()
    model = SwinTransformer(
        img_size=config.DATA.IMG_SIZE,
        num_classes=config.MODEL.NUM_CLASSES,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=config.MODEL.SWIN.DEPTHS,
        num_heads=config.MODEL.SWIN.NUM_HEADS,
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        device=config.MODEL.DEVICE,
        train_loss_fn=train_loss_fn,
        use_half=True if config.PRECISION[1] == 'half' else False
    )

    set_pipeline(model, config.IPU.LAYERS_PER_IPU, config.IPU.MAPPING_FOR_21K)
    return model
