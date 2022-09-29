# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import poptorch
from typing import Optional

from optimum.graphcore.modeling_utils import (register,
                                              recomputation_checkpoint,
                                              PipelineMixin)

from optimum.utils import logging
from transformers.models.perceiver.modeling_perceiver import (space_to_depth,
                                                              PerceiverEncoder,
                                                              PerceiverEmbeddings,
                                                              PerceiverImagePreprocessor,
                                                              PerceiverTrainablePositionEncoding)


logger = logging.get_logger(__name__)


def register_subclass(huggingface_cls, recomputation: bool = False):

    @register(huggingface_cls)
    class PipelinedPerceiver(
        huggingface_cls,
        PipelineMixin
    ):

        def parallelize(self):
            super().parallelize()
            self.perceiver.embeddings.__class__ = WorkaroundPerceiverEmbeddings
            self.perceiver.input_preprocessor.__class__ = WorkaroundPerceiverImagePreprocessor
            self.perceiver.decoder.decoder.output_position_encodings.__class__ = WorkaroundPerceiverTrainablePositionEncoding

            if self.ipu_config.ipus_per_replica > 1:
                print('[pipelining] perceiver.input_preprocessor --> IPU0')
                self.perceiver.input_preprocessor = poptorch.BeginBlock(
                    self.perceiver.input_preprocessor,
                    'preprocessor',
                    ipu_id=0,
                )
                print('[pipelining] perceiver.embeddings --> IPU0')
                self.perceiver.embeddings = poptorch.BeginBlock(
                    self.perceiver.embeddings,
                    'embedding',
                    ipu_id=0,
                )

                ipu_ids = []
                for i, num_layers in enumerate(self.ipu_config.layers_per_ipu):
                    ipu_ids += [i] * num_layers
                self.perceiver.encoder.__class__ = setup_encoder_for_pipelining(ipu_ids=ipu_ids)

                print('[pipelining] perceiver.decoder --> IPU0')
                self.perceiver.decoder = poptorch.BeginBlock(
                    self.perceiver.decoder,
                    'decoder',
                    ipu_id=0,
                )

            for idx, self_attend in enumerate(self.perceiver.encoder.self_attends):
                if recomputation:
                    print(f'[recomputation] perceiver.encoder.self_attends[{idx}]')
                    self._hooks.append(recomputation_checkpoint(self_attend))

            return self

        def forward(self, pixel_values, labels=None):
            return super().forward(pixel_values=pixel_values, labels=labels, return_dict=False)


def setup_encoder_for_pipelining(ipu_ids):

    class WorkaroundPerceiverEncoder(PerceiverEncoder):
        """The Perceiver Encoder: a scalable, fully attentional encoder."""

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            inputs=None,
            inputs_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        ):
            print('[pipelining] perceiver.encoder.cross_attention --> IPU0')
            poptorch.Block.start(
                user_id='cross_attend',
                ipu_id=0
            )
            # Apply the cross-attention between the latents (hidden_states) and inputs:
            layer_outputs = self.cross_attention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=None,
                inputs=inputs,
                inputs_mask=inputs_mask,
                output_attentions=False
            )
            hidden_states = layer_outputs[0]

            # Apply the block of self-attention layers more than once:
            for block_i in range(self.config.num_blocks):
                for i, layer_module in enumerate(self.self_attends):
                    print(f'[pipelining] perceiver.encoder.self_attends[{i}] '
                          f'from block[{block_i}] --> IPU{ipu_ids[i]}')
                    poptorch.Block.start(
                        user_id=f'self_attend{block_i}{ipu_ids[i]}',
                        ipu_id=ipu_ids[i]
                    )

                    layer_head_mask = head_mask[i] if head_mask is not None else None
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=layer_head_mask,
                        output_attentions=False
                    )

                    hidden_states = layer_outputs[0]

            return tuple(v for v in [hidden_states, None, None, None] if v is not None)

    return WorkaroundPerceiverEncoder


class WorkaroundPerceiverImagePreprocessor(PerceiverImagePreprocessor):

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        if self.prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)

        elif self.prep_type == "pixels":
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :, :: self.temporal_downsample, :, :: self.spatial_downsample, :: self.spatial_downsample
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        elif self.prep_type == "patches":
            # Space2depth featurization.
            # Video: B x T x C x H x W
            inputs = space_to_depth(
                inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample
            )

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # for flow
                inputs = inputs.squeeze(dim=1)

            # Optionally apply conv layer.
            inputs = self.conv_after_patches(inputs)

        if self.prep_type != "patches":
            # move channels to last dimension, as the _build_network_inputs method below expects this
            if inputs.ndim == 4:
                inputs = torch.permute(inputs, (0, 2, 3, 1))
            elif inputs.ndim == 5:
                inputs = torch.permute(inputs, (0, 1, 3, 4, 2))
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        return inputs, modality_sizes, inputs_without_pos


class WorkaroundPerceiverEmbeddings(PerceiverEmbeddings):
    """Construct the latent embeddings."""

    def forward(self, batch_size):
        return self.latents.expand(batch_size, self.latents.shape[0], self.latents.shape[1])


class WorkaroundPerceiverTrainablePositionEncoding(PerceiverTrainablePositionEncoding):
    """Trainable position encoding."""

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings
        if batch_size is not None:
            position_embeddings = position_embeddings.expand(
                batch_size, position_embeddings.shape[0], position_embeddings.shape[1]
            )
        return position_embeddings
