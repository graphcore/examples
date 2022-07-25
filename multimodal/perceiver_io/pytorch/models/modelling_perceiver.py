# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import numpy as np
from typing import Optional

from optimum.graphcore.modeling_utils import register, PipelineMixin

from optimum.utils import logging
from transformers.models.perceiver.modeling_perceiver import (space_to_depth,
                                                              _check_or_build_spatial_positions,
                                                              PerceiverEmbeddings,
                                                              PerceiverImagePreprocessor,
                                                              PerceiverFourierPositionEncoding,
                                                              PerceiverTrainablePositionEncoding)


logger = logging.get_logger(__name__)


def register_subclass(huggingface_cls):

    @register(huggingface_cls)
    class PipelinedPerceiver(
        huggingface_cls,
        PipelineMixin
    ):

        def parallelize(self):
            super().parallelize()
            self.perceiver.embeddings.__class__ = WorkaroundPerceiverEmbeddings
            self.perceiver.input_preprocessor.__class__ = WorkaroundPerceiverImagePreprocessor
            self.perceiver.input_preprocessor.position_embeddings.__class__ = WorkaroundPerceiverFourierPositionEncoding
            self.perceiver.decoder.decoder.output_position_encodings.__class__ = WorkaroundPerceiverTrainablePositionEncoding

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
            print('[pipelining] perceiver.encoder --> IPU1')
            self.perceiver.encoder = poptorch.BeginBlock(
                self.perceiver.encoder,
                'encoder',
                ipu_id=1
            )
            print(f'[pipelining] perceiver.decoder --> IPU0')
            self.perceiver.decoder = poptorch.BeginBlock(
                self.perceiver.decoder,
                'decoder',
                ipu_id=0,
            )

            return self

        def forward(self, pixel_values, labels=None):
            return super().forward(pixel_values=pixel_values, labels=labels, return_dict=False)


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


class WorkaroundPerceiverFourierPositionEncoding(PerceiverFourierPositionEncoding):

    def forward(self, index_dims, batch_size, device, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.

    Args:
      pos (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`):
        The Tensor containing the position of n points in d dimensional space.
      num_bands (`int`):
        The number of frequency bands (K) to use.
      max_resolution (`Tuple[int]`, *optional*, defaults to (224, 224)):
        The maximum resolution (i.e. the number of pixels per dim). A tuple representing resolution for each dimension.
      concat_pos (`bool`, *optional*, defaults to `True`):
        Whether to concatenate the input position encoding to the Fourier features.
      sine_only (`bool`, *optional*, defaults to `False`):
        Whether to use a single phase (sin) or two (sin/cos) for each frequency band.

    Returns:
      `torch.FloatTensor` of shape `(batch_size, sequence_length, n_channels)`: The Fourier position embeddings. If
      `concat_pos` is `True` and `sine_only` is `False`, output dimensions are ordered as: [dim_1, dim_2, ..., dim_d,
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, per_pos_features.shape[0], per_pos_features.shape[1])],
            dim=-1
        )
    return per_pos_features
