# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2021 the HuggingDace team. All rights reserved.

from typing import Tuple, Optional, Union, Mapping

import torch
import numpy as np
from transformers.models.perceiver.modeling_perceiver import (_check_or_build_spatial_positions,
                                                              restructure,
                                                              PerceiverEmbeddings,
                                                              PerceiverBasicDecoder,
                                                              PerceiverClassifierOutput,
                                                              PerceiverMultimodalDecoder,
                                                              PerceiverMultimodalPreprocessor,
                                                              PerceiverMultimodalPostprocessor,
                                                              PerceiverFourierPositionEncoding,
                                                              PerceiverTrainablePositionEncoding,
                                                              PerceiverForMultimodalAutoencoding)

from optimum.graphcore.modeling_utils import register


PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]


register(PerceiverForMultimodalAutoencoding)


class IPUPerceiverForMultimodalAutoencoding(PerceiverForMultimodalAutoencoding):

    def __init__(self, config):
        super().__init__(config)
        self.perceiver.embeddings.__class__ = IPUPerceiverEmbeddings
        self.perceiver.decoder.__class__ = IPUPerceiverMultimodalDecoder
        self.perceiver.decoder.decoder__class__ = IPUPerceiverBasicDecoder
        self.perceiver.decoder.modalities['audio'].__class__ = IPUPerceiverBasicDecoder
        self.perceiver.decoder.modalities['image'].decoder.output_position_encodings.__class__ = IPUPerceiverFourierPositionEncoding
        self.perceiver.decoder.modalities['label'].decoder.output_position_encodings.__class__ = IPUPerceiverTrainablePositionEncoding
        self.perceiver.input_preprocessor.__class__ = IPUPerceiverMultimodalPreprocessor
        self.perceiver.input_preprocessor.modalities['image'].position_embeddings.__class__ = IPUPerceiverFourierPositionEncoding
        self.perceiver.input_preprocessor.modalities['audio'].position_embeddings.__class__ = IPUPerceiverFourierPositionEncoding
        self.perceiver.output_postprocessor.__class__ = IPUPerceiverMultimodalPostprocessor

    def forward(
            self,
            image: torch.Tensor,
            audio: torch.Tensor,
            label: torch.Tensor,
            image_subsampling: torch.tensor = None,
            audio_subsampling: torch.tensor = None,
            label_subsampling: torch.tensor = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: bool = False) -> Union[Tuple, PerceiverClassifierOutput]:

        inputs = {
            'image': image,
            'audio': audio,
            'label': label
        }

        subsampled_output_points = {
            'image': image_subsampling,
            'audio': audio_subsampling,
            'label': label_subsampling
        }

        outputs = self.perceiver(
            inputs=inputs,
            subsampled_output_points=subsampled_output_points,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        logits = outputs[0]

        # Training is not supported.
        loss = None
        if labels is not None:
            raise NotImplementedError("Multimodal autoencoding training is not yet supported")

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class IPUPerceiverMultimodalDecoder(PerceiverMultimodalDecoder):

    def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None):
        # Partition the flat inputs among the different modalities
        inputs = restructure(modality_sizes, inputs)

        # Obtain modality-specific decoders' queries
        subsampled_points = subsampled_points or dict()

        decoder_queries = dict()
        for modality, decoder in self.modalities.items():
            query = decoder.decoder_query(
                inputs=inputs[modality],
                modality_sizes=None,
                inputs_without_pos=None,
                subsampled_points=subsampled_points.get(modality, None),
            )
            decoder_queries[modality] = query

        # Pad all queries with trainable position encodings to make them have the same channels

        def embed(modality, x):
            x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
            pos = self.padding[modality]
            pos = pos.expand(x.shape[0], x.shape[1], self.num_query_channels - x.shape[2])
            return torch.cat([x, pos], dim=2)

        # Apply a predictable ordering to the modalities
        return torch.cat(
            [embed(modality, decoder_queries[modality]) for modality in sorted(self.modalities.keys())], dim=1
        )


class IPUPerceiverBasicDecoder(PerceiverBasicDecoder):

    @property
    def num_query_channels(self) -> int:
        if self.position_encoding_only:
            return self.position_encoding_kwargs.get(
                'project_pos_dim', self.output_position_encodings.output_size()
            )
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def decoder_query(self,
                      inputs,
                      modality_sizes=None,
                      inputs_without_pos=None,
                      subsampled_points=None):

        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")

        if subsampled_points is not None:
            # subsampled_points are the indices if the inputs would be flattened
            # however, the inputs aren't flattened, that's why we use unravel_index
            # to get the indices for the unflattened array
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            indices = list(
                torch.from_numpy(x) for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)
            )
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = pos.expand(batch_size, pos.shape[0], pos.shape[1])
            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, pos=pos
                )

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(index_dims, batch_size, device=inputs.device)

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = torch.cat([inputs_without_pos, pos_emb], div=-1)

        return pos_emb


class IPUPerceiverMultimodalPreprocessor(PerceiverMultimodalPreprocessor):

    def forward(self,
                inputs: Mapping[str, torch.Tensor],
                pos: Optional[torch.Tensor] = None,
                network_input_is_1d: bool = True) -> PreprocessorOutputType:

        padded = {}
        modality_sizes = {}
        inputs_without_pos = {}

        for modality, preprocessor in self.modalities.items():
            # preprocess each modality using the respective preprocessor.
            output, _, inputs_without_pos[modality] = preprocessor(
                inputs[modality], pos=pos, network_input_is_1d=network_input_is_1d
            )

            # pad to the same common_channel_size.
            batch_size, num_samples, num_channels = output.shape
            pos_enc = self.padding[modality].expand(
                batch_size,
                self.padding[modality].shape[0],
                self.padding[modality].shape[1]
            )

            padding = pos_enc.expand(
                batch_size,
                num_samples,
                self.num_channels - num_channels
            )
            output_padded = torch.cat([output, padding], dim=2)

            # mask if required
            if modality in self.mask_probs:
                mask_token = self.mask[modality].expand(
                    batch_size, self.mask[modality].shape[0], self.mask[modality].shape[1])
                mask_prob = self.mask_probs[modality]
                mask = torch.bernoulli(torch.full([batch_size, num_samples], mask_prob))
                mask = torch.unsqueeze(mask, dim=2).to(mask_token.device)
                output_padded = (1 - mask) * output_padded + mask * mask_token

            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape[1]

        # Apply a predictable ordering to the modalities
        padded_ls = [padded[k] for k in sorted(padded.keys())]

        # Finally, concatenate along the time dimension
        final_inputs = torch.cat(padded_ls, dim=1)

        return final_inputs, modality_sizes, inputs_without_pos


class IPUPerceiverMultimodalPostprocessor(PerceiverMultimodalPostprocessor):

    def forward(self,
                inputs: torch.Tensor,
                pos: Optional[torch.Tensor] = None,
                modality_sizes=None) -> tuple:

        if not self.input_is_dict:
            # Slice up modalities by their sizes.
            if modality_sizes is None:
                raise ValueError("Modality sizes should be specified if input is not a dictionary.")
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)

        outputs = []
        for modality, postprocessor in self.modalities.items():
            outputs.append(postprocessor(inputs[modality], pos=pos, modality_sizes=None))

        return outputs


class IPUPerceiverTrainablePositionEncoding(PerceiverTrainablePositionEncoding):

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings
        if batch_size is not None:
            position_embeddings = position_embeddings.expand(
                batch_size,
                position_embeddings.shape[0],
                position_embeddings.shape[1]
            )
        return position_embeddings


class IPUPerceiverEmbeddings(PerceiverEmbeddings):

    def forward(self, batch_size):
        return self.latents.expand(
            batch_size,
            self.latents.shape[0],
            self.latents.shape[1]
        )


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
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
            [pos, per_pos_features.expand(batch_size, per_pos_features.shape[0], per_pos_features.shape[1])], dim=-1
        )
    return per_pos_features


class IPUPerceiverFourierPositionEncoding(PerceiverFourierPositionEncoding):
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
