# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
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

"""Utility to map weights from default model definition to the phased execution model"""
import numpy as np
from typing import Mapping, Callable
from functools import partial

__all__ = [
    "default_to_phased_mapping",
    "phased_to_default_mapping",
    "default_to_phased_transform",
    "add_phased_from_default_initializers"
]

WEIGHT_MAPPING = {}

# Embedding layers
WEIGHT_MAPPING['Embedding/Embedding_Dict'] = "BertModel/Encoder/Embeddings/Token/weight"
WEIGHT_MAPPING['Embedding/Segment_Dict'] = "BertModel/Encoder/Embeddings/Segment/weight"
WEIGHT_MAPPING['Embedding/Positional_Dict'] = "BertModel/Encoder/Embeddings/Position/weight"
WEIGHT_MAPPING['Embedding/Gamma'] = "BertModel/Encoder/Embeddings/Norm/Gamma"
WEIGHT_MAPPING['Embedding/Beta'] = "BertModel/Encoder/Embeddings/Norm/Beta"


def layers_mapping(N):
    ''' Returns a mapping default -> phased weights for N transformer layers'''
    mapping = {}
    for i in range(N):
        # Attention layer
        mapping[f'Layer{i}/Attention/QKV'] = f'BertModel/Encoder/Layer{i}/Attention/QKV'
        mapping[f'Layer{i}/Attention/Out'] = f'BertModel/Encoder/Layer{i}/Attention/Out'
        mapping[f'Layer{i}/Attention/Gamma'] = f'BertModel/Encoder/Layer{i}/Attention/Norm/Gamma'
        mapping[f'Layer{i}/Attention/Beta'] = f'BertModel/Encoder/Layer{i}/Attention/Norm/Beta'

        # Feedforward layer
        mapping[f'Layer{i}/FF/1/W'] = f'BertModel/Encoder/Layer{i}/FF/1/Dense/Weight'
        mapping[f'Layer{i}/FF/1/B'] = f'BertModel/Encoder/Layer{i}/FF/1/Dense/Bias'
        mapping[f'Layer{i}/FF/2/W'] = f'BertModel/Encoder/Layer{i}/FF/2/Dense/Weight'
        mapping[f'Layer{i}/FF/2/B'] = f'BertModel/Encoder/Layer{i}/FF/2/Dense/Bias'
        mapping[f'Layer{i}/FF/Gamma'] = f'BertModel/Encoder/Layer{i}/FF/Norm/Gamma'
        mapping[f'Layer{i}/FF/Beta'] = f'BertModel/Encoder/Layer{i}/FF/Norm/Beta'
    return mapping


# MaskLM
WEIGHT_MAPPING['CLS/LMPredictionW'] = "BertModel/MLM/LMPrediction/Dense/Weight"
WEIGHT_MAPPING['CLS/LMPredictionB'] = "BertModel/MLM/LMPrediction/Dense/Bias"
WEIGHT_MAPPING['CLS/Gamma'] = "BertModel/MLM/LMPrediction/Norm/Gamma"
WEIGHT_MAPPING['CLS/Beta'] = "BertModel/MLM/LMPrediction/Norm/Beta"

# NSP
WEIGHT_MAPPING['NSP/PoolW'] = "BertModel/NSP/Pool/Dense/Weight"
WEIGHT_MAPPING['NSP/PoolB'] = "BertModel/NSP/Pool/Dense/Bias"
WEIGHT_MAPPING['NSP/NspW'] = "BertModel/NSP/Classifier/Dense/Weight"
WEIGHT_MAPPING['NSP/NspB'] = "BertModel/NSP/Classifier/Dense/Bias"

# SQUAD
WEIGHT_MAPPING['Squad/SquadW'] = 'BertModel/Squad/Dense/Weight'
WEIGHT_MAPPING['Squad/SquadB'] = 'BertModel/Squad/Dense/Bias'

# MaskLM is on a different scope when serialising the embedding
SPLIT_EMBEDDING_MAPPING = {}
SPLIT_EMBEDDING_MAPPING['CLS/LMPredictionW'] = "BertModel/MLMSerialised/Slice/LMPrediction/Dense/Weight"
SPLIT_EMBEDDING_MAPPING['CLS/LMPredictionB'] = "BertModel/MLMSerialised/Slice/LMPrediction/Dense/Bias"
SPLIT_EMBEDDING_MAPPING['CLS/Gamma'] = "BertModel/MLMSerialised/Slice/LMPrediction/Norm/Gamma"
SPLIT_EMBEDDING_MAPPING['CLS/Beta'] = "BertModel/MLMSerialised/Slice/LMPrediction/Norm/Beta"


def split_embedding_mapping(N):
    '''Returns a mapping phased -> default weights for N split embedding'''
    mapping = {}
    for i in range(N):
        mapping[f'BertModel/Encoder/Embeddings/Token/split{i}/weight'] = "Embedding/Embedding_Dict"
    return mapping


def default_to_phased_mapping(args) -> Mapping[str, str]:
    '''Returns weight name mapping of default -> phased.
        Does not include split weights.'''
    mapping = {**WEIGHT_MAPPING}
    mapping.update(**layers_mapping(args.num_layers))

    # Handle serialisation of layers.
    if args.embedding_serialization_vocab_steps > 1:
        mapping.update(**SPLIT_EMBEDDING_MAPPING)

    return mapping


def phased_to_default_mapping(args) -> Mapping[str, str]:
    '''Returns weight name mapping of phased -> default.
        Including split weights.'''
    mapping = {v: k for k, v in default_to_phased_mapping(args).items()}

    # Handle serialisation of layers
    if args.embedding_serialization_vocab_steps > 1:
        mapping.update(**split_embedding_mapping(args.embedding_serialization_vocab_steps))

    return mapping


def phased_from_default_transform(args) -> Mapping[str, Callable[[np.ndarray], np.ndarray]]:
    '''Returns a mapping of phased -> fn.
        where fn takes the default numpy weight and returns the phased numpy weight.'''
    transform = {}

    # Handle serialisation of layers
    vocab_splits = args.embedding_serialization_vocab_steps
    if vocab_splits > 1:
        def get_split(idx, full_t):
            vocab_axis = full_t.shape.index(args.vocab_length)
            return np.split(full_t, vocab_splits, axis=vocab_axis)[idx]
        for i in range(vocab_splits):
            transform[f'BertModel/Encoder/Embeddings/Token/split{i}/weight'] = partial(get_split, i)

    return transform


def get_phased_initializers_from_default(args, initializers: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    '''Returns an initializer mapping for phased execution from a mapping for default execution.
        This will add splits weights as specfied by args.'''
    phased_initializers = {}
    mapping = phased_to_default_mapping(args)
    transform = phased_from_default_transform(args)
    for phased, default in mapping.items():
        if default in initializers.keys():
            weight = initializers[default]
            if phased in transform.keys():
                weight = transform[phased](weight)
            phased_initializers[phased] = weight
    return phased_initializers


def get_default_initializers_from_phased(initializers: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    '''Returns a dict with mappings for the default execution mode.
        This will concat any split weights detected in initializers.

        This is intended help to go from any phased init to any other phased init via the default mapping.

        old_phased_init = onnx.load(..)
        default_init = get_default_initializers_from_phased(old_phased_init)
        new_phased_init = add_phased_from_default_initializers(args, default_init)'''
    raise NotImplementedError()
