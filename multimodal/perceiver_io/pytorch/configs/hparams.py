# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from dataclasses import dataclass, field
from typing import Optional, List

from transformers import (PerceiverForImageClassificationFourier,
                          PerceiverForImageClassificationLearned,
                          PerceiverForImageClassificationConvProcessing)

from optimum.graphcore import IPUTrainingArguments


# dict of supported datasets
# with the data column name
SUPPORTED_DATASETS = {
    'cifar10': 'img',
    'imagenet-1k': 'image'
}


AVAILABLE_MODELS = {
    'deepmind/vision-perceiver-learned': PerceiverForImageClassificationLearned,
    'deepmind/vision-perceiver-fourier': PerceiverForImageClassificationFourier,
    'deepmind/vision-perceiver-conv': PerceiverForImageClassificationConvProcessing
}


@dataclass
class PerceiverTrainingArguments(IPUTrainingArguments):
    """
    Defines or overwrites configuration variables for training on the IPU.
    """

    report_to: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            'help': 'For now only "wandb" is an acceptable argument.'
        }
    )

    profile_dir: Optional[str] = field(
        default='',
        metadata={
            'help': 'If given, a profile will be generated and saved under the given path.'
        }
    )

    constant_cosine: bool = field(
        default=False,
        metadata={
            'help': 'If provided use learning rate schedule with cosine decay preceded with a constant warmup phase.'
        }
    )


@dataclass
class DatasetArguments:
    """
    Defines configuration variables for the dataset and its preprocessing pipeline.
    """

    dataset_name: str = field(
        default='imagenet-1k',
        metadata={'help': 'Name of the dataset to use. '
                  f'Supported datasets: {list(SUPPORTED_DATASETS.keys())}.'}
    )

    dataset_path: str = field(
        default='/localdata/datasets/imagenet_object_localization_patched2019.tar.gz',
        metadata={'help': 'The file/folder on the local machine with the dataset.'}
    )


@dataclass
class ModelArguments:
    """
    Defines configuration variables for the model architecture.
    """

    config: str = field(
        default=None,
        metadata={'help': 'Path to a .json configuration file.'}
    )

    model_name: str = field(
        default='deepmind/vision-perceiver-fourier',
        metadata={'help': 'Version of the perceiver to use. '
                  f'The options include: {list(AVAILABLE_MODELS.keys())}'}
    )

    num_latents: int = field(
        default=512,
        metadata={'help': 'The number of latents.'}
    )

    d_latents: int = field(
        default=1024,
        metadata={'help': 'Dimension of the latent embeddings.'}
    )

    num_blocks: int = field(
        default=8,
        metadata={'help': 'Number of blocks in the Transformer encoder.'}
    )

    num_self_attends_per_block: int = field(
        default=6,
        metadata={'help': 'The number of self-attention layers per block.'}
    )

    num_self_attention_heads: int = field(
        default=8,
        metadata={'help': 'Number of attention heads for each self-attention'
                          'layer in the Transformer encoder.'}
    )

    num_cross_attention_heads: int = field(
        default=1,
        metadata={'help': 'Number of attention heads for each cross-attention'
                          'layer in the Transformer encoder.'}
    )
