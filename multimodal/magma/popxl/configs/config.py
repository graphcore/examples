# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from utils.simple_parsing_tools import Config, Choice, flag, parse_args_with_config_file
import popxl

__all__ = ["ResNetConfig", "GPTJConfig", "MagmaConfig"]


@dataclass
class GPTJExecution(Config):
    """Changes the execution of the model."""

    micro_batch_size: int = 1
    """The number of samples that contribute to a
        gradient accumulation step."""

    available_memory_proportion: Tuple[float, ...] = (0.45, 0.45, 0.45, 0.45)

    tensor_parallel: int = 4
    """Number of ipus used for tensor model parallel axis"""

    attention_serialisation: int = 1
    """Number of serialisation steps when computing attention scores.
        Each step will be recomputed separately which can reduce the temporary
        activation requirement when using longer sequence lengths."""


@dataclass
class ResnetExecution(Config):
    micro_batch_size: int = 1
    """The number of samples that contribute to a
        gradient accumulation step."""
    available_memory_proportion: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)


@dataclass
class ResNetConfig(Config):
    """Change the size/setup of the Magma model"""

    layers: Tuple[int] = (6, 8, 18, 8)
    """The number of blocks in each layer. Corresponds to vision_layers in Clip/Magma"""

    width: int = 96
    """Base width for the blocks. Corresponding to vision_width in Clip/Magma"""

    image_resolution: int = 384
    """Input image resolution - size of the image"""

    execution: ResnetExecution = ResnetExecution()

    class Precision(Choice):
        float32 = popxl.float32
        float16 = popxl.float16

    precision: Precision = Precision.float16
    """Set the precision used for parameters in the model. Supported: float32, float16."""

    @property
    def embed_dim(self) -> int:
        """
        The ResNet feature dimension
        """
        return self.width * 32

    @property
    def heads(self) -> int:
        """
        Number of attention heads used in the AttentionPool layers. Corresponding to vision_heads in Clip/Magma
        """
        return self.embed_dim // 64

    @property
    def dtype(self) -> popxl.dtype:
        return self.precision.value

    @dtype.setter
    def dtype(self, value: popxl.dtype):
        self.precision = ResNetConfig.Precision(value)


@dataclass
class GPTJConfig(Config):
    """Configuration of PopXL GPT"""

    """Change the size/setup of the GPT model"""

    layers: int = 28
    """The number of decoder blocks in
        the model (hf: n_layer)"""

    hidden_size: int = 4096
    """The hidden size of the decoder blocks (hf: n_embd)."""

    sequence_length: int = 2048
    """Number of tokens in a sample."""

    class Precision(Choice):
        float32 = popxl.float32
        float16 = popxl.float16

    precision: Precision = Precision.float16
    """Set the precision used for parameters in the model. Supported: float32, float16."""

    @property
    def dtype(self) -> popxl.dtype:
        return self.precision.value

    @dtype.setter
    def dtype(self, value: popxl.dtype):
        self.precision = GPTJConfig.Precision(value)

    @dataclass
    class Embedding(Config):
        """Configuration of GPT Embedding layers"""

        vocab_size: int = 50400
        """Dimension of the language model head final projection.
        This is also the dimension of the embedding matrix (vocab size) size in gpt-j official HF implementation."""
        real_vocab_size: int = 50258
        """Number of tokens in GPT2/3 tokenizer.
        This is also the dimension of the embedding matrix (vocab size) in magma gpt-j language model."""

    embedding: Embedding = Embedding()

    @dataclass
    class Attention(Config):
        """Configuration of GPT Attention layers"""

        heads: int = 16
        """Number of Attention Heads"""

        rotary_positional_embeddings_base: int = 10000
        """Rotary positional embeddings base"""

        rotary_dim: Optional[int] = 64
        """Number of dimensions that rotary positional embedding is applied to"""

    attention: Attention = Attention()

    execution: GPTJExecution = GPTJExecution()

    @dataclass
    class Adapter(Config):
        """Configuration of MAGMA GPT-J Adapters"""

        layer_norm: bool = flag(False)
        downsample_factor: int = 8
        mode: Optional[str] = None

    ff_adapter: Adapter = Adapter()


@dataclass
class MagmaConfig(Config):
    """Configuration of PopXL MAGMA"""

    seed: int = 42
    """The random seed used by the model and host-side data generation (numpy and pytorch)."""

    visual: ResNetConfig = ResNetConfig()
    transformer: GPTJConfig = GPTJConfig()

    @property
    def ipus(self):
        """Total number of IPUs required for execution"""
        return self.transformer.execution.tensor_parallel

    def validate(self):
        assert self.visual.execution.micro_batch_size == self.transformer.execution.micro_batch_size
        assert (
            self.transformer.ff_adapter.mode == "normal"
        ), f"Expected feed forward adapters mode to be normal, found {self.transformer.ff_adapter.mode}"


if __name__ == "__main__":
    config = parse_args_with_config_file(MagmaConfig)
    print(config.dumps_yaml())
