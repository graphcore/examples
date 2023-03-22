# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import Tuple, Optional

import popxl
from utils.simple_parsing_tools import Choice, Config, flag, parse_args_with_config_file


@dataclass
class ModelConfig(Config):
    """Change the size/setup of the Bloom model"""

    layers: int = 70
    """The number of decoder blocks in
        the model (hf: n_layer)"""

    hidden_size: int = 14_336
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
        self.precision = ModelConfig.Precision(value)

    seed: int = 42
    """The random seed used by the model and host-side data generation (numpy and pytorch)."""

    @dataclass
    class Embedding(Config):
        """Configuration of Bloom Embedding layers"""

        vocab_size: int = 250_880
        """Number of entries in the word vocabulary"""

    embedding: Embedding = Embedding()

    @dataclass
    class Attention(Config):
        """Configuration of Bloom Attention layers"""

        heads: int = 112
        """Number of Attention Heads"""

    attention: Attention = Attention()


@dataclass
class Execution(Config):
    """Changes the execution of the model."""

    io_tiles: int = 1

    available_memory_proportion: Tuple[float, ...] = (0.28, 0.28, 0.28, 0.28)

    tensor_parallel_1: int = 1
    """Number of IPUs used for the first tensor model parallel axis. This is the outermost axis."""

    tensor_parallel_2: int = 1
    """Number of IPUs used for the second tensor model parallel axis. This is the innermost axis."""

    disable_fc_pass: bool = flag(False)

    memmap_dir: Optional[str] = None
    """Directory to store memmap tensor data. This helps avoid running out of host memory."""


@dataclass
class BloomConfig(Config):
    """Configuration of PopXL BLOOM"""

    model: ModelConfig = ModelConfig()
    execution: Execution = Execution()

    @property
    def ipus(self):
        """Total number of IPUs required for execution"""
        TP1 = self.execution.tensor_parallel_1
        TP2 = self.execution.tensor_parallel_2
        return TP1 * TP2

    def validate(self):
        assert (
            self.model.hidden_size % self.model.attention.heads == 0
        ), "Hidden size should be a multiple of attention heads"


if __name__ == "__main__":
    config = parse_args_with_config_file(BloomConfig)
    print(config.dumps_yaml())
