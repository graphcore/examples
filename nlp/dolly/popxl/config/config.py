# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import List, Tuple, Optional
from dataclasses import dataclass
from utils.simple_parsing_tools import Config, Choice, flag, parse_args_with_config_file
import popxl


@dataclass
class ModelConfig(Config):
    """Change the size/setup of the GPT model"""

    layers: int = 36
    """The number of decoder blocks in
        the model (hf: n_layer)"""

    hidden_size: int = 5120
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
        """Configuration of GPT Embedding layers"""

        vocab_size: int = 50280
        """Number of entries in the word vocabulary"""

    embedding: Embedding = Embedding()

    @dataclass
    class Attention(Config):
        """Configuration of GPT Attention layers"""

        heads: int = 40
        """Number of Attention Heads"""

        rotary_positional_embeddings_base: int = 10000
        """Rotary positional embeddings base"""

        rotary_dim: Optional[int] = 32
        """Number of dimensions that rotary positional embedding is applied to"""

    attention: Attention = Attention()


@dataclass
class Execution(Config):
    """Changes the execution of the model."""

    micro_batch_size: int = 1
    """The number of samples that contribute to a
        gradient accumulation step."""

    data_parallel: int = 1
    """Set the number of model replicas to use for data-parallelism."""

    device_iterations: int = 1
    """Number of times the training loop is executed before relinquishing control and reporting to the host """

    io_tiles: int = 1

    available_memory_proportion: Tuple[float, ...] = (0.28, 0.28, 0.28, 0.28)

    tensor_parallel: int = 1
    """Number of IPUs used for tensor model parallel axis"""

    attention_tensor_parallel: Optional[int] = None
    """Number of IPUs used for tensor model parallel axis in the attention layer. If `None`, will be set to be equal to `tensor_parallel"""

    code_load: bool = flag(False)
    """Store the code for each layer graph in remote memory"""


@dataclass
class DollyConfig(Config):
    """Configuration of PopXL GPT"""

    model: ModelConfig = ModelConfig()
    execution: Execution = Execution()

    @property
    def ipus(self):
        """Total number of IPUs required for execution"""
        DP = self.execution.data_parallel
        TP = self.execution.tensor_parallel
        return DP * TP

    def validate(self):
        assert (
            self.model.hidden_size % self.model.attention.heads == 0
        ), "Hidden size should be a multiple of attention heads"


if __name__ == "__main__":
    config = parse_args_with_config_file(DollyConfig)
    print(config.dumps_yaml())
