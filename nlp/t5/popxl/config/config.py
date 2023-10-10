# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Tuple, Optional
from dataclasses import dataclass
from utils.simple_parsing_tools import Config, Choice, flag, parse_args_with_config_file
import popxl


@dataclass
class ModelConfig(Config):
    """Change the size/setup of the T5 model"""

    layers: int = 24
    """The number of encoder and decoder blocks in
        the model (hf: num_layers)"""

    hidden_size: int = 4096
    """The hidden size of the decoder blocks (hf: d_model)."""

    d_ff: int = 10240
    """The hidden size of the ff blocks in each encoder and decoder."""

    sequence_length: int = 512
    """Number of tokens in a sample."""

    eval: bool = flag(False)
    """If eval mode is enabled the model will be built for inference or validation
    - disabling steps such as dropout and optimisation."""

    dropout_prob: float = 0.0
    """Dropout probability"""

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

    eps: float = 1e-5
    """Epsilon for the layer normalisation layers."""

    scale_ff: int = 1
    """Scale factor to apply in the feed forward modules and undo after,
    in order to prevent overflows when using float16."""

    @dataclass
    class Embedding(Config):
        """Configuration of T5 Embedding layers"""

        vocab_size: int = 32128
        """Number of entries in the word vocabulary"""

    embedding: Embedding = Embedding()

    @dataclass
    class Attention(Config):
        """Configuration of T5 Attention layers"""

        heads: int = 64
        """Number of Attention Heads"""

        d_kv: int = 64
        """The size of the q, k, v projections for each head, in each encoder and decoder."""

        relative_attention_num_buckets: int = 32
        """Number of buckets for possible distances in relative positional encoding"""

        relative_attention_max_distance: Optional[int] = 128
        """Maximum distance represented in the relative positional encoding"""

    attention: Attention = Attention()


@dataclass
class Training(Config):
    """Changes the training setup."""

    steps: int = 1
    """The number of training steps."""

    global_batch_size: int = 32
    """The number of samples that contribute to an optimizer step."""

    stochastic_rounding: bool = flag(True)
    """See https://docs.graphcore.ai/projects/ai-float-white-paper/en/latest/ai-float.html#deterministic-versus-stochastic-rounding"""

    @dataclass
    class Optimizer(Config):
        """Changes the optimizer setup and hyperparameters."""

        name: str = "adamw"

        @dataclass
        class LearningRate(Config):
            """Changes the learning rate function."""

            maximum: float = 0.01
            """The maximum value that the learning rate will reach."""

            warmup_steps: int = 0
            """Number of training steps where the learning rate
            will increase linearly up to the maximum value."""

        learning_rate: LearningRate = LearningRate()

        beta1: float = 0.9
        """The beta1 coefficient of the Adam optimiser."""

        beta2: float = 0.999
        """The beta2 coefficient of the Adam optimiser."""

        weight_decay: float = 0.01
        """The weight decay factor (L2 penalty)."""

        gradient_clipping: float = 1.0
        """Clip gradients with values higher than this."""

    optimizer: Optimizer = Optimizer()


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
    """Number of tiles for each IPU that will be dedicated to I/O operations."""

    available_memory_proportion: Tuple[float, ...] = (0.28, 0.28, 0.28, 0.28)
    """Available memory proportion for each IPU."""

    loss_scaling: int = 1
    """The loss scaling to apply to gradient. This helps avoid underflowing
        in the gradient calculations."""

    tensor_parallel: int = 1
    """Number of ipus used for tensor model parallel axis"""

    attention_serialisation: int = 1
    """Number of serialisation steps when computing attention scores.
        Each step will be recomputed separately which can reduce the temporary
        activation requirement when using longer sequence lengths."""


@dataclass
class Checkpoint(Config):
    """Configuration of how to manage loading/storing of checkpoints"""

    load: Optional[str] = None
    """Load checkpoint from this location"""

    save: Optional[str] = None
    """Save checkpoint to this location"""

    steps: int = 0
    """Save a checkpoint every X steps. Disable with 0."""

    to_keep: int = 4
    """Maximum number of checkpoints to keep"""

    optim_state: bool = True
    """Whether to include the optimiser state in checkpoints"""


@dataclass
class Inference(Config):
    output_length: int = 5
    """Number of tokens to generate"""


@dataclass
class T5Config(Config):
    """Configuration of PopXL T5"""

    model: ModelConfig = ModelConfig()
    training: Training = Training()
    execution: Execution = Execution()
    checkpoint: Checkpoint = Checkpoint()
    inference: Inference = Inference()

    @property
    def gradient_accumulation(self):
        if self.model.eval:
            return 0
        denom = self.execution.data_parallel * self.execution.micro_batch_size
        if self.training.global_batch_size % denom != 0:
            raise RuntimeError(
                "Unable to set gradient accumulation to match the global batch size. "
                "global_batch_size % (data_parallel * micro_batch_size) != 0. "
                f"{self.training.global_batch_size} % "
                f"({self.execution.data_parallel} * {self.execution.micro_batch_size}) != 0"
            )

        return self.training.global_batch_size // denom

    @property
    def ipus(self):
        """Total number of IPUs required for execution"""
        DP = self.execution.data_parallel
        TP = self.execution.tensor_parallel
        return DP * TP

    def validate(self):
        if self.checkpoint.steps > 0:
            assert self.checkpoint.save, (
                "You need to specify a save path to save the checkpoint every X steps. "
                "Disable this error by setting `checkpoint.steps = 0`"
            )


if __name__ == "__main__":
    config = parse_args_with_config_file(T5Config)
    print(config.dumps_yaml())
