# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import json
import tensorflow.compat.v1 as tf

from ipu_sparse_ops.model_baseclass import SparseModelOptions
os.sys.path.append("../")  # dynamic_sparsity
from ipu_sparse_ops.transformer.transformer_baseclass import TransformerOptions   # noqa: E402


def get_program_options():
    # General transformer options
    parser = TransformerOptions()

    # Special options for sparse models
    SparseModelOptions.add_all_arguments(parser)

    # Additional options
    parser.add_argument("--extra-poplar-options-disable", action='store_true', help='Disable the setting of extra options for poplar')
    parser.add_argument("--extra-poplar-options-sync-enable", action='store_true', help='Enable the setting of extra sync options for poplar')
    parser.add_argument(
        "--extra-poplar-options-num-callback-threads", type=str, default='4',
        help="Change the number of threads used for stream callbacks. Set to 'auto' to let poplar choose the value. Set to '0' for single threaded.")
    parser.add_argument("--mode", choices=['train', 'test', 'all'], default="all", help="Choices are [training, test, all]")
    parser.add_argument("--nepochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--train-checkpoint-path", type=str, help="where should checkpoints go. Warning: non-pipelined "
                        "checkpoints are not compatible with the pipelined version and vice-versa.")
    parser.add_argument("--autoregression-offset", type=int, default=8,
                        help="Number of tokens at start of sequence to ignore in the autoregressive loss.")
    parser.add_argument("--repeat-count", type=int, default=50, help="Number batch serialization iterations")
    parser.add_argument("--recompute", action="store_true", help="Turns recomputation on")
    parser.add_argument("--sparse-embeddings", action="store_true", help="Enables sparse embeddings and projection."
                        "Currently only block size 1 is supported and will be set, regardless of the block size used for other layers.")
    parser.add_argument("--restore-epoch", type=int, default=None, help="In test mode, if specified, the checkpoint corresponding to the"
                        "specified epoch completion will be restore. Otherwise the latest will")

    # Optimizer options
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["GradientDescent", "Momentum", "Adam"],
                        help="Which optimizer to use.")
    parser.add_argument("--grad-norm-clip", type=float, default=None, help="Enables gradient clipping by the specified norm.")
    parser.add_argument("--loss-scale", type=float, default=1.0, help="Enables loss scaling before the gradient computation "
                        "followed by unscaling of the gradients. May help prevent under and over flow")
    parser.add_argument("--unscale-grad-pre-acc", action='store_true', help="If true when loss scaling is on, "
                        "the gradient are unscaled before they are accumulated, else it is done after accumulation, "
                        "right before applying them.")
    parser.add_argument("--grad-acculation-mode", type=str, choices=['Sum', 'Avg'], default='Sum', help="Changes the accumulation "
                        "type in the pipeline gradient accumulation optimizer.")
    parser.add_argument("--scale-grad-pre-acc", action='store_true', help="If true when gradient accumulation type is Avg, "
                        "the gradient are scaled before they are accumulated, else it is done after accumulation, "
                        "right before applying them.")
    parser.add_argument("--slots-fp-type", type=str, default=None, choices=['float32', 'float16'], help="If set, the slots for the "
                        "optimizer will use the selected type")
    parser.add_argument("--force-fp32-weight-update", action="store_true", help="When choosing the slots fp type independently "
                        "from the model, this forces the weight update computation to use fp32, no matter what the var and slots types are")

    # Learning rate schedule options
    parser.add_argument("--warmup-steps", type=int, default=100000, help="Linear warm-up steps for learning rate schedule.")
    parser.add_argument("--cooldown-steps", type=int, default=1000000, help="Linear warm-up steps for learning rate schedule.")
    parser.add_argument("--peak-learning-rate", type=float, default=1e-4, help="The peak learning rate to use.")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5, help="The min learning rate to use.")
    parser.add_argument("--decay-power", type=float, default=0.5, help="The power to use for the polynomial decay.")
    # Pipeline options
    parser.add_argument("--pipeline", action="store_true", help="Turns pipelining on for sparse_training")
    parser.add_argument("--gradient-accumulation-count", type=int, default=36, help="Sets number of micro-batches in each pipeline run")
    parser.add_argument(
        "--gradient-accumulation-dtype", choices=["float32", "float16"], type=tf.as_dtype, default=None,
        help="Overrides default dtype of gradient accumulation buffer")
    parser.add_argument(
        "--offload-activations", action='store_true', help="Offloads intermediate activations to remote buffers")
    parser.add_argument(
        "--offload-gradient-accumulation-buffers", action='store_true', help="Offloads gradient accumulation buffers to remote buffers")
    parser.add_argument(
        "--offload-weight-update-variables", action='store_true', help="Offloads weight update variables to remote buffers")
    # Data options
    parser.add_argument("--use-synthetic-data", action='store_true', help="Uses random synthetic data generated on the host")
    parser.add_argument("--shuffle", action='store_true', help="Shuffles the order in which dataset sequences are read")
    parser.add_argument("--disable-dataset-cache", action='store_true', help="Disable dataset caching")
    parser.add_argument("--disable-dataset-prefetch", action='store_true', help="Disable dataset prefetching")
    parser.add_argument("--data-dir", default=None, type=str, help="Path to the directory where the dataset is stored")
    # Logging options
    parser.add_argument("--log-level", type=str, default='INFO', choices=['NOTSET', 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument("--decode", action="store_true", help="Enable decoding sequneces to human readable text for debug purposes.")
    parser.add_argument("--log-histograms", action="store_true", help="Whether to log full histograms. "
                        "Std, mean, max and min will always be logged either way")
    parser.add_argument("--bins-count", type=int, default=100, help="Number of bins to use for the tensorboard histograms")
    parser.add_argument("--use-wandb", action="store_true", help="Exports results to Weights and Biases for experiments tracking")
    parser.add_argument("--wandb-project-name", type=str, default="dynsparse-language-model", help="The name of the wandb project")
    parser.add_argument("--wandb-tags", type=str, nargs='+', default=None,
                        help="Tags to use for the current run in wandb. Can be used in the dashboard for sorting runs.")
    parser.add_argument("--wandb-name", type=str, default=None, help="A name for this run which will be used in wandb.")
    parser.add_argument("--debug-dense-grad", action='store_true', help="Enable debug printing whenever the dense gradient is calculated.")

    # Compile options
    parser.add_argument("--compile-only", action='store_true', help='Compile without running or attaching to device.')
    parser.add_argument("--compile-only-ipu-version", choices=['ipu1', 'ipu2'], type=str, default=None,
                        help='If --compile-only is set this determines the IPU version to target.')
    parser.add_argument("--on-demand", action='store_true', help='Defer IPU attach until execution.')

    def parse_optimizer_arg(arg: str):
        name, value = arg.split('=')
        return (name, json.loads(value))

    parser.add_argument("--optimizer-arg", type=parse_optimizer_arg, action="append",
                        help="Extra argument for the chosen optimizer of the form argname=value. "
                        "Example: `use_nesterov=false`. "
                        "Can be input multiple times.")

    default_settings = dict(
        # Model parameters
        encoder_layers=2,
        dtype='float32',
        embedding_length=128,
        hidden_length=512,
        ff_length=512,
        attention_heads=16,
        qkv_length=32,
        # Sparse model parameters
        sparsity=0.9,
        block_size=8,
        prune_ratio=0.3,
        regrow_type='rigl',
        pooling_type="MAX",
        # Specify the parameters of the sequence data
        source_sequence_length=64,
        source_vocab_length=16384,  # a.k.a embedding/dictionary size
        source_pad_id=3,
        source_bos_id=0,
        source_eos_id=1,
        # Program config
        warmup_steps=100000,
        cooldown_steps=1000000,
        gradient_accumulation_count=24,  # pipeline only, overrides batches per io step
        gradient_accumulation_dtype=None,  # pipeline only, overrides dtype for accumulators
        autoregression_offset=16,  # do not compute loss on the first 16 tokens
        batch_size=1,
        nepochs=200,
        optimizer="Adam",
        peak_learning_rate=8e-5,
        min_learning_rate=8e-6,
        num_shards=2,
        log_level="INFO",
        train_checkpoint_path="checkpoints",
        mode="train"
    )

    parser.set_defaults(**default_settings)
    opts = parser.parse_args()

    return opts
