# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from email.policy import default
from random import choices
from typing import List, Optional

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import ClosedUnitInterval, PositiveInt


def get_parser():
    """Command line argument parser.

    Returns:
        jsonargparser: processed arguments into jsonargparse object
    """
    parser = ArgumentParser()

    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=1984)
    # Model inputs
    parser.add_argument(
        "--inputs",
        default=["node_feat", "edge_feat"],
        type=list,
        help="Inputs to model, must align with the features selected",
    )
    # Model hyper-parameters
    parser.add_argument(
        "--model.micro_batch_size",
        type=PositiveInt,
        default=8,
        help='Compute batch size (if using packing this is measured in "packs per batch")',
    )
    parser.add_argument("--model.valid_batch_size", type=PositiveInt, help="Batch size for use in validation")
    parser.add_argument(
        "--model.target_total_batch_size",
        type=PositiveInt,
        default=None,
        help="Try and match training batch size to this value.",
    )
    parser.add_argument("--model.n_nodes_per_pack", default=80, type=PositiveInt, help='nodes per "pack"')
    parser.add_argument("--model.n_edges_per_pack", default=160, type=PositiveInt, help='edges per "pack"')
    parser.add_argument(
        "--model.n_graphs_per_pack", default=16, type=PositiveInt, help='maximum number of graphs per "pack"'
    )
    parser.add_argument("--model.epochs", default=100, type=PositiveInt, help="Maximum number of epochs to run for")
    parser.add_argument("--model.lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--model.learning_rate_schedule",
        default="cosine",
        choices=["cosine", "linear", "static"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--model.cosine_lr", type=bool, default=False, help="use a cosine lr decay"
    )  # add this back to accommodate ensembling for previous runs
    parser.add_argument("--model.min_lr", default=0, type=float, help="minimum learning rate for the lr scheduler")
    parser.add_argument("--model.lr_warmup_epochs", default=0.0, type=float, help="Number of warmup epochs")
    parser.add_argument(
        "--model.lr_init_prop", default=1.0 / 8.0, type=float, help="Initial scale of lr when warming up"
    )
    parser.add_argument(
        "--model.loss_scaling",
        default=16,
        type=float,
        help="loss scaling factor (to keep gradients representable in IEEE FP16)",
    )

    parser.add_argument(
        "--model.node_latent", default=300, type=PositiveInt, help="number of latent units in the network"
    )
    parser.add_argument("--model.node_exp_ratio", default=2.0, type=float, help="Ratio between latent and hidden size.")
    parser.add_argument(
        "--model.node_mlp_layers",
        default=2,
        type=PositiveInt,
        help="total number of layers in the MLPs (including output)",
    )
    parser.add_argument("--model.node_dropout", default=0.0, type=float, help="dropout for nodes")
    parser.add_argument("--model.node_prenorm", default=False, type=bool, help="Add norm+act to start of MLP")

    parser.add_argument("--model.use_edges", default=True, type=bool, help="use edges features")
    parser.add_argument(
        "--model.edge_latent", default=None, type=PositiveInt, help="number of edge latent units in the network"
    )
    parser.add_argument(
        "--model.edge_exp_ratio", default=None, type=float, help="Ratio between latent and hidden size."
    )
    parser.add_argument(
        "--model.edge_mlp_layers",
        default=None,
        type=PositiveInt,
        help="total number of layers in the MLPs (including output)",
    )
    parser.add_argument("--model.edge_dropout", default=0.0, type=float, help="dropout for edges")
    parser.add_argument(
        "--model.edge_dna_dropout", default=0.0, type=float, help="dropout for direct neighbour aggregation of edges"
    )
    parser.add_argument(
        "--model.eigv_rand_sign_flip",
        default=True,
        type=bool,
        help="Add random sign flipping to laplacian eigen vectors.",
    )
    parser.add_argument("--model.edge_prenorm", type=bool, help="Add norm+act to start of MLP")
    parser.add_argument(
        "--model.edge_dropout_loc",
        default="before_residual_add",
        type=str,
        choices=["before_residual_add", "before_scatter"],
        help="Location for edge dropout",
    )
    parser.add_argument("--model.use_globals", default=False, type=bool, help="Use global features")
    parser.add_argument("--model.global_latent", default=None, type=PositiveInt, help="Number of global latents")
    parser.add_argument(
        "--model.global_exp_ratio", default=None, type=float, help="Ratio between latent and hidden size."
    )
    parser.add_argument(
        "--model.global_mlp_layers",
        default=None,
        type=PositiveInt,
        help="total number of layers in the MLPs (including output)",
    )
    parser.add_argument("--model.global_dropout", default=0.0, type=float, help="dropout for globals")
    parser.add_argument("--model.global_prenorm", type=bool, help="Add norm+act to start of MLP")

    parser.add_argument("--model.encoder_latent", default=None, type=PositiveInt, help="Number of global latents")
    parser.add_argument(
        "--model.encoder_exp_ratio", default=None, type=float, help="Ratio between latent and hidden size."
    )
    parser.add_argument(
        "--model.encoder_mlp_layers",
        default=None,
        type=PositiveInt,
        help="total number of layers in the MLPs (including output)",
    )
    parser.add_argument("--model.encoder_dropout", default=0.0, type=float, help="dropout for globals")
    parser.add_argument("--model.encoder_prenorm", type=bool, help="Add norm+act to start of MLP")
    parser.add_argument(
        "--model.encoder_norm_pos",
        choices=["none", "layer_hidden", "layer_output"],
        help="For the MLPs, whether and where to use normalization.",
    )
    parser.add_argument(
        "--model.encoder_act_fn",
        choices=["relu", "gelu", "swish"],
        help="Activation function used for the encoder MLPs.",
    )
    parser.add_argument(
        "--model.atom_encoder_model",
        default="node",
        choices=["node", "encoder"],
        help="Which model to use for the atom encoder",
    )
    parser.add_argument(
        "--model.bond_encoder_model",
        default="edge",
        choices=["edge", "encoder"],
        help="Which model to use for the bond encoder",
    )
    parser.add_argument(
        "--model.override_encoder_dropout",
        type=dict,
        default={},
        help="Dictionary to override dropout for specific encoder features",
    )
    parser.add_argument("--model.expand_first_hidden", default=True, type=bool, help="Expand first MLP hidden")
    parser.add_argument(
        "--model.encoder_concat_mode",
        type=str,
        default="concat_all",
        choices=["concat_all", "sum_all", "sum_extras"],
        help="Whether to sum or concat encoders",
    )

    parser.add_argument(
        "--model.n_embedding_channels",
        default=100,
        type=PositiveInt,
        help="how many channels to use for the input embeddings",
    )
    parser.add_argument(
        "--model.n_graph_layers", default=5, type=PositiveInt, help="how many message-passing steps in the model"
    )
    parser.add_argument(
        "--model.opt", default="adam", choices=["SGD", "adam", "tf_adam"], help="which optimizer to use"
    )
    parser.add_argument("--model.grad_clip_value", default=None, type=float, help="Clipping value for gradients")
    parser.add_argument("--model.l2_regularization", default=None, type=float, help="L2 weight regularization scale")

    parser.add_argument("--model.use_noisy_nodes", default=False, type=bool, help="Use noisy nodes or not")
    parser.add_argument("--model.noisy_nodes_weight", default=1.0, type=float, help="Weight of the noisy nodes loss")
    parser.add_argument(
        "--model.noisy_nodes_noise_prob", default=0.025, type=float, help="Probability of applying noise"
    )
    parser.add_argument(
        "--model.noisy_node_method",
        default="split_softmax",
        choices=["combined_softmax", "split_softmax"],
        type=str,
        help="Method for performing noisy node/edge softmax.",
    )

    parser.add_argument(
        "--model.noisy_node_model",
        default="dense",
        choices=["dense", "mlp"],
        type=str,
        help="Use a single dense layer for the noisy node decoder or a multi layer mlp.",
    )
    parser.add_argument("--model.use_noisy_edges", default=False, type=bool, help="Use noisy edges or not")
    parser.add_argument("--model.noisy_edges_weight", default=1.0, type=float, help="Weight of the noisy nodes loss")
    parser.add_argument(
        "--model.noisy_edges_noise_prob", default=0.025, type=float, help="Probability of applying noise"
    )
    parser.add_argument("--model.layer_output_scale", default=1.0, type=float, help="Scaling layer outputs")

    parser.add_argument(
        "--model.adam_m_dtype",
        default="float16",
        choices=["float16", "float32"],
        help="dtype for the m part of the adam optimizer",
    )
    parser.add_argument(
        "--model.adam_v_dtype",
        default="float16",
        choices=["float16", "float32"],
        help="dtype for the v part of the adam optimizer",
    )
    parser.add_argument(
        "--model.dtype", default="float16", choices=["float16", "mixed_float16", "float32"], help="data dtype"
    )
    parser.add_argument(
        "--model.eval_mode", default="ogb", choices=["ogb", "keras", "both"], help="Evaluator to use in inference"
    )

    # Hybrid args
    parser.add_argument(
        "--model.layer_specs",
        nargs="+",
        type=str,
        default=["MPNN+MHSA+FFN"],
        help="Config of each GPS layer in the model body",
    )
    parser.add_argument(
        "--model.layer_repeats", nargs="+", type=int, default=[], help="Repeat count for each entry in layer_specs"
    )
    parser.add_argument("--model.n_attn_heads", type=int, default=32, help="Number of self-attention heads")
    parser.add_argument(
        "--model.ffn_dim", type=int, default=768, help="Hidden dimension in the middle of the FFN (boom) layers"
    )
    parser.add_argument(
        "--model.attention_dropout_rate", type=float, default=0.1, help="Dropout for self-attention mask"
    )
    parser.add_argument("--model.ffn_dropout_rate", type=float, default=0.1, help="Dropout in the ffn boom layer")
    parser.add_argument(
        "--model.gnn_output_dropout_rate",
        type=float,
        default=0.1,
        help="Dropout for the output of the gnn in hybrid model",
    )
    parser.add_argument(
        "--model.mhsa_output_dropout_rate",
        type=float,
        default=0.1,
        help="Dropout for the output of the mhsa in hybrid model",
    )
    parser.add_argument(
        "--model.ffn_output_dropout_rate",
        type=float,
        default=0.1,
        help="Dropout for the output of the ffn in hybrid model",
    )
    parser.add_argument("--model.num_gaussian_kernels", type=int, help="Number of Gaussian basis kernels")
    parser.add_argument(
        "--model.gaussian_kernel_init_low",
        type=float,
        default=0.0,
        help="Lower bound for gaussian kernel mean/std initialisation",
    )
    parser.add_argument(
        "--model.gaussian_kernel_init_high",
        type=float,
        default=1.5,
        help="Upper bound for gaussian kernel mean/std initialisation",
    )
    parser.add_argument(
        "--model.gaussian_kernel_scale", type=float, default=1.0, help="Static scale for gaussian kernels"
    )
    parser.add_argument(
        "--model.gaussian_kernel_epsilon", type=float, default=1e-5, help="Epsilon for gaussian kernels"
    )
    parser.add_argument(
        "--model.max_path_length",
        type=int,
        default=5,
        help="max edges in a path that contribute to the edge feature encoding",
    )
    parser.add_argument(
        "--model.max_shortest_path_distance",
        type=int,
        default=100,
        help="Maximum SPD in any molecule, should be > max graph diameter in the dataset",
    )
    parser.add_argument(
        "--model.graph_dropout_rate",
        type=float,
        default=0.0,
        help="Dropout whole graphs in the stochastic depth fashion, rather than individual nodes."
        " Applied to MHSA, MPNN and FFN in the hybrid model.",
    )
    parser.add_argument(
        "--model.override_graph_dropout_rate.FFN",
        type=Optional[float],
        default=None,
        help="Dropout whole graphs in the FFN rather than individual nodes.",
    )
    parser.add_argument(
        "--model.override_graph_dropout_rate.MHSA",
        type=Optional[float],
        default=None,
        help="Dropout whole graphs in the MHSA rather than individual nodes.",
    )
    parser.add_argument(
        "--model.override_graph_dropout_rate.MPNN",
        type=Optional[float],
        default=None,
        help="Dropout whole graphs in the MPNN rather than individual nodes.",
    )
    parser.add_argument(
        "--model.hybrid_mpnn_extra_node_residual",
        type=bool,
        default=True,
        help="Add the extra residual connection on nodes around the MPNN, even there's already one inside the MPNN",
    )

    # Dataset + Generated Data
    parser.add_argument(
        "--dataset.split_path",
        type=str,
        default="./pcqm4mv2-cross_val_splits/",
        help="The path where split files are saved.",
    )
    parser.add_argument(
        "--dataset.split_mode",
        type=str,
        default="original",
        choices=["original", "incl_half_valid", "47_kfold", "train_plus_valid"],
        help="Which dataset split to use (options: original, 47_k_fold, add_half_valid, train_plus_valid)",
    )
    parser.add_argument("--dataset.split_num", type=int, default=0, help="Which dataset split number to use.")
    parser.add_argument(
        "--dataset.trim_chemical_features", type=bool, default=False, help="Trim chemical input features"
    )
    parser.add_argument(
        "--dataset.chemical_node_features",
        type=list,
        default=[
            "atomic_num",
            "chiral_tag",
            "degree",
            "possible_formal_charge",
            "possible_numH",
            "possible_number_radical_e",
            "possible_hybridization",
            "possible_is_aromatic",
            "possible_is_in_ring",
        ],
        help="Which chemical node features to use.",
    )
    parser.add_argument(
        "--dataset.chemical_edge_features",
        type=list,
        default=["possible_bond_type", "possible_bond_stereo", "possible_is_conjugated"],
        help="Which chemical edge features to use.",
    )
    parser.add_argument(
        "--dataset.use_periods_and_groups",
        default=False,
        type=bool,
        help="Convert atomic number to groups and periods as additional node input features",
    )
    parser.add_argument(
        "--dataset.do_not_use_atomic_number",
        default=False,
        type=bool,
        help="Option to not use the atomic number as input feature when groups and periods are used",
    )
    parser.add_argument(
        "--dataset.dataset_name",
        default="pcqm4mv2",
        choices=["generated", "pcqm4mv2", "pcqm4mv2_conformers_28features"],
        help="which dataset to use",
    )
    parser.add_argument("--dataset.cache_path", default=".", type=str, help="Path to download the datasets to.")
    parser.add_argument(
        "--dataset.generated_data",
        default=False,
        type=bool,
        help="Use randomly generated data instead of a real dataset.",
    )
    parser.add_argument(
        "--dataset.generated_data_n_nodes",
        default=24,
        type=PositiveInt,
        help="nodes per graph for the randomly generated dataset",
    )
    parser.add_argument(
        "--dataset.generated_data_n_edges",
        default=50,
        type=PositiveInt,
        help="edges per graph for the randomly generated dataset",
    )
    parser.add_argument(
        "--dataset.generated_data_n_graphs",
        default=2048,
        type=PositiveInt,
        help="Number of graphs for the randomly generated dataset",
    )
    parser.add_argument(
        "--dataset.normalize_labels",
        default=False,
        type=bool,
        help="Optionally normalize the labels [Only valid for PCQ].",
    )
    parser.add_argument("--dataset.prop_to_use", default=1.0, type=float, help="Proportion of the dataset to use.")
    parser.add_argument(
        "--dataset.valid.prop_to_use", default=1.0, type=float, help="Proportion of the dataset to use."
    )
    parser.add_argument(
        "--dataset.clean_train.prop_to_use", default=0.1, type=float, help="Proportion of the dataset to use."
    )
    parser.add_argument("--dataset.features", default={}, type=dict, help="Which features to use.")

    parser.add_argument(
        "--dataset.load_from_cache",
        default=True,
        type=bool,
        help="Whether to attempt to load preprocessed dataset from cache.",
    )
    parser.add_argument(
        "--dataset.save_to_cache",
        default=True,
        type=bool,
        help="Whether to attempt to save preprocessed dataset to cache.",
    )
    parser.add_argument(
        "--dataset.packing_strategy",
        default="streaming",
        choices=["pad_to_max", "streaming"],
        help="Which packing strategy to use.",
    )
    parser.add_argument(
        "--dataset.parallel_processes",
        default=240,
        type=int,
        help="How many parallel processes to use when processing smiles.",
    )
    parser.add_argument(
        "--dataset.ogbBL_norm",
        default="mean_only",
        type=str,
        choices=["z_score", "std_only", "mean_only", "None"],
        help="What normalization method for OGB bond lengths.",
    )
    parser.add_argument(
        "--dataset.distance_norm",
        default="mean_only",
        type=str,
        choices=["z_score", "std_only", "mean_only", "None"],
        help="What normalization method for OGB atom distances.",
    )

    parser.add_argument(
        "--dataset.input_masking_groups",
        type=Optional[List[List[str]]],
        default=None,
        help="Groups of input features to mask together. The first group is always used for inference",
    )
    parser.add_argument(
        "--dataset.input_masking_weights",
        type=Optional[List[float]],
        default=None,
        help="Weights for the input masking groups will use 1:1:... by default",
    )

    parser.add_argument(
        "--dataset.use_distance_sum_feature",
        default=True,
        type=bool,
        help="If atom distances are being used, also generate node features for the sum of distances",
    )

    # Training + Validation + Test Options
    parser.add_argument("--do_training", default=True, type=bool, help="Run training on the dataset")
    parser.add_argument("--do_validation", default=True, type=bool, help="Run validation on the dataset")
    parser.add_argument(
        "--validate_every_n_epochs",
        default=10,
        type=int,
        help="How often to try and validate the training. Important when using moving average as checkpoints saved each epoch.",
    )
    parser.add_argument("--do_clean_training", default=True, type=bool, help="Evaluate on training data without noise")
    parser.add_argument("--do_test", default=False, type=bool, help="Run test on the dataset")
    parser.add_argument(
        "--inference_fold",
        default="valid",
        type=str,
        help="Run inference on 'valid', 'test-dev' or 'test-challenge' fold.",
    )
    # Monitoring (wandb, execution profiles, checkpoints)
    parser.add_argument(
        "--execution_profile", default=False, type=bool, help="Create an execution profile in TensorBoard."
    )
    parser.add_argument("--wandb", default=True, type=bool, help="Enable logging to Weights & Biases")
    parser.add_argument("--upload_final_ckpt", default=True, type=bool, help="Upload the final checkpoint to wandb.")
    parser.add_argument("--wandb_entity", default="ogb-lsc-comp", help="WandB entity.")
    parser.add_argument("--wandb_project", default="PCQM4Mv2", help="WandB project.")
    parser.add_argument("--_note", default=None, help="Add note to config to view in WandB")
    parser.add_argument(
        "--note", default=None, help="Add note to config to view in WandB"
    )  # add this back to accommodate ensembling for previous runs
    parser.add_argument(
        "--checkpoint_dir", default="checkpoints", help="Base directory to save checkpoints to. Usually `checkpoints`."
    )
    parser.add_argument(
        "--submission_results_dir", default="submission_results", help="Base directory to save submission results to."
    )
    parser.add_argument(
        "--save_checkpoints_locally",
        default=False,
        type=bool,
        help="Save the checkpoints to the local dir. Otherwise saved to tmp/",
    )
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint file if skipping training.")
    parser.add_argument("--checkpoint_every_n_epochs", default=1, type=int, help="Create checkpoints every N epochs.")

    # ipu options
    parser.add_argument(
        "--ipu_opts.replicas", default=1, type=int, help="The number of replicas to scale the model over."
    )
    parser.add_argument(
        "--ipu_opts.gradient_accumulation_factor",
        default=1,
        type=int,
        help="The number of times to locally accumulate gradients.",
    )
    parser.add_argument(
        "--ipu_opts.gradient_accumulation_dtype",
        default=None,
        type=str,
        help="Dtype to store accumulated gradients in.",
    )
    parser.add_argument(
        "--ipu_opts.num_pipeline_stages", default=1, type=int, help="The number of pipeline stages to use."
    )
    parser.add_argument(
        "--ipu_opts.pipeline_stages",
        type=Optional[List[List[str]]],
        default=None,
        help="""Pipeline stages, a list of [enc, hid, dec] layers forming the pipeline.""",
    )
    parser.add_argument(
        "--ipu_opts.pipeline_device_mapping", type=List[int], help="""Mapping of pipeline stages to IPU"""
    )
    parser.add_argument("--ipu_opts.recompute", default=False, type=bool, help="Do recomputation")
    parser.add_argument(
        "--ipu_opts.offload_optimizer_state",
        default=False,
        type=bool,
        help="Offload optimizer state to external memory",
    )

    parser.add_argument("--ipu_opts.RTS", default=False, type=bool, help="Turn on replicated optimizer state sharding")

    parser.add_argument(
        "--ipu_opts.available_memory_proportion",
        default=[0.2],
        type=List[float],
        help="memory proportion to reserve for matmuls",
    )
    parser.add_argument(
        "--ipu_opts.optimization_target",
        default="cycles",
        choices=["balanced", "cycles", "memory"],
        help="optimization target for the planner",
    )
    parser.add_argument(
        "--ipu_opts.scheduling_algorithm",
        default="CHOOSE_BEST",
        choices=["CHOOSE_BEST", "SHORTEST_PATH", "CLUSTERING", "POST_ORDER", "LOOK_AHEAD"],
        help="the schedling algorithm to use.",
    )
    parser.add_argument(
        "--ipu_opts.maximum_cross_replica_sum_buffer_size",
        default=1000000,
        type=int,
        help="max size of the cross-replica sum buffer",
    )
    parser.add_argument("--ipu_opts.fp_exceptions", default=False, type=bool, help="Turn on floating point exceptions.")
    parser.add_argument("--ipu_opts.nanoo", default=False, type=bool, help="Turn on NaN on overflow.")

    # Layers Options
    parser.add_argument(
        "--layer.rn_multiplier", default="none", choices=["constant", "softplus", "none"], help="RN multiplier"
    )
    parser.add_argument(
        "--layer.decoder_mode", default="node_global", choices=["node_global", "global", "node"], help="decoder mode"
    )
    parser.add_argument("--layer.weight_dtype", choices=["float16", "float32"], help="decoder mode")
    parser.add_argument(
        "--layer.mlp_norm",
        default="layer_hidden",
        choices=["none", "layer_hidden", "layer_output"],
        help="For the MLPs, whether and where to use normalization.",
    )
    parser.add_argument(
        "--layer.activation_function",
        default="relu",
        choices=["relu", "gelu", "swish"],
        help="Activation function used for the MLPs.",
    )
    parser.add_argument(
        "--layer.gather_scatter",
        default="grouped",
        choices=["grouped", "debug", "dense"],
        help="if `grouped`, use the batch axis to separate packs which cannot speak to each other. This may "
        "speed up computation by using grouped gather/scatter underlying implementations. "
        "If `dense`, senders/receivers will be one-hot matrices and matmuls will be used. "
        "If `debug`, will use a list comprehension over the batch dimension (this is bad and slow "
        "but may be useful for debugging",
    )

    parser.add_argument(
        "--layer.one_hot_embeddings", default=False, type=bool, help="Use a one-hot formulation of the embedding lookup"
    )

    # New Args
    parser.add_argument(
        "--layer.gather_from",
        default="both",
        choices=["both", "receivers", "senders"],
        help="gather from option in interaction network",
    )
    parser.add_argument(
        "--layer.scatter_to",
        default="receivers",
        choices=["both", "receivers", "senders"],
        help="scatter to option in interaction network",
    )
    parser.add_argument(
        "--layer.concat_globals_to",
        default=["nodes", "edges"],
        type=list,
        # choices=["nodes", "edges", "both"],
        help="Which inputs to concat globals to",
    )
    parser.add_argument(
        "--layer.aggregator",
        default=["sum"],
        type=list,
        # choices=["sum", "max", "min", "mean", "var", "std", "sqrtN", "softmax"],
        help="aggregation function to use in scatter and pooling layers for GNN",
    )
    parser.add_argument(
        "--layer.direct_neighbour_aggregation",
        default=False,
        type=bool,
        help="Append node representation to outgoing edge message",
    )
    parser.add_argument(
        "--layer.node_combine_method",
        default="concat",
        choices=["concat", "sum"],
        help="How to combine nodes after the gathers from senders/receivers and scatter to senders/receivers",
    )

    parser.add_argument(
        "--debug.last_layer_only",
        default=False,
        help="Add debug stats for final processing layer only (i.e. just before the decoder)",
    )
    parser.add_argument("--debug.max_abs", type=bool, default=True, help="Include 'max_abs' in debug stats")
    parser.add_argument("--debug.mean_abs", type=bool, default=False, help="Include 'mean_abs' in debug stats")
    parser.add_argument("--debug.mean", type=bool, default=False, help="Include 'mean' in debug stats")
    parser.add_argument("--debug.var", type=bool, default=False, help="Include 'var' in debug stats")
    parser.add_argument(
        "--debug.check_data_for_nans", type=bool, default=False, help="When collecting dataset stas also check for NaNs"
    )
    parser.add_argument(
        "--debug.calc_conformer_stats",
        type=bool,
        default=False,
        help="When collecting dataset calculate the conformer position stats",
    )
    return parser


def parse_args():
    parser = get_parser()
    return parser.parse_args()


def parse_dict(in_dict):
    parser = get_parser()
    return parser.parse_object(in_dict)
