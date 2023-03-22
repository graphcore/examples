# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging

import numpy as np
import tensorflow as tf
import wandb

from data_utils.packed_batch_generator import PackedBatchGenerator
from model.gnn.loss import get_loss_functions_gnn, MaskedMeanAbsoluteError
from model.gnn.losses_and_metrics import MaskedMeanAbsoluteError
from model.hybrid.model import create_hybrid
from model.hybrid.utils import enforce_GNN_param_defaults


def create_model_input_dict(micro_batch_size, input_spec):
    return {
        input_s["input_name"]: tf.keras.Input(
            (input_s["shape"]), name=input_s["input_name"], dtype=input_s["model_dtype"], batch_size=micro_batch_size
        )
        for input_s in input_spec.values()
    }


def create_model(batch_generator, dataset, options, input_spec=None):

    inputs = create_model_input_dict(options.model.micro_batch_size, input_spec)
    logging.info(f"Model inputs: {inputs}")

    if options.dataset.input_masking_groups:
        # Find the unique set of input features to be masked
        masked_features = list(set([x for y in options.dataset.input_masking_groups for x in y]))
    else:
        masked_features = []

    general_options = dict(
        n_nodes_per_pack=batch_generator.n_nodes_per_pack,
        n_edges_per_pack=batch_generator.n_edges_per_pack,
        n_graphs_per_pack=batch_generator.n_graphs_per_pack,
        micro_batch_size=options.model.micro_batch_size,
        eigv_rand_sign_flip=options.model.eigv_rand_sign_flip,
        dataset=dataset,
        dataset_name=options.dataset.dataset_name,
        weight_dtype=options.layer.weight_dtype,
        pipeline_stages=options.ipu_opts.num_pipeline_stages,
        num_gaussian_kernels=options.model.num_gaussian_kernels,
        gaussian_kernel_init=(options.model.gaussian_kernel_init_low, options.model.gaussian_kernel_init_high),
        gaussian_kernel_scale=options.model.gaussian_kernel_scale,
        gaussian_kernel_epsilon=options.model.gaussian_kernel_epsilon,
        use_distance_sum_feature=options.dataset.use_distance_sum_feature,
        masked_features=masked_features,
        inputs=inputs,
    )
    gnn_options = dict(
        node_latent=options.model.node_latent,
        node_exp_ratio=options.model.node_exp_ratio,
        node_mlp_layers=options.model.node_mlp_layers,
        node_dropout=options.model.node_dropout,
        node_prenorm=options.model.node_prenorm,
        use_edges=options.model.use_edges,
        edge_latent=options.model.edge_latent,
        edge_exp_ratio=options.model.edge_exp_ratio,
        edge_mlp_layers=options.model.edge_mlp_layers,
        edge_dropout=options.model.edge_dropout,
        edge_prenorm=options.model.edge_prenorm,
        edge_dropout_loc=options.model.edge_dropout_loc,
        use_globals=options.model.use_globals,
        global_latent=options.model.global_latent,
        global_exp_ratio=options.model.global_exp_ratio,
        global_mlp_layers=options.model.global_mlp_layers,
        global_dropout=options.model.global_dropout,
        global_prenorm=options.model.global_prenorm,
        encoder_latent=options.model.encoder_latent,
        encoder_exp_ratio=options.model.encoder_exp_ratio,
        encoder_mlp_layers=options.model.encoder_mlp_layers,
        encoder_dropout=options.model.encoder_dropout,
        encoder_prenorm=options.model.encoder_prenorm,
        encoder_norm_pos=options.model.encoder_norm_pos,
        encoder_act_fn=options.model.encoder_act_fn,
        encoder_atom_model=options.model.atom_encoder_model,
        encoder_bond_model=options.model.bond_encoder_model,
        encoder_concat_mode=options.model.encoder_concat_mode,
        override_encoder_dropout=options.model.override_encoder_dropout,
        n_embedding_channels=options.model.n_embedding_channels,
        one_hot_embeddings=options.layer.one_hot_embeddings,
        expand_first_hidden=options.model.expand_first_hidden,
        mlp_norm=options.layer.mlp_norm,
        activation_function=options.layer.activation_function,
        gather_from=options.layer.gather_from,
        scatter_to=options.layer.scatter_to,
        concat_globals_to=options.layer.concat_globals_to,
        node_combine_method=options.layer.node_combine_method,
        aggregators=options.layer.aggregator,
        direct_neighbour_aggregation=options.layer.direct_neighbour_aggregation,
        rn_multiplier=options.layer.rn_multiplier,
        gather_scatter=options.layer.gather_scatter,
        decoder_mode=options.layer.decoder_mode,
    )
    output_options = dict(
        noisy_nodes=options.model.use_noisy_nodes,
        noisy_edges=options.model.use_noisy_edges,
        noisy_node_model=options.model.noisy_node_model,
    )
    hybrid_options = dict(
        ffn_dim=options.model.ffn_dim,
        n_attn_heads=options.model.n_attn_heads,
        attention_dropout_rate=options.model.attention_dropout_rate,
        ffn_dropout_rate=options.model.ffn_dropout_rate,
        gnn_output_dropout_rate=options.model.gnn_output_dropout_rate,
        mhsa_output_dropout_rate=options.model.mhsa_output_dropout_rate,
        ffn_output_dropout_rate=options.model.ffn_output_dropout_rate,
        max_shortest_path_distance=options.model.max_shortest_path_distance,
        graph_dropout_rate=options.model.graph_dropout_rate,
        override_graph_dropout_rate=options.model.override_graph_dropout_rate,
        hybrid_mpnn_extra_node_residual=options.model.hybrid_mpnn_extra_node_residual,
    )
    return create_hybrid(
        layer_specs=options.model.layer_specs,
        layer_repeats=options.model.layer_repeats,
        **general_options,
        **enforce_GNN_param_defaults(**gnn_options),
        **output_options,
        **hybrid_options,
        dtype=options.model.dtype,
    )


def get_tf_dataset(
    preprocessed_dataset, split_name, shuffle, options, pad_remainder=False, input_spec=None, ensemble=False
):
    if split_name == "train":
        if shuffle:
            prop_to_use = options.dataset.prop_to_use
        else:
            prop_to_use = options.dataset.clean_train.prop_to_use
    elif split_name == "valid":
        prop_to_use = options.dataset.valid.prop_to_use
    else:
        prop_to_use = 1.0

    batch_generator = PackedBatchGenerator(
        dataset=preprocessed_dataset,
        n_packs_per_batch=options.model.micro_batch_size,
        fold=split_name,
        n_graphs_per_pack=options.model.n_graphs_per_pack,
        n_edges_per_pack=options.model.n_edges_per_pack,
        n_nodes_per_pack=options.model.n_nodes_per_pack,
        n_epochs=options.model.epochs,
        noisy_nodes=options.model.use_noisy_nodes,
        noisy_edges=options.model.use_noisy_edges,
        noisy_nodes_noise_prob=options.model.noisy_nodes_noise_prob,
        noisy_edges_noise_prob=options.model.noisy_edges_noise_prob,
        normalize_labels=options.dataset.normalize_labels,
        ogbBL_norm=options.dataset.ogbBL_norm,
        distance_norm=options.dataset.distance_norm,
        packing_strategy=options.dataset.packing_strategy,
        input_masking_groups=options.dataset.input_masking_groups,
        input_masking_weights=options.dataset.input_masking_weights,
        randomize=shuffle,
        pad_remainder=pad_remainder,
        prop_to_use=prop_to_use,
        input_spec=input_spec,
    )

    batch_generator.get_averaged_global_batch_size(
        options.model.micro_batch_size, options.ipu_opts.gradient_accumulation_factor, options.ipu_opts.replicas
    )
    logging.info(f"Packing stats: {batch_generator.stats}")
    ground_truth_and_masks = batch_generator.get_ground_truth_and_masks() if not shuffle else None
    return batch_generator, ground_truth_and_masks


def get_loss_functions(dataset, options):
    return get_loss_functions_gnn(
        dataset=dataset,
        use_noisy_nodes=options.model.use_noisy_nodes,
        use_noisy_edges=options.model.use_noisy_edges,
        noisy_nodes_weight=options.model.noisy_nodes_weight,
        noisy_edges_weight=options.model.noisy_edges_weight,
        noisy_node_method=options.model.noisy_node_method,
    )


def get_metrics(denormalize_fn, options):
    if options.dataset.dataset_name in (
        "pcqm4mv2",
        "generated",
        "pcqm4mv2_28features",
        "pcqm4mv2_conformers",
        "pcqm4mv2_conformers_28features",
    ):

        def MAE(y_true, y_pred):
            if (
                options.dataset.dataset_name
                in ("pcqm4mv2", "pcqm4mv2_28features", "pcqm4mv2_conformers", "pcqm4mv2_conformers_28features")
                and options.dataset.normalize_labels
            ):
                return MaskedMeanAbsoluteError(y_true, y_pred, transform=denormalize_fn)
            else:
                return MaskedMeanAbsoluteError(y_true, y_pred)

        metrics = {"Main": MAE}
        return metrics

    else:
        # these need to be updated to masked versions
        return [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(dtype=tf.float32)]


def check_loaded_weights(model, all_initial_weights):
    for layer, initial_weights in zip(model.layers, all_initial_weights):
        weights = layer.get_weights()
        logging.info(f"Layer name {layer.name}")
        logging.info(f"No. of weights in layer: {len(weights)}")
        for weight, initial_weight in zip(weights, initial_weights):
            if np.array_equal(weight, initial_weight):
                logging.warning(f"------Checkpoint does not contain weights for weight {weight.shape}------")
            else:
                logging.info(f"++++++Checkpoint contains weights for weight {weight.shape}++++++")


def load_checkpoint_into_model(model, ckpt_path, debug=False):
    logging.info("Attempting to load checkpoint from" f" path {ckpt_path}.")
    all_initial_weights = [layer.get_weights() for layer in model.layers]

    model.load_weights(ckpt_path).expect_partial()
    if debug:
        check_loaded_weights(model, all_initial_weights)
