# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
from functools import partial
from pickle import NONE
import tensorflow as tf
from keras.layers import Lambda
from model.encoders.base_encoder import BaseEncoder, ConcatFeatures

from model.gnn.layers import DecoderLayer, Dense
from model.hybrid.layers import GPSLayer, build_gnn_mlps
from model.hybrid.utils import parse_layer_specs


def create_hybrid(
    # main model specs
    layer_specs,
    layer_repeats,
    # general options
    n_nodes_per_pack,
    n_edges_per_pack,
    n_graphs_per_pack,
    micro_batch_size,
    dataset,
    dataset_name,
    masked_features,
    use_distance_sum_feature,
    weight_dtype,
    pipeline_stages,
    inputs,
    # encoder options
    encoder_latent,
    encoder_exp_ratio,
    encoder_mlp_layers,
    encoder_dropout,
    encoder_prenorm,
    encoder_act_fn,
    encoder_atom_model,
    encoder_bond_model,
    encoder_concat_mode,
    override_encoder_dropout,
    n_embedding_channels,
    one_hot_embeddings,
    # GNN options
    node_latent,
    node_exp_ratio,
    node_mlp_layers,
    node_dropout,
    node_prenorm,
    use_edges,
    edge_latent,
    edge_exp_ratio,
    edge_mlp_layers,
    edge_dropout,
    edge_prenorm,
    edge_dropout_loc,
    eigv_rand_sign_flip,
    use_globals,
    global_latent,
    global_exp_ratio,
    global_mlp_layers,
    global_dropout,
    global_prenorm,
    expand_first_hidden,
    mlp_norm,
    activation_function,
    gather_from,
    scatter_to,
    concat_globals_to,
    node_combine_method,
    aggregators,
    direct_neighbour_aggregation,
    rn_multiplier,
    gather_scatter,
    decoder_mode,
    decoder_mlp_layers,
    decoder_hidden,
    decoder_prenorm,
    noisy_node_model,
    # output options
    noisy_nodes,
    noisy_edges,
    # hybrid/transformer options
    ffn_dim,
    n_attn_heads,
    attention_dropout_rate,
    ffn_dropout_rate,
    gnn_output_dropout_rate,
    mhsa_output_dropout_rate,
    ffn_output_dropout_rate,
    num_gaussian_kernels,
    gaussian_kernel_init,
    gaussian_kernel_scale,
    gaussian_kernel_epsilon,
    max_shortest_path_distance,
    graph_dropout_rate,
    override_graph_dropout_rate,
    hybrid_mpnn_extra_node_residual,
    **kwargs,
):
    logging.info(f"The following input parameters are not used by the hybrid model {kwargs}")
    # add space for dummy node and dummy graph
    n_nodes_per_pack = n_nodes_per_pack + 1
    n_edges_per_pack = n_edges_per_pack + 1
    n_graphs_per_pack = n_graphs_per_pack + 1

    layer_outputs = {}

    node_mlp, edge_mlp, global_mlp, encoder_mlp, decoder_mlp = build_gnn_mlps(
        activation_function=activation_function,
        expand_first_hidden=expand_first_hidden,
        mlp_norm=mlp_norm,
        node_latent=node_latent,
        node_exp_ratio=node_exp_ratio,
        node_mlp_layers=node_mlp_layers,
        node_prenorm=node_prenorm,
        edge_latent=edge_latent,
        edge_exp_ratio=edge_exp_ratio,
        edge_mlp_layers=edge_mlp_layers,
        edge_prenorm=edge_prenorm,
        global_latent=global_latent,
        global_exp_ratio=global_exp_ratio,
        global_mlp_layers=global_mlp_layers,
        global_prenorm=global_prenorm,
        encoder_latent=encoder_latent,
        encoder_exp_ratio=encoder_exp_ratio,
        encoder_mlp_layers=encoder_mlp_layers,
        encoder_prenorm=encoder_prenorm,
        encoder_act_fn=encoder_act_fn,
        decoder_mlp_layers=decoder_mlp_layers,
        decoder_hidden=decoder_hidden,
        decoder_prenorm=decoder_prenorm,
        weight_dtype=weight_dtype,
    )

    dense_layer = partial(Dense, weight_dtype=weight_dtype)

    if encoder_atom_model == "node":
        atom_model = node_mlp
    elif encoder_atom_model == "encoder":
        atom_model = encoder_mlp
    else:
        raise ValueError(f"encoder_atom_model {encoder_atom_model} not recognised")

    if encoder_bond_model == "edge":
        bond_model = edge_mlp
    elif encoder_bond_model == "encoder":
        bond_model = encoder_mlp
    else:
        raise ValueError(f"encoder_bond_model {encoder_bond_model} not recognised")

    # ENCODER #
    x = BaseEncoder(
        inputs,
        node_model_fn=partial(atom_model, activate_final=False, name="node_encoder"),
        edge_model_fn=partial(bond_model, activate_final=False, name="edge_encoder"),
        encoder_model_fn=partial(encoder_mlp, activate_final=False, name="encoder"),
        use_globals=use_globals,
        global_latent=global_latent,
        encoder_latent=encoder_latent,
        use_edges=use_edges,
        n_embedding_channels=n_embedding_channels,
        n_attn_heads=n_attn_heads,
        encoder_dropout_rate=encoder_dropout,
        override_encoder_dropout=override_encoder_dropout,
        one_hot_embeddings=one_hot_embeddings,
        micro_batch_size=micro_batch_size,
        n_nodes_per_pack=n_nodes_per_pack,
        n_edges_per_pack=n_edges_per_pack,
        n_graphs_per_pack=n_graphs_per_pack,
        eigv_rand_sign_flip=eigv_rand_sign_flip,
        node_feature_dims=dataset.node_feature_dims,
        edge_feature_dims=dataset.edge_feature_dims,
        max_shortest_path_distance=max_shortest_path_distance,
        masked_features=masked_features,
        num_gaussian_kernels=num_gaussian_kernels,
        gaussian_kernel_init=gaussian_kernel_init,
        gaussian_kernel_scale=gaussian_kernel_scale,
        gaussian_kernel_epsilon=gaussian_kernel_epsilon,
        use_distance_sum_feature=use_distance_sum_feature,
    )(inputs)

    # Here need a concatenation into node and edge features
    x = ConcatFeatures(
        node_latent=node_latent,
        edge_latent=edge_latent,
        use_edges=use_edges,
        concat_mode=encoder_concat_mode,
        input_names=x.keys(),
    )(x)

    layer_specs = parse_layer_specs(layer_specs, layer_repeats)

    # BODY #
    layer_outputs.update({f"Nodes_Encoder": x[0], f"Edges_Encoder": x[1], f"Globals_Encoder": x[4]})
    for i, spec in enumerate(layer_specs):
        x = GPSLayer(
            node_model=partial(node_mlp, activate_final=False, name="node"),
            edge_model=partial(edge_mlp, activate_final=False, name="edge"),
            global_model=partial(global_mlp, activate_final=False, name="global"),
            micro_batch_size=micro_batch_size,
            nodes_per_pack=n_nodes_per_pack,
            edges_per_pack=n_edges_per_pack,
            graphs_per_pack=n_graphs_per_pack,
            node_latent=node_latent,
            ffn_dim=ffn_dim,
            n_attn_heads=n_attn_heads,
            attention_dropout_rate=attention_dropout_rate,
            ffn_dropout_rate=ffn_dropout_rate,
            gnn_output_dropout_rate=gnn_output_dropout_rate,
            mhsa_output_dropout_rate=mhsa_output_dropout_rate,
            ffn_output_dropout_rate=ffn_output_dropout_rate,
            gnn_node_dropout=node_dropout,
            gnn_edge_dropout=edge_dropout,
            gnn_global_dropout=global_dropout,
            gnn_edge_dropout_loc=edge_dropout_loc,
            use_edges=use_edges,
            use_globals=use_globals,
            gather_from=gather_from,
            scatter_to=scatter_to,
            aggregators=aggregators,
            concat_globals_to=concat_globals_to,
            direct_neighbour_aggregation=direct_neighbour_aggregation,
            node_combine_method=node_combine_method,
            rn_multiplier=rn_multiplier,
            gather_scatter=gather_scatter,
            layer_spec=spec,
            graph_dropout_rate=graph_dropout_rate,
            override_graph_dropout_rate=override_graph_dropout_rate,
            layer_depth_frac=(i / (len(layer_specs) - 1)) if i > 0 else 0.0,
            hybrid_mpnn_extra_node_residual=hybrid_mpnn_extra_node_residual,
            skip_global_out=(i == len(layer_specs) - 1 and "global" not in decoder_mode),
        )(x)
        layer_outputs.update({f"Nodes_Layer_{i}": x[0], f"Edges_Layer_{i}": x[1], f"Globals_Layer_{i}": x[4]})

    decoder_outputs = DecoderLayer(
        decoder_model_fn=decoder_mlp,
        noisy_node_model=decoder_mlp if noisy_node_model == "mlp" else dense_layer,
        decoder_mode=decoder_mode,
        n_graphs_per_pack=n_graphs_per_pack,
        gather_scatter=gather_scatter,
        pooling_aggregators=aggregators,
        use_edges=use_edges,
        use_globals=use_globals,
        noisy_nodes=noisy_nodes,
        noisy_edges=noisy_edges,
        node_feature_sum=dataset.node_feature_sum,
        edge_feature_sum=dataset.edge_feature_sum,
    )(x)

    if noisy_nodes or noisy_edges:
        output_prob = decoder_outputs.pop(0)
        outputs = [Lambda(lambda x: x, name="Main")(output_prob)]
        if noisy_nodes:
            outputs += [Lambda(lambda x: x, name="Noisy_Nodes")(decoder_outputs.pop(0))]
        if noisy_edges:
            outputs += [Lambda(lambda x: x, name="Noisy_Edges")(decoder_outputs.pop(0))]
    else:
        outputs = decoder_outputs
        outputs = Lambda(lambda x: x, name="Main")(outputs)

    return tf.keras.Model(inputs, outputs)
