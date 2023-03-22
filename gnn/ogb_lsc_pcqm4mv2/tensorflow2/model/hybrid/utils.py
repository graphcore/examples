# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

ALLOWED_SPEC_FLAGS = [
    "MPNN",
    "MHSA",
    "FFN",
]


def _parse_layer_spec(spec):
    spec = set(f.strip() for f in spec.split("+"))
    for flag in spec:
        if flag not in ALLOWED_SPEC_FLAGS:
            raise ValueError(f"Unrecognised layer_spec value {flag}")

    # Rules: to prevent illegal combianations #
    assert spec, "Non-empty spec"
    assert "MPNN" in spec or "MHSA" in spec, "No option selected before FFN"

    return spec


def parse_layer_specs(input_specs, repeats):
    if isinstance(input_specs, str):
        input_specs = [input_specs]
    if isinstance(repeats, int):
        repeats = [repeats]
    assert isinstance(input_specs, list) and isinstance(repeats, list)
    assert (
        len(input_specs) == len(repeats) or repeats == []
    ), "If layer_repeats is specified, there should be one number per layer spec"
    if len(input_specs) != len(repeats):
        repeats = [1] * len(input_specs)

    out_specs = []
    for input_spec, repeat in zip(input_specs, repeats):
        sub_specs = [_parse_layer_spec(s) for s in input_spec.split("__")]
        out_specs.extend(sub_specs * repeat)

    return out_specs


def enforce_GNN_param_defaults(
    node_latent,
    node_exp_ratio,
    node_mlp_layers,
    node_prenorm,
    use_edges,
    edge_latent,
    edge_exp_ratio,
    edge_mlp_layers,
    edge_prenorm,
    use_globals,
    global_latent,
    global_exp_ratio,
    global_mlp_layers,
    global_prenorm,
    encoder_latent,
    encoder_exp_ratio,
    encoder_mlp_layers,
    encoder_prenorm,
    encoder_act_fn,
    activation_function,
    decoder_mode,
    **kwargs,
):
    # transfer edge/global values from node if not specified
    if use_edges:
        edge_mlp_layers = edge_mlp_layers or node_mlp_layers
        edge_latent = edge_latent or node_latent
        edge_exp_ratio = edge_exp_ratio or node_exp_ratio
        edge_prenorm = edge_prenorm if edge_prenorm is not None else node_prenorm  # False is a valid value

    global_latent = global_latent or node_latent
    if use_globals:
        global_mlp_layers = global_mlp_layers or node_mlp_layers
        global_exp_ratio = global_exp_ratio or node_exp_ratio
        global_prenorm = global_prenorm if global_prenorm is not None else node_prenorm

    encoder_mlp_layers = encoder_mlp_layers or node_mlp_layers
    encoder_latent = encoder_latent or node_latent
    encoder_exp_ratio = encoder_exp_ratio or node_exp_ratio
    encoder_prenorm = encoder_prenorm if encoder_prenorm is not None else node_prenorm
    encoder_act_fn = encoder_act_fn or activation_function

    decoder_mlp_layers = global_mlp_layers or node_mlp_layers
    decoder_exp_ratio = global_exp_ratio or node_exp_ratio
    decoder_hidden = int(global_latent * decoder_exp_ratio)
    decoder_mode = decoder_mode if use_globals else "node"
    decoder_prenorm = global_prenorm if global_prenorm is not None else node_prenorm
    out = dict(
        node_latent=node_latent,
        node_exp_ratio=node_exp_ratio,
        node_mlp_layers=node_mlp_layers,
        node_prenorm=node_prenorm,
        use_edges=use_edges,
        edge_latent=edge_latent,
        edge_exp_ratio=edge_exp_ratio,
        edge_mlp_layers=edge_mlp_layers,
        edge_prenorm=edge_prenorm,
        use_globals=use_globals,
        global_latent=global_latent,
        global_exp_ratio=global_exp_ratio,
        global_mlp_layers=global_mlp_layers,
        global_prenorm=global_prenorm,
        encoder_latent=encoder_latent,
        encoder_exp_ratio=encoder_exp_ratio,
        encoder_mlp_layers=encoder_mlp_layers,
        encoder_prenorm=encoder_prenorm,
        encoder_act_fn=encoder_act_fn,
        activation_function=activation_function,
        decoder_mlp_layers=decoder_mlp_layers,
        decoder_hidden=decoder_hidden,
        decoder_mode=decoder_mode,
        decoder_prenorm=decoder_prenorm,
    )
    out.update(kwargs)
    return out
