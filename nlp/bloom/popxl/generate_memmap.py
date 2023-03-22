# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import logging
from argparse import ArgumentParser
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

import numpy as np
import popxl_addons as addons
import torch

import popxl
import popxl_addons
from config import CONFIG_DIR, BloomConfig
from modelling.bloom_lm import BloomLMHeadTP2D
from modelling.decoder import BloomDecoderBlockTP2D
from modelling.embedding import BloomEmbeddingTP2D
from utils.setup import logging_setup, parse_args_with_presets

# monkey patch `utils.WeightDict` to disable shape checks, which will fail as
# `fake_hf` contains numpy arrays rather than PopXL tensors.
popxl_addons.WeightsDict = dict


def build_embedding_mapping(shard: Dict[str, torch.Tensor], config: BloomConfig, memmap_root: str):
    fake_hf = SimpleNamespace()
    fake_hf.word_embeddings = SimpleNamespace()
    fake_hf.word_embeddings.weight = shard["word_embeddings.weight"]

    fake_hf.word_embeddings_layernorm = SimpleNamespace()
    fake_hf.word_embeddings_layernorm.weight = shard["word_embeddings_layernorm.weight"]
    fake_hf.word_embeddings_layernorm.bias = shard["word_embeddings_layernorm.bias"]

    tp1 = config.execution.tensor_parallel_1
    tp2 = config.execution.tensor_parallel_2

    ir = popxl.Ir(replication=tp1 * tp2)
    dummy_input = np.ones((1, config.model.sequence_length), dtype=np.int32)
    print("Creating PopXL graph")
    with ir.main_graph:
        x = popxl.constant(dummy_input, dtype=popxl.int32, name="words")
        _, ff_graph = BloomEmbeddingTP2D(config).create_graph(x)

    print("Creating in-memory hf_mapping")
    mapping = BloomEmbeddingTP2D.hf_mapping(config, ff_graph.args, fake_hf)
    reverse_args = {
        k: "embedding." + v + ".npy" for k, v in zip(ff_graph.args.values_flat(), ff_graph.args.keys_flat())
    }

    for k, v in mapping.items():
        print(f"Memory mapping {reverse_args[k]} [{v.shape}, {v.dtype}]")
        f = np.memmap(Path(memmap_root) / reverse_args[k], dtype=v.dtype, mode="w+", shape=v.shape)
        f[:] = v[:]


def build_decoder_mapping(
    shard: Dict[str, torch.Tensor],
    layer_idx: int,
    config: BloomConfig,
    memmap_root: str,
):
    name_prefix = f"h.{layer_idx}."

    fake_hf = SimpleNamespace()
    fake_hf.self_attention = SimpleNamespace()
    fake_hf.input_layernorm = SimpleNamespace()
    fake_hf.post_attention_layernorm = SimpleNamespace()
    fake_hf.mlp = SimpleNamespace()

    fake_hf.self_attention.query_key_value = SimpleNamespace()
    fake_hf.self_attention.query_key_value.weight = shard[name_prefix + "self_attention.query_key_value.weight"]
    fake_hf.self_attention.query_key_value.bias = shard[name_prefix + "self_attention.query_key_value.bias"]

    fake_hf.self_attention.dense = SimpleNamespace()
    fake_hf.self_attention.dense.weight = shard[name_prefix + "self_attention.dense.weight"]
    fake_hf.self_attention.dense.bias = shard[name_prefix + "self_attention.dense.bias"]

    fake_hf.input_layernorm.weight = shard[name_prefix + "input_layernorm.weight"]
    fake_hf.input_layernorm.bias = shard[name_prefix + "input_layernorm.bias"]

    fake_hf.post_attention_layernorm.weight = shard[name_prefix + "post_attention_layernorm.weight"]
    fake_hf.post_attention_layernorm.bias = shard[name_prefix + "post_attention_layernorm.bias"]

    fake_hf.mlp.dense_h_to_4h = SimpleNamespace()
    fake_hf.mlp.dense_h_to_4h.weight = shard[name_prefix + "mlp.dense_h_to_4h.weight"]
    fake_hf.mlp.dense_h_to_4h.bias = shard[name_prefix + "mlp.dense_h_to_4h.bias"]

    fake_hf.mlp.dense_4h_to_h = SimpleNamespace()
    fake_hf.mlp.dense_4h_to_h.weight = shard[name_prefix + "mlp.dense_4h_to_h.weight"]
    fake_hf.mlp.dense_4h_to_h.bias = shard[name_prefix + "mlp.dense_4h_to_h.bias"]

    tp1 = config.execution.tensor_parallel_1
    tp2 = config.execution.tensor_parallel_2

    ir = popxl.Ir(replication=tp1 * tp2)
    dummy_input = np.ones(
        (config.model.sequence_length, config.model.hidden_size // tp1),
        dtype=config.model.dtype.as_numpy(),
    )
    print("Creating PopXL graph")
    with ir.main_graph:
        *_, x = addons.host_load(dummy_input, config.model.dtype, name="x")
        _, ff_graph = BloomDecoderBlockTP2D(config).create_graph(x)

    print("Creating in-memory hf_mapping")
    mapping = BloomDecoderBlockTP2D.hf_mapping(config, ff_graph.args, fake_hf)
    reverse_args = {
        k: f"decoder.{layer_idx}." + v + ".npy" for k, v in zip(ff_graph.args.values_flat(), ff_graph.args.keys_flat())
    }

    for k, v in mapping.items():
        print(f"Memory mapping {reverse_args[k]} [{v.shape}, {v.dtype}]")
        f = np.memmap(Path(memmap_root) / reverse_args[k], dtype=v.dtype, mode="w+", shape=v.shape)
        f[:] = v[:]


def build_head_mapping(shard: Dict[str, torch.Tensor], config: BloomConfig, memmap_root: str):
    fake_hf = SimpleNamespace()
    fake_hf.ln_f = SimpleNamespace()
    fake_hf.ln_f.weight = shard["ln_f.weight"]
    fake_hf.ln_f.bias = shard["ln_f.bias"]

    tp1 = config.execution.tensor_parallel_1
    tp2 = config.execution.tensor_parallel_2

    ir = popxl.Ir(replication=tp1 * tp2)
    dummy_input = np.ones(
        (config.model.sequence_length, config.model.hidden_size // tp1),
        dtype=config.model.dtype.as_numpy(),
    )
    dummy_embedding = np.ones(
        (
            config.model.embedding.vocab_size // tp1,
            config.model.hidden_size // (tp2 * 2),
        ),
        dtype=config.model.dtype.as_numpy(),
    )
    print("Creating PopXL graph")
    with ir.main_graph:
        *_, x = addons.host_load(dummy_input, config.model.dtype, name="x")
        _, ff_graph = BloomLMHeadTP2D(config).create_graph(x, dummy_embedding, dummy_embedding)

    print("Creating in-memory hf_mapping")
    mapping = BloomLMHeadTP2D.hf_mapping(config, ff_graph.args, fake_hf)
    reverse_args = {k: f"head." + v + ".npy" for k, v in zip(ff_graph.args.values_flat(), ff_graph.args.keys_flat())}

    for k, v in mapping.items():
        print(f"Memory mapping {reverse_args[k]} [{v.shape}, {v.dtype}]")
        f = np.memmap(Path(memmap_root) / reverse_args[k], dtype=v.dtype, mode="w+", shape=v.shape)
        f[:] = v[:]


"""
    args:
        world_size [int]: Total number of processes generating memory mappings
        rank [int]: Index of the current process when run using multiprocessing
        num_shards [int]: Number of Bloom shards in `shard_root`.
        shard_root [str]: Directory containing `*.bin` Bloom checkpoint shards.
        memmap_dir [str]: Output directory for memory mappings. Should match
                          value in `config/inference.yml`
"""


def main(args, config):
    assert config.execution.memmap_dir, "Provided config does not use memory mapped tensors!"
    assert args.config == "bloom_176B_pod16", "Script only supports config 'bloom_176B_pod16'"
    rank = args.rank
    world_size = args.world_size
    num_shards = args.num_shards
    shards_per_process = num_shards // world_size
    memmap_dir = args.memmap_dir

    print(
        f"Launching rank {rank} (world size={world_size}, num_shards={num_shards}, shards/process={shards_per_process})"
    )
    shards = list(range(1, num_shards + 1))[rank * shards_per_process : (rank + 1) * shards_per_process]

    def _generate_filename(i: int) -> str:
        return Path(args.shard_root) / f"pytorch_model_{i:05d}-of-{num_shards:05d}.bin"

    def _check_prefix(target_prefix: Union[str, List[str]], shard: Dict[str, torch.Tensor]) -> bool:
        flag = False
        keys = list(shard.keys())

        if isinstance(target_prefix, str):
            target_prefix = [target_prefix]

        if keys[0].split(".")[0] in target_prefix:
            flag = True

        if flag and not all(k.split(".")[0] in target_prefix for k in keys):
            raise ValueError(
                f"Non-homogeneous layers in single shard not supported, but received mixture {set(keys)}. Expected {target_prefix} only."
            )

        return flag

    def _cast_from_bfloat(shard):
        return {k: v.to(torch.float16) for k, v in shard.items()}

    is_embedding = partial(_check_prefix, ["word_embeddings", "word_embeddings_layernorm"])
    is_decoder = partial(_check_prefix, "h")
    is_head = partial(_check_prefix, "ln_f")

    def shard_generator():
        for s in shards:
            yield torch.load(_generate_filename(s))

    shard_counter = Counter()

    for i, shard in enumerate(shard_generator()):
        print(f"Processing shard {i+1}/{len(shards)}")
        shard = _cast_from_bfloat(shard)
        if is_embedding(shard):
            print("Detected embedding shard")
            assert shard_counter["embedding"] == 0, "Multiple embedding layers found but at most one expected."
            shard_counter["embedding"] += 1
            build_embedding_mapping(shard, config, memmap_dir)
        elif is_decoder(shard):
            print("Detected decoder shard")
            shard_counter["decoder"] += 1
            decoder_idx = int(list(shard.keys())[0].split(".")[1])
            build_decoder_mapping(shard, decoder_idx, config, memmap_dir)
        elif is_head(shard):
            print("Detected head shard")
            assert shard_counter["head"] == 0, "Multiple head layers found but at most one expected."
            shard_counter["head"] += 1
            build_head_mapping(shard, config, memmap_dir)
        else:
            raise ValueError(f"Shard type not recognized. shard={list(shard.keys())}")

    print("Completed memmap generation.")
    print("Summary of shards processed.")
    print(shard_counter)
    print(f"Memmaps written to '{Path(args.memmap_dir).resolve()}'")


if __name__ == "__main__":

    def custom_args(parser: ArgumentParser):

        parser.add_argument("--world-size", type=int, default=1)
        parser.add_argument("--rank", type=int, default=0)
        parser.add_argument("--num-shards", type=int, default=72)
        parser.add_argument(
            "--shard-root",
            type=str,
            default="bloom-ckpt/models--bigscience--bloom/snapshots/4d8e28c67403974b0f17a4ac5992e4ba0b0dbb6f/",
        )
        parser.add_argument("--memmap-dir", type=str, default="bloom-test-memmap")

    config, args = parse_args_with_presets(
        BloomConfig, CONFIG_DIR / "inference.yml", "release", "bloom_176B", custom_args
    )

    main(args, config)
