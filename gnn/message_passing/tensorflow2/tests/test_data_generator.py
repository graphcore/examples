# Copyright (c) 2022 Graphcore Ltd. All rights reserved.


from data_utils.data_generators import PackedBatchGenerator


def test_packed_data_generator():
    n_packs_per_batch = 2
    max_graphs_per_pack = 2
    max_nodes_per_pack = 248
    max_edges_per_pack = 512

    pbg = PackedBatchGenerator(
        n_packs_per_batch=n_packs_per_batch,
        n_epochs=1,
        max_graphs_per_pack=max_graphs_per_pack,
        max_nodes_per_pack=max_nodes_per_pack,
        max_edges_per_pack=max_edges_per_pack,
    )
    ds = pbg.get_tf_dataset()
    _ = [batch for batch in ds]
