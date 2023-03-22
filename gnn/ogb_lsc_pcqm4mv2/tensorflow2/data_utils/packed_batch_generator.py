# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from data_utils.load_dataset import OGBGraphData
from data_utils.packing import pack_dataset
from data_utils.utils import apply_categorical_feature_noise, normalize_ogbBL, normalize_atom_distances, weighted_sample


@dataclass
class PackedBatchGenerator:
    """
    Generates batched representations of graph data for various datasets.
    Each 'pack' of a batch is guaranteed to only exchange information with other members of the same 'pack'. This
      allows the use of 'batched' gather/scatter implementations. In this case, the compiler is able to use this
      constraint in order to improve throughput.
    """

    n_packs_per_batch: int
    n_graphs_per_pack: int
    n_nodes_per_pack: int
    n_edges_per_pack: int
    dataset: OGBGraphData
    n_epochs: int = 100
    randomize: bool = True
    noisy_nodes: bool = False
    noisy_edges: bool = False
    noisy_nodes_noise_prob: float = 0.05
    noisy_edges_noise_prob: float = 0.05
    normalize_labels: bool = False
    ogbBL_norm: str = None
    distance_norm: str = None
    prop_to_use: float = 1.0
    fold: str = "train"
    packing_strategy: str = "streaming"
    input_masking_groups: [[str]] = None
    input_masking_weights: [[float]] = None
    pad_remainder: bool = False
    input_spec: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.packing_strategy in ("pad_to_max", "streaming")

        self.label_dtype = self.dataset.tf_labels_dtype

        self.orig_dataset = self.dataset

        self.dataset = [self.dataset.dataset[fold_idx] for fold_idx in self.dataset.dataset.split_dict[self.fold]]

        logging.info(f"Dataset has {len(self.dataset)} elements")
        if self.prop_to_use == 1.0:
            pass
        elif (self.prop_to_use > 0) and (self.prop_to_use < 1):
            samples_to_take = int(self.prop_to_use * len(self.dataset))
            self.dataset = self.dataset[:samples_to_take]
            logging.info(f"Dataset trimmed to {len(self.dataset)} elements ({100*self.prop_to_use} %)")
        else:
            raise ValueError(f"Value of 'prop_to_use' of {self.prop_to_use} invalid")

        if self.randomize:
            self.dataset = np.random.permutation(self.dataset)

        self.n_graphs_per_epoch = len(self.dataset)

        if self.packing_strategy == "pad_to_max":
            self.n_graphs_per_pack = 1
            self.n_nodes_per_pack = self.orig_dataset.stats[self.fold]["nodes"]["max"]
            self.n_edges_per_pack = self.orig_dataset.stats[self.fold]["edges"]["max"]
            logging.info(
                f"Doing pad_to_max packing with max number of nodes {self.n_nodes_per_pack} and max number of edges {self.n_edges_per_pack}"
            )
            self.packed_dataset, pack_stats = pack_dataset(
                self.dataset,
                self.n_nodes_per_pack,
                self.n_edges_per_pack,
                self.n_graphs_per_pack,
                self.input_spec,
                silence_logging=False,
            )

        elif self.packing_strategy == "streaming":
            logging.info("Doing streaming packing")
            self.packed_dataset, pack_stats = pack_dataset(
                self.dataset,
                self.n_nodes_per_pack,
                self.n_edges_per_pack,
                self.n_graphs_per_pack,
                self.input_spec,
                silence_logging=False,
            )
            # randomize again
            if self.randomize:
                self.dataset = np.random.permutation(self.dataset)
        else:
            raise ValueError()

        self.stats = {}
        self.stats.update(pack_stats)
        self.stats["graphs_per_epoch"] = self.n_graphs_per_epoch
        self.packs_per_epoch = len(self.packed_dataset)
        self.stats["packs_per_epoch"] = self.packs_per_epoch

        self.batches_per_epoch = math.ceil(self.packs_per_epoch / self.n_packs_per_batch)

        if self.pad_remainder:
            self.packs_per_epoch_with_padding = self.batches_per_epoch * self.n_packs_per_batch
            padding_amount = self.packs_per_epoch_with_padding - self.packs_per_epoch
            assert padding_amount >= 0
            self.packed_dataset += [self.get_empty_batch_dict()] * padding_amount
        else:
            self.packs_per_epoch_with_padding = self.packs_per_epoch

        self.is_training = (self.fold == "train") and (self.randomize)

        self.dataset_generator = iter(self.packed_dataset)

        self.pack_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pack_counter % max(1, self.packs_per_epoch_with_padding // 5) or self.pack_counter == (
            self.packs_per_epoch_with_padding - 1
        ):
            pct = int(round(100 * self.pack_counter / self.packs_per_epoch_with_padding))
            logging.info(f"Processed {self.pack_counter} packs ({pct}%)")
        self.pack_counter += 1
        return self.dataset_generator.__next__()

    def change_micro_batch_size(self, new_micro_batch_size):
        if self.pad_remainder:
            # dataset is depended on dataset which makes this more tricky
            ValueError("Cannot change batch side if pad remainder is used.")

        self.n_packs_per_batch = new_micro_batch_size
        self.batches_per_epoch = math.ceil(self.packs_per_epoch / self.n_packs_per_batch)

    def get_ground_truth_and_masks(self):
        assert not self.randomize, "getting the ground truth and masks can only be done without randomization"

        ground_truths = np.full([self.packs_per_epoch_with_padding, self.n_graphs_per_pack + 1], -1.0)
        for batch_idx, pack in enumerate(self.packed_dataset):
            ground_truths[batch_idx, :] = pack["labels"]
        include_sample_mask = ground_truths != -1.0
        return ground_truths, include_sample_mask

    def get_empty_batch_dict(self):
        batch_dict = {
            input_s["input_name"]: np.full(input_s["shape"], input_s["pad_value"]).astype(
                input_s["input_dtype"].as_numpy_dtype
            )
            for input_s in self.input_spec.values()
        }

        # this is used for masking: for a graph id that corresponds to '-1' label, we will not include its loss
        batch_dict["labels"] = -np.ones([self.n_graphs_per_pack + 1])

        if "shortest_path_distances" in batch_dict:
            batch_dict["shortest_path_distances"][np.diag_indices(self.n_nodes_per_pack + 1)] = 0

        if "atom_distances" in batch_dict:
            batch_dict["atom_distances"][np.diag_indices(self.n_nodes_per_pack + 1)] = 0

        return batch_dict

    def get_tf_dataset(self, repeat_num=None):
        # If `repeat_num=None, the repeat will be in infinite, otherwise will repeate the dataset for the given value.
        n_graphs = self.n_graphs_per_pack + 1
        output_signature = {
            input_s["input_name"]: tf.TensorSpec(shape=input_s["shape"], dtype=input_s["input_dtype"])
            for input_s in self.input_spec.values()
        }
        output_signature["labels"] = tf.TensorSpec(shape=(n_graphs,), dtype=self.label_dtype)

        ds = tf.data.Dataset.from_generator(self.__iter__, output_signature=output_signature)
        ds = ds.take(self.packs_per_epoch_with_padding).cache()
        if self.randomize:
            ds = ds.shuffle(int(self.packs_per_epoch_with_padding), reshuffle_each_iteration=True)
        ds = ds.repeat(repeat_num)
        # shouldn't be any remainder but drop_remainder keeps shapes static
        ds = ds.batch(self.n_packs_per_batch, drop_remainder=True)
        ds = ds.map(lambda batch: self.batch_to_outputs(batch))
        return ds

    def batch_to_outputs(self, batch):
        batch = batch.copy()
        batch_output = dict()
        for inputs in self.input_spec.values():
            input_name = inputs["input_name"]
            batch_output[input_name] = batch.pop(input_name)
        ground_truth = batch.pop("labels")

        nodes = batch_output["node_feat"]
        edges = batch_output["edge_feat"]
        node_graph_idx = batch_output["node_graph_idx"]
        edge_graph_idx = batch_output["edge_graph_idx"]
        # mask should be 1 for values to keep
        mask = tf.where(ground_truth != -1, tf.cast(1, nodes.dtype), tf.cast(0, nodes.dtype))

        if self.normalize_labels:
            ground_truth = self.orig_dataset.normalize(ground_truth)
            ground_truth = tf.where(mask == 1, ground_truth, -1)

        # must have a final dummy dimension to match the output of the sigmoid
        ground_truth = ground_truth[..., None]

        # set for cases when noisy nodes/edges do not corrupt them
        maybe_noisy_nodes = nodes
        maybe_noisy_edges = edges

        # Apply normalization to atom_distances.
        if "atom_distances" in batch_output.keys():
            batch_output["atom_distances"] = normalize_atom_distances(
                batch_output["atom_distances"], self.distance_norm
            )

        # Apply masking groups after the noised atom_distances if 3D denoising is used.
        if self.input_masking_groups is not None:
            n_graphs_per_pack = self.n_graphs_per_pack + 1
            assert len(self.input_masking_groups) == len(self.input_masking_weights)
            all_masking_features = set([x for y in self.input_masking_groups for x in y])
            all_features = set(batch_output.keys())

            logging.info(f"Features requested for grouped masking: {all_masking_features}")
            logging.info(f"Missing features: {all_masking_features - all_features}")

            all_masking_features = all_masking_features.intersection(all_features)
            mask_groups = {
                f: tf.constant([f in g for g in self.input_masking_groups]) for f in all_masking_features
            }  # true if feature is kept!

            if self.is_training:
                masking_mode_ids = weighted_sample(
                    self.input_masking_weights, self.n_packs_per_batch * n_graphs_per_pack
                )
                if "nan_in_conformer" in batch_output.keys():
                    nan_in_conformer = batch_output["nan_in_conformer"]
                    masking_mode_ids = tf.where(
                        tf.reshape(nan_in_conformer, masking_mode_ids.shape),
                        tf.zeros_like(masking_mode_ids),
                        masking_mode_ids,
                    )
            else:
                masking_mode_ids = tf.zeros([1, self.n_packs_per_batch * n_graphs_per_pack], dtype=tf.int64)

            mask_per_sample = {
                f: tf.reshape(tf.gather(mask_groups[f], masking_mode_ids), [self.n_packs_per_batch, n_graphs_per_pack])
                for f in all_masking_features
            }

            possible_node_features = [
                "lap_eig_vals",
                "lap_eig_vecs",
                "random_walk_landing_probs",
                "centrality_encoding",
            ]
            possible_edge_features = ["relative_features", "ogb_bond_lengths"]
            possible_attn_features = ["atom_distances", "shortest_path_distances"]

            for f in all_masking_features:
                if f in possible_node_features + possible_attn_features:  # node
                    graph_idx = node_graph_idx
                elif f in possible_edge_features:
                    graph_idx = edge_graph_idx
                else:
                    raise ValueError(f"Can't find a graph_idx for feature {f}")

                # -128 is a special value used to zero them later and get a mask
                masking_value = -128
                full_mask = tf.gather(mask_per_sample[f], graph_idx, batch_dims=1)

                if f in possible_attn_features:
                    # need to project node mask to nxn
                    full_mask = tf.cast(full_mask, dtype=tf.int32)
                    full_mask = tf.cast(tf.expand_dims(full_mask, -1) * tf.expand_dims(full_mask, -2), dtype=tf.bool)

                feature = batch_output[f]
                full_mask = tf.squeeze(full_mask)

                while len(full_mask.get_shape().as_list()) < len(feature.get_shape().as_list()):
                    full_mask = tf.expand_dims(full_mask, -1)
                full_mask = tf.broadcast_to(full_mask, feature.shape)
                batch_output[f] = tf.where(full_mask, feature, tf.cast(masking_value, feature.dtype))

        mask_ext = mask[..., tf.newaxis]
        # run on CPU so use batched gather rather than grouped gather custom op
        node_mask = tf.gather(mask_ext, node_graph_idx, batch_dims=1)
        batch_output["node_mask"] = tf.squeeze(node_mask)
        full_node_mask = tf.broadcast_to(node_mask, nodes.shape)
        edge_mask = tf.gather(mask_ext, edge_graph_idx, batch_dims=1)
        batch_output["edge_mask"] = tf.squeeze(edge_mask)
        full_edge_mask = tf.broadcast_to(edge_mask, edges.get_shape().as_list())
        labels = [ground_truth]

        if self.noisy_nodes or self.noisy_edges:
            if self.noisy_nodes:
                # ensure -1 values can be used to define mask in loss
                node_labels = tf.where(full_node_mask == 1, nodes, tf.cast(-1, nodes.dtype))
                labels += [node_labels]

            if self.noisy_edges:
                # ensure -1 values can be used to define mask in loss
                edges_labels = tf.where(full_edge_mask == 1, edges, tf.cast(-1, edges.dtype))
                labels += [edges_labels]

            if self.is_training:
                assert (self.fold == "train") and (self.randomize), "Can only do noisy nodes in training mode"
                if self.noisy_nodes:
                    maybe_noisy_nodes = apply_categorical_feature_noise(
                        nodes, self.orig_dataset.node_feature_dims, self.noisy_nodes_noise_prob
                    )
                    # mask any corrupted elements on padding
                    maybe_noisy_nodes = maybe_noisy_nodes * tf.cast(full_node_mask, maybe_noisy_nodes.dtype)

                if self.noisy_edges:
                    maybe_noisy_edges = apply_categorical_feature_noise(
                        edges, self.orig_dataset.edge_feature_dims, self.noisy_edges_noise_prob
                    )
                    # mask any corrupted elements on padding
                    maybe_noisy_edges = maybe_noisy_edges * tf.cast(full_edge_mask, maybe_noisy_edges.dtype)

        labels = tuple(labels)

        batch_output["node_feat"] = maybe_noisy_nodes
        batch_output["edge_feat"] = maybe_noisy_edges

        assert not batch, "all fields of the batch must be used"
        return batch_output, labels

    def get_averaged_global_batch_size(self, micro_batch_size, GA_factor, replicas):
        averaged_global_batch_size = self.stats["avg_pack"]["graphs"] * micro_batch_size * GA_factor * replicas
        self.stats["average_global_batch_size"] = averaged_global_batch_size
        self.stats["micro_batch_size"] = micro_batch_size
        self.stats["GA_factor"] = GA_factor


if __name__ == "__main__":
    pbg = PackedBatchGenerator(
        n_packs_per_batch=6,
        n_epochs=1,
        n_graphs_per_pack=16,
        n_nodes_per_pack=248,
        n_edges_per_pack=512,
    )
    print(pbg.n_graphs_per_epoch)
    ds = pbg.get_tf_dataset()
    data = [batch for batch in ds]
    print(data[-1][-1])
