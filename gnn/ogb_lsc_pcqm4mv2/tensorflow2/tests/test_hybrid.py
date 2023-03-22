# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from tests.subprocess_checker import SubProcessChecker

WORKING_PATH = Path(__file__).parent.parent


class TestHybrid(SubProcessChecker):
    @classmethod
    def setup_class(cls):
        cls.cmd = (
            "python3 run_training.py "
            "--model.micro_batch_size=16 "
            "--model.n_nodes_per_pack=40 "
            "--model.n_edges_per_pack=80 "
            "--model.n_graphs_per_pack=4 "
            "--model.node_latent=128 "
            "--model.node_exp_ratio=2 "
            "--model.node_mlp_layers=2 "
            "--model.edge_latent=32 "
            "--model.global_latent=128 "
            "--model.n_embedding_channels=100 "
            "--model.node_dropout=0.056 "
            "--model.edge_dropout=0.11 "
            "--model.global_dropout=0.33 "
            "--model.ffn_dim=128 "
            "--model.n_attn_heads=4 "
            "--model.attention_dropout_rate=0.1 "
            "--model.ffn_dropout_rate=0.1 "
            "--model.mhsa_output_dropout_rate=0.1 "
            "--model.gnn_output_dropout_rate=0.1 "
            "--model.ffn_output_dropout_rate=0.1 "
            "--model.epochs=3 "
            "--model.opt=adam "
            "--model.adam_m_dtype=float32 "
            "--model.adam_v_dtype=float32 "
            "--model.lr=2e-4 "
            "--model.learning_rate_schedule=cosine "
            "--model.loss_scaling=16 "
            "--model.dtype=float32 "
            "--model.layer_output_scale=0.707 "
            "--layer.rn_multiplier=none "
            "--layer.decoder_mode=node_global "
            "--layer.activation_function=gelu "
            "--layer.mlp_norm=layer_hidden "
            "--layer.gather_scatter=grouped "
            "--layer.one_hot_embeddings=false "
            "--layer.direct_neighbour_aggregation=true "
            "--layer.scatter_to=both "
            "--dataset.dataset_name=generated "
            "--dataset.packing_strategy=streaming "
            "--dataset.normalize_labels=true "
            "--dataset.generated_data=false "
            "--do_training=true "
            "--do_validation=false "
            "--do_test=False "
            "--execution_profile=false "
            "--wandb=False "
            "--wandb_entity=ogb-lsc-comp "
            "--wandb_project=PCQM4Mv2 "
            "--checkpoint_every_n_epochs=5 "
            "--ipu_opts.available_memory_proportion=[0.2] "
            "--ipu_opts.optimization_target=cycles "
            "--ipu_opts.scheduling_algorithm=CHOOSE_BEST "
            "--ipu_opts.maximum_cross_replica_sum_buffer_size=1000000 "
            "--ipu_opts.fp_exceptions=false "
            "--ipu_opts.nanoo=true "
            "--dataset.save_to_cache=False "
        )

    def test_hybrid_with_edge_and_global_feature(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'laplacian_eig':{'max_freqs':3,'eigvec_norm':'L2'},'random_walk':{'k_steps':[1]},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','lap_eig_vals','lap_eig_vecs','random_walk_landing_probs','shortest_path_distances']"
        )
        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 740,381", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_with_noisy_nodes_and_additional_features(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'laplacian_eig':{'max_freqs':3,'eigvec_norm':'L2'},'random_walk':{'k_steps':[1]},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','lap_eig_vals','lap_eig_vecs','random_walk_landing_probs','shortest_path_distances'] "
            "--model.use_noisy_nodes=True --model.noisy_nodes_noise_prob=0.001 --model.noisy_nodes_weight=2.0"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: ", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        # self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_with_noisy_nodes(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','shortest_path_distances'] "
            "--model.use_noisy_nodes=True --model.noisy_nodes_noise_prob=0.001 --model.noisy_nodes_weight=2.0"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 610,762", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        # self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_with_noisy_nodes_and_edges(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','shortest_path_distances'] "
            "--model.use_noisy_nodes=True --model.noisy_nodes_noise_prob=0.001 --model.noisy_nodes_weight=2.0 "
            "--model.use_noisy_edges=True"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 611,191", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        # self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_no_additional_features(self):
        cmd = (
            self.cmd + "--model.use_edges=false "
            "--model.use_globals=false "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','shortest_path_distances']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 490,153", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_with_edge_features(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=false "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','shortest_path_distances']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 407,197", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_multi_replica(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN+MHSA+FFN "
            "--ipu_opts.replicas=4 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'laplacian_eig':{'max_freqs':3,'eigvec_norm':'L2'},'random_walk':{'k_steps':[1]},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','lap_eig_vals','lap_eig_vecs','random_walk_landing_probs','shortest_path_distances']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 740,381", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 2.0)

    def test_hybrid_MPNN_only_with_additional_features(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'laplacian_eig':{'max_freqs':3,'eigvec_norm':'L2'},'random_walk':{'k_steps':[1]}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','lap_eig_vals','lap_eig_vecs','random_walk_landing_probs']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 640,393", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_MHSA_only(self):
        cmd = (
            self.cmd + "--model.use_edges=false "
            "--model.use_globals=false "
            "--model.layer_specs MHSA "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','shortest_path_distances']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 193,577", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_MPNN_FFN_only(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.layer_specs MPNN+FFN "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'laplacian_eig':{'max_freqs':3,'eigvec_norm':'L2'},'random_walk':{'k_steps':[1]}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','lap_eig_vals','lap_eig_vecs','random_walk_landing_probs']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 673,673", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)

    def test_hybrid_MPNN_MHSA_graph_dropout(self):
        cmd = (
            self.cmd + "--model.use_edges=true "
            "--model.use_globals=true "
            "--model.node_latent=32 "
            "--model.layer_specs MPNN+MHSA MPNN "
            "--model.graph_dropout_rate=0.1 "
            "--ipu_opts.replicas=1 "
            "--dataset.features={'senders_receivers':{},'graph_idxs':{},'laplacian_eig':{'max_freqs':3,'eigvec_norm':'L2'},'random_walk':{'k_steps':[1]},'shortest_path_distances':{}} "
            "--inputs=['node_feat','edge_feat','receivers','senders','node_graph_idx','edge_graph_idx','lap_eig_vals','lap_eig_vecs','random_walk_landing_probs','shortest_path_distances']"
        )

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 335,581", "Throughput:"))
        losses, _ = self.parse_result_for_metrics(output)
        self.loss_seems_reasonable(losses, 0.001, 1.0)
