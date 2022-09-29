# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import re
from pathlib import Path

from examples_tests.test_util import SubProcessChecker


WORKING_PATH = Path(__file__).parent.parent


@pytest.mark.usefixtures("ipu_static_ops")
class MPNN_TEST(SubProcessChecker):

    @pytest.mark.ipus(1)
    def test_graph_isomorphism_network(self):
        cmd = "python3 run_training.py --model=graph_isomorphism --epochs=5 --replicas=1 " \
              "--use_edges=True --micro_batch_size=128 --generated_data=True --wandb=False " \
              "--n_nodes_per_pack=24 --n_edges_per_pack=50 --n_graphs_per_pack=1 " \
              "--generated_batches_per_epoch=128 --lr=1e-5 --loss_scaling=1.0"

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 1,725,006", "throughput:"))
        losses, _ = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    @pytest.mark.ipus(2)
    def test_graph_isomorphism_network_multi_replica(self):
        cmd = "python3 run_training.py --model=graph_isomorphism --epochs=5 --replicas=2 " \
              "--use_edges=True --micro_batch_size=64 --generated_data=True --wandb=False " \
              "--n_nodes_per_pack=24 --n_edges_per_pack=50 --n_graphs_per_pack=1 " \
              "--generated_batches_per_epoch=128 --lr=1e-5 --loss_scaling=1.0"

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 1,725,006", "throughput:"))
        losses, _ = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    @pytest.mark.ipus(1)
    def test_interaction_network(self):
        cmd = "python3 run_training.py --model=interaction_network --epochs=10 --replicas=1 " \
              "--micro_batch_size=32 --n_hidden=128 --n_latent=128 --mlp_norm=none --dtype=float32 " \
              "--generated_data=True --wandb=False --n_nodes_per_pack=24 --n_edges_per_pack=50 " \
              "--n_graphs_per_pack=1 --generated_batches_per_epoch=128 --lr=1e-5 --loss_scaling=1.0"

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 703,145", "throughput:"))
        losses, _ = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    @pytest.mark.ipus(1)
    def test_graph_network(self):
        cmd = "python3 run_training.py --model=graph_network --epochs=10 --replicas=1 --micro_batch_size=32 " \
              "--n_hidden=128 --n_latent=128 --mlp_norm=none --n_graph_layers=1 --dtype=float32 " \
              "--generated_data=True --wandb=False --n_nodes_per_pack=24 --n_edges_per_pack=50 " \
              "--n_graphs_per_pack=1 --generated_batches_per_epoch=128 --lr=1e-5 --loss_scaling=1.0"

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 340,905", "throughput:"))
        losses, _ = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    def parse_result_for_metrics(self, output, cmd):
        losses, throughputs = [], []

        match_loss_re = re.compile(r"loss: ([\d.]+)")
        throughput_re = re.compile(r'^throughput: ([0-9.,]+) samples/sec')

        for line in output.split("\n"):
            match_loss = match_loss_re.search(line)
            throughput = throughput_re.match(line)

            if match_loss:
                losses.append(float(match_loss.groups()[0]))
            elif throughput:
                throughputs.append(float(throughput.groups()[0]))

        # Print output if it failed to run (no losses or accuracies found)
        if not losses:
            self.fail((f"The following command failed: {cmd}\nWorking path: {WORKING_PATH}\n"
                       f"No losses were detected at all.\nOutput of failed command:\n{output}")
                      )
        return losses, throughputs

    def loss_seems_reasonable(self, losses, output, cmd):
        min_expected_loss = .65
        max_expected_loss = .73
        # Test that loss at end is less than loss at start
        if not (losses[-1] < losses[0] or (min_expected_loss < losses[0] < max_expected_loss)):
            self.fail((f"The losses do not seem reasonable.\nThe following command failed: {cmd}\n"
                       f"Working path: {WORKING_PATH}\n"
                       f"Output of failed command:\n{output}"))
        # the loss should converge to ln(2) (coinflip)
        if not min_expected_loss < losses[-1] < max_expected_loss:
            self.fail(f"The loss is not converging to the desired value of ln(2).\n"
                      f"Achieved loss {losses[-1]},"
                      f" expected between {min_expected_loss} and {max_expected_loss}\n"
                      f"The following command failed: {cmd}\n"
                      f"Working path: {WORKING_PATH}\n"
                      f"Output of failed command:\n{output}")
