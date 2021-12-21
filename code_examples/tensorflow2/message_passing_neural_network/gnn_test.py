# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import re
import os
from examples_tests.test_util import SubProcessChecker

WORKING_PATH = os.path.dirname(os.path.realpath(__file__))


class MPNN_TEST(SubProcessChecker):

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_graph_isomorphism_network(self):
        cmd = "python3 benchmark.py --model=graph_isomorphism --epochs=5 --num_ipus=1 " \
              "--use_edges=True --batch_size=128"
        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 1,725,006", "Throughput:"))
        losses, throughputs = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_interaction_network(self):
        cmd = "python3 benchmark.py --model=interaction_network --epochs=10 --num_ipus=1 " \
              "--batch_size=32 --n_hidden=128 --n_latent=128 --mlp_norm=none --dtype=float32"

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 703,145", "Throughput:"))
        losses, throughputs = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    def test_graph_network(self):
        cmd = "python3 benchmark.py --model=graph_network --epochs=10 --num_ipus=1 --batch_size=32 " \
              "--n_hidden=128 --n_latent=128 --mlp_norm=none --n_graph_layers=1 --dtype=float32"

        output = self.run_command(cmd, WORKING_PATH, ("Total Parameters: 340,905", "Throughput:"))
        losses, throughputs = self.parse_result_for_metrics(output, cmd)
        self.loss_seems_reasonable(losses, output, cmd)

    def parse_result_for_metrics(self, output, cmd):
        losses, throughputs = [], []

        match_loss_re = re.compile(r"loss: ([\d.]+)")
        throughput_re = re.compile(r'^Throughput: ([0-9.,]+) samples')

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
        # Test that loss at end is less than loss at start
        if not (losses[-1] < losses[0] or (.65 < losses[0] < .73)):
            self.fail((f"The losses do not seem reasonable.\nThe following command failed: {cmd}\n"
                       f"Working path: {WORKING_PATH}\n"
                       f"Output of failed command:\n{output}"))
        # the loss should converge to ln(2) (coinflip)
        if not .65 < losses[-1] < .73:
            self.fail(f"The loss is not converging to the desired value of ln(2).\n"
                      f"The following command failed: {cmd}\n"
                      f"Working path: {WORKING_PATH}\n"
                      f"Output of failed command:\n{output}")
