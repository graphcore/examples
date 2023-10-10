# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from examples_tests.test_util import SubProcessChecker


class ShakespeareRNNTest(SubProcessChecker):
    """Test throughput examples using ipu_inference.py."""

    current_path = Path(__file__).parent

    def setUp(self):
        self.run_command(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt",
            self.current_path,
            [],
        )

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_train(self):
        self.run_command("python3 train.py --ipus 2 --epochs 1", self.current_path, ["Epoch 0: 100%"])
