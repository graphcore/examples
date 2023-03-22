# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path
import pytest
from examples_tests.test_util import SubProcessChecker


class ApplicationsTest(SubProcessChecker):
    """Test throughput examples using ipu_inference.py."""

    current_path = Path(__file__).parent.parent

    def setUp(self):
        self.run_command("make -C ../pytorch", self.current_path, [])

    @pytest.mark.category2
    @pytest.mark.ipus(16)
    @pytest.mark.ipu_version("ipu2")
    def test_train_resnet(self):
        self.run_command(
            "python3 train_lightning.py --config resnet50 --model resnet18 --data generated --epoch 1 --validation-mode none --replica 1",
            self.current_path,
            ["Validation_accuracy"],
        )


class FashionMnistTest(SubProcessChecker):
    """Test throughput examples using ipu_inference.py."""

    current_path = Path(__file__).parent.parent / "code-examples" / "fashion-mnist"

    def setUp(self):
        self.run_command("make -C ../../../pytorch", self.current_path, [])
        pass

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_train(self):
        self.run_command("python3 train.py --num-epochs 2 --ipus 2", self.current_path, ["Epoch 1/1"])

    @pytest.mark.category2
    @pytest.mark.ipus(2)
    @pytest.mark.ipu_version("ipu2")
    def test_train(self):
        self.run_command(
            "python3 train.py --num-epochs 2 --ipus 2 --model torchvision", self.current_path, ["Epoch 1/1"]
        )


class ShakespeareRNNTest(SubProcessChecker):
    """Test throughput examples using ipu_inference.py."""

    current_path = Path(__file__).parent.parent / "code-examples" / "shakespeare-rnn"

    def setUp(self):
        self.run_command("make -C ../../../pytorch", self.current_path, [])
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


class TutorialsTest(SubProcessChecker):
    """Test throughput examples using ipu_inference.py."""

    current_path = Path(__file__).parent.parent / "tutorials"

    def setUp(self):
        self.run_command("make -C ../../pytorch", self.current_path, [])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_custom_losses(self):
        self.run_command(
            "python3 custom_losses.py", self.current_path, [r"pred [\d\.]+, actual [\d\.]+, difference [\d\.]+"]
        )

    @pytest.mark.category2
    @pytest.mark.ipus(9)
    @pytest.mark.ipu_version("ipu2")
    def test_ipu_strategy_dataloading(self):
        self.run_command("python3 ipu_strategy_and_dataloading.py", self.current_path, [r"Epoch 0: 100%"])

    @pytest.mark.category2
    @pytest.mark.ipus(4)
    @pytest.mark.ipu_version("ipu2")
    def test_ipu_pipelining(self):
        self.run_command("python3 pipelined_ipu.py", self.current_path, [r"Epoch 0: 100%"])

    @pytest.mark.category1
    @pytest.mark.ipus(1)
    @pytest.mark.ipu_version("ipu2")
    def test_ipu_pipelining(self):
        self.run_command("python3 simple_lightning_ipu.py", self.current_path, [r"Epoch 2: 100%"])
