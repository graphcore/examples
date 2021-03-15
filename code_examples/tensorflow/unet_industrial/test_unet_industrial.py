# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent


class TestUNet(SubProcessChecker):
    """High-level integration test for TensorFlow UNet-Industrial example """


    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_unet_industrial(self):
        self.run_command("python3 unet_industrial.py "
                         "--training --evaluation --inference "
                         "--num-replicas-train=1 --num-replicas-infer=2 "
                         "--batch-size-train=1 --batch-size-infer=1 "
                         "--input-size=128 --epochs=50",
                         working_path,
                         ["UNet Industrial example",
                          "Completed"])
