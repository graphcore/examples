# Copyright 2020 Graphcore Ltd.
from pathlib import Path
import pytest

# NOTE: The import below is dependent on 'pytest.ini' in the root of
# the repository
from tests.test_util import SubProcessChecker

working_path = Path(__file__).parent


class TestCosmoflow(SubProcessChecker):

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_cosmoflow_model(self):
        self.run_command("python3 train.py configs/graphcore.yaml --num-ipus=2",
                         working_path, [])
