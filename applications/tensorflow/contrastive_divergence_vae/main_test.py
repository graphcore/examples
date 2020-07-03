# Copyright 2020 Graphcore Ltd
import os
import pytest
from tests.test_util import SubProcessChecker

working_path = os.path.dirname(__file__)


class Test(SubProcessChecker):
    """ Test the contrastive divergence vae model. """
    @pytest.mark.ipus(1)
    @pytest.mark.category1
    def test_train_one_epoch_one_ipu(self):
        """ Test that the model can run training for one epoch using one IPU."""
        self.run_command(
            "python3 main.py --no-testing --only-ipu --config-file configs/test_config.json",
            working_path, ["Finished training"])
