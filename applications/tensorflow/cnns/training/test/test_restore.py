# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import glob
import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from test_common import run_train, run_restore, cifar10_data_dir
from examples_tests.test_util import SubProcessChecker

working_path = Path(__file__).parent.parent


@pytest.mark.ipus(1)
class TestiCifar10Restoring(SubProcessChecker):
    """restore options for cifar-10. """
    """train cifar-10 first to get checkpoint and restore-path """

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        tmpdir = str(self.tmpdir.name)
        out = run_train(
            self,
            **{
                '--data-dir': cifar10_data_dir,
                '--name-suffix': 'restore_test',
                '--log-dir': tmpdir,
                '--iterations': 10,
                '--batches-per-step': 10})
        list_of_files = glob.glob(str(tmpdir) + "/*")
        self.logdir = list_of_files[0]

    def test_cifar10_restore(self):
        print(os.path.isfile(self.logdir))
        cmd = "python3 restore.py --restore-path " + self.logdir
        self.run_command(cmd,
                         working_path,
                         ["Saved initial checkpoint to",
                          "ckpt",
                          "Restoring training from latest checkpoint:"])
