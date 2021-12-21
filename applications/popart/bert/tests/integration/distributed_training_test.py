# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import sys
from pathlib import Path
from tests.utils import bert_root_dir
from examples_tests.test_util import SubProcessChecker

bert_root_dir()



class TestBuildAndRun(SubProcessChecker):

    def _get_bert_command(self, extra_args=None):
        base_cmd = ("poprun --num-replicas 2 --ipus-per-replica 2 --num-instances 2 --only-output-from-instance 0 --mpi-global-args='--allow-run-as-root' "
                    "python bert.py --config configs/mk2/pretrain_large_128.json "
                    " --global-batch-size 32 --micro-batch-size 1 --gradient-accumulation-factor 16 --replication-factor 1 --generated-data --epochs 1"
                    " --replicated-tensor-sharding false --encoder-start-ipu 1 --layers-per-ipu 1 --num-layers 1 --batches-per-step 1 --no-model-save --steps-per-log 1 "
                    " --no-validation --max-copy-merge-size 32000 --hidden-size 128 --vocab-length 9728 --embedding-serial 4")
        print(base_cmd)
        if extra_args is not None and len(extra_args) > 0:
            base_cmd += " " + " ".join(extra_args)
        return base_cmd

    def setUp(self):
        self.run_command("make", bert_root_dir(), [])

    @pytest.mark.ipus(4)
    def test_poprun_complete(self):
        cmd = self._get_bert_command()
        self.run_command(cmd,
                         bert_root_dir(),
                         ["Compiling Training Graph", "Compiled.", "Training Started", "Training Finished"])
