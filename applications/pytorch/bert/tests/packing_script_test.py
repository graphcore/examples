# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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


import sys
import re
import subprocess
from pathlib import Path
import pytest
# Append bert directory
bert_root_path = str(Path(__file__).parent.parent)
sys.path.append(bert_root_path)

bert_root_dir = Path(__file__).parent.parent.resolve()


def pack_sample_text(cmd):
    try:
        out = subprocess.check_output(
            cmd, cwd=bert_root_dir, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


@pytest.mark.skip_longtest_needs_dataset
def test_packing_script():

    cmd_pack_sample_text = [
        "python3", "data/packing/pack_pretraining_data.py",
        "--input-files", "./data/sample_text.tfrecord",
        "--output-dir", "./data/packed_usample_text",
        "--mask-tokens", "20",
        "--sequence-length", "128"]

    out = pack_sample_text(cmd_pack_sample_text)

    for line in out.split("\n"):
        if line.find("Packing took:") != -1:
            split = re.split(r" ", line)
            print(split)
            time, packs_left = float(split[2]), float(split[4])
            break
    assert (time > 0)
    assert (packs_left == 0.0)
