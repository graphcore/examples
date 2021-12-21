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
import subprocess
import re
import os
import tempfile
from itertools import chain, islice
from tests.utils import TestFailureError, bert_root_dir


configs = [
    "tests/configs/mk2/pretrain_nightly_pipeline.json",
]


def check_output(*args, **kwargs):
    try:
        out = subprocess.check_output(
            *args, cwd=bert_root_dir(), stderr=subprocess.PIPE, **kwargs)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


def unique_words(lines):
    vocab = set(chain(*(line.split() for line in lines if line)))
    vocab.update(["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    return sorted(vocab)


def generate_vocab(input_file, vocab_file):
    with open(input_file, 'r') as f:
        vocab = unique_words(f)
        with open(vocab_file, 'w') as o:
            for word in vocab:
                o.write("%s\n" % word)
    return vocab_file


def create_sample_text(input_file, output_file, num_lines=20):
    with open(output_file, 'w') as of:
        with open(input_file, 'r') as f:
            of.writelines(islice(f, num_lines))
    return output_file


def generate_sample_input_files(input_file, workdir):
    sample_text_file_name = os.path.join(workdir, "test_sample_text_gen.txt")
    create_sample_text(input_file, sample_text_file_name)
    vocab_file_name = os.path.join(workdir, "vocab_gen.txt")
    generate_vocab(sample_text_file_name, vocab_file_name)
    generated_dataset_file = os.path.join(workdir, "sample_dataset_gen")
    args = ["python", "bert_data/create_pretraining_data.py",
            "--input-file", sample_text_file_name,
            "--output-file", generated_dataset_file,
            "--vocab-file", vocab_file_name,
            "--do-lower-case", "--sequence-length=32",
            "--mask-tokens=2", "--duplication-factor=1",
            "--pad-position-value=32", "--max-samples=20"
            ]
    check_output(args)
    return generated_dataset_file + "_0"


@pytest.fixture(scope="module")
def generated_sample_input_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_file = generate_sample_input_files(
            os.path.join(bert_root_dir(), "bert_data/sample_text.txt"),
            tmpdirname)
        yield input_file


@pytest.mark.parametrize("config", configs)
def test_pretraining(custom_ops, generated_sample_input_file, config):
    args = ["python", "bert.py", "--config",
            config,
            f"--input-files={generated_sample_input_file}",
            "--device-connection-type=ondemand",
            "--seed=1984"
            ]
    output_strs = str(check_output(args)).split("\\n")
    output_strs = output_strs[-4:]
    matches = re.search(r"Accuracy \(MLM NSP\): ([\d\.]+) ([\d\.]+)", output_strs[0])
    if matches is None:
        print(output_strs)
        raise TestFailureError("Unexpected output format")
    elif matches.groups()[0] != '1.000' or matches.groups()[1] != '1.000':
        raise TestFailureError("Unexpected Accuracy (MLM NSP):"+str(matches.groups()))
