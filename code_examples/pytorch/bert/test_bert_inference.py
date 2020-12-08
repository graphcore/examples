#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import re
import sys
import subprocess
import unittest

import pytest

EXPECTED_RESULTS = [
    ("How many islands are there in Scotland?", "more than 790"),
    ("What sea is to the south of Scotland?", "irish sea"),
    ("How long is Scotland's border in km?", "154"),
    ("Where is England in relation to scotland?", "southeast")
]


def parse_results(out):
    lines = out.split('\n')
    questions, answers = [], []
    for line in lines:
        match_question = re.match('Question: (.*)', line)
        match_answer = re.match('Answer: (.*)', line)
        if match_question:
            questions.append(match_question.group(1))
        if match_answer:
            answers.append(match_answer.group(1))
    return list(zip(questions, answers))


def run_poptorch_bert_inference(**kwargs):
    cmd = ['python3', './bert_inference.py']
    # Flatten kwargs and convert to strings
    args = [str(item) for sublist in kwargs.items()
            for item in sublist if item != '']
    cmd.extend(args)
    out = subprocess.check_output(
        cmd, cwd=os.path.dirname(__file__)).decode("utf-8")
    return out


class TestPopTorchBERTInference(unittest.TestCase):
    """High-level integration tests for BERT inference in PopTorch"""

    @classmethod
    def setUpClass(cls):
        pass

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_poptorch_bert_batch_size_2(self):
        out = run_poptorch_bert_inference(**{'--batch-size': 2})
        results = parse_results(out)
        # Check both lists match in sizes and contents.
        self.assertEqual(results, EXPECTED_RESULTS)

    @pytest.mark.ipus(2)
    @pytest.mark.category2
    def test_poptorch_bert_batch_size_4(self):
        out = run_poptorch_bert_inference(**{'--batch-size': 4})
        results = parse_results(out)
        # Check both lists match in sizes and contents.
        self.assertEqual(results, EXPECTED_RESULTS)
