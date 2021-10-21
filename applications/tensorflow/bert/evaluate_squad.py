#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import subprocess
from log import logger
import json
import wandb


def evaluate_squad(output_prediction_file, opts):
    if opts["version_2_with_negative"]:
        logger.info("Using SQuAD 2.0 evaluation file")
        SQUAD_TRUTH_PATH = 'data/squad2/dev-v2.0.json'
        SQUAD_EVAL_SCRIPT_PATH = 'data/squad2/evaluate-v2.0.py'
    else:
        logger.info("Using SQuAD 1.1 evaluation file")
        SQUAD_TRUTH_PATH = 'data/squad/dev-v1.1.json'
        SQUAD_EVAL_SCRIPT_PATH = 'data/squad/evaluate-v1.1.py'
    stdout = subprocess.check_output(['python3', SQUAD_EVAL_SCRIPT_PATH, SQUAD_TRUTH_PATH, output_prediction_file])
    em_f1_results = json.loads(stdout)
    if opts["version_2_with_negative"]:
        logger.info(f"SQuAD 2.0 evaluation results: Exact Match: {em_f1_results['exact']:5.2f}, F1: {em_f1_results['f1']:5.2f}")
    else:
        logger.info(f"SQuAD 1.1 evaluation results: Exact Match: {em_f1_results['exact_match']:5.2f}, F1: {em_f1_results['f1']:5.2f}")
    if opts['wandb']:
        for k, v in em_f1_results.items():
            wandb.run.summary[k] = v
