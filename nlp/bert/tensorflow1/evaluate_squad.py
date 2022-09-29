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
import datasets


def evaluate_squad(output_prediction_file, opts):
    if opts["version_2_with_negative"]:
        logger.info("Using SQuAD 2.0 evaluation file")
        squad_metric = datasets.load_metric("squad_v2")
        squad_references = [
            {"id": sample["id"], "answers": sample["answers"]}
            for sample in datasets.load_dataset("squad_v2")["validation"]
            ]
        with open(output_prediction_file) as prediction_file:
            predictions = [
                {"id": k,
                 "prediction_text": v["prediction_text"],
                 "no_answer_probability": v["no_answer_probability"]}
                for k,v in json.load(prediction_file).items()]
    else:
        logger.info("Using SQuAD 1.1 evaluation file")
        squad_metric = datasets.load_metric("squad")
        squad_references = [
            {"id": sample["id"], "answers": sample["answers"]}
            for sample in datasets.load_dataset("squad")["validation"]
            ]        
        with open(output_prediction_file) as prediction_file:
            predictions = [
                {"id": k, "prediction_text": v}
                for k,v in json.load(prediction_file).items()]
    em_f1_results = squad_metric.compute(
        predictions=predictions, references=squad_references)

    if opts["version_2_with_negative"]:
        logger.info(f"SQuAD 2.0 evaluation results: Exact Match: {em_f1_results['exact']:5.2f}, F1: {em_f1_results['f1']:5.2f}")
    else:
        logger.info(f"SQuAD 1.1 evaluation results: Exact Match: {em_f1_results['exact_match']:5.2f}, F1: {em_f1_results['f1']:5.2f}")
    if opts['wandb']:
        for k, v in em_f1_results.items():
            wandb.run.summary[k] = v
