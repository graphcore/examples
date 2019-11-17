# Copyright 2019 Graphcore Ltd.
import os
import json
import numpy as np
from pathlib import Path
import pytest
from typing import NamedTuple

import popart
import onnx

from bert import main
from bert_model import Bert, BertConfig
from bert_optimizer import ScheduledOptimizerFactory
from utils import parse_bert_args

from tests.utils import (
    run_py,
    extract_initializers,
    check_tensors,
    check_onnx_model)
from tests.regression.RegressionTestResult import (
    BaselineRegressionResult,
    RegressionResult,
    OutOfMemoryTestResult,
    ResultStatus)


def get_test_baseline(uid):
    """Read in the baseline from the file-system.
    TODO: For now this is mocked using the current benchmark figures.
    Once the test goes into the CI, a method of passing in these values
    should be established with the infrastructure team.
    """
    return None


def get_accuracy_stats(args, iteration):
    if not args.inference and not args.synthetic_data:
        if args.task == "PRETRAINING":
            accuracies = np.mean(np.stack([
                iteration.mlm_accuracies,
                iteration.nsp_accuracies,
            ], axis=0), axis=0)
        else:
            accuracies = iteration.accuracies
    else:
        accuracies = None
    return accuracies


def config(filename, synthetic=False, extra_args=None, uid=None):
    path = Path(__file__).parent.parent.parent / "configs" / filename
    path = path.resolve()
    if uid is None:
        uid = path.stem
    return str(path), synthetic, extra_args, uid

TESTS = [
    config("demo.json", uid="demo_default"),
    config("demo.json", False, ["--pipeline"], "demo_pipelined"),
    config("squad_base.json", True),
    config("squad_large.json", True),
    config("squad_base_inference.json", True),
    config("squad_large_inference.json", True),
    config("squad_base_384.json", True)
]


@pytest.mark.parametrize(
    "bert_config_file,synthetic,extra_args,uid",
    TESTS,
    ids=list(map(lambda test: test[3], TESTS)))
def test_bert_regression(custom_ops, output_path,
                         bert_config_file, synthetic, extra_args, uid,
                         synthetic_steps=100):
    """
    Run a pretraining pass of BERT up to the specified number of epochs.

    This test will gather a number of statistics and assert that performance
    hasn't dropped substantially (with 10% leeway in some cases).

    `utils.run_py` only carries out a single step, we need to run multiple
    epochs to check accuracy, so this is based on the training example.
    """
    # We'll try to create the output path straight-away so as not to waste
    # time if we get an error
    os.makedirs(output_path, exist_ok=True)

    args_string = ["--config", bert_config_file, "--no-validation", "--no-model-save"]
    if synthetic:
        args_string.append("--synthetic")
        args_string += ["--epochs", str(synthetic_steps)]
        args_string += ["--aggregate-metrics-over-steps", str(synthetic_steps)]
    if extra_args is not None:
        args_string += extra_args
    args = parse_bert_args(args_string)
    session, iteration = main(args)

    # Graph report statistics
    graph_report = json.loads(session.getGraphReport())
    max_tile_memory = max(graph_report["memory"]["byTile"]["total"])
    total_memory = int(np.sum(graph_report["memory"]["byTile"]["total"]))

    baseline_result = get_test_baseline(uid)

    accuracies = get_accuracy_stats(args, iteration)

    # TODO: Add epochs_to_full back in.
    result = RegressionResult(args.input_files, baseline_result,
                              accuracies,
                              total_memory,
                              max_tile_memory,
                              iteration.throughput)

    result.write(output_path, uid)

    # Could probably roll these into a single check for success, but for now
    # this will cause the reason of the failure to be line-highlighted
    assert (not result.status & ResultStatus.FAILED_ACCURACY)
    assert (not result.status & ResultStatus.FAILED_MEM_USAGE)
    assert (not result.status & ResultStatus.FAILED_TILE_MEM)
    assert (not result.status & ResultStatus.FAILED_THROUGHPUT)
