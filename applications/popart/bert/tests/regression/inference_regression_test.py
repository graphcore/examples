# Copyright 2019 Graphcore Ltd.

import logging
import os
from pathlib import Path
import numpy as np
import pytest
import json

from bert import main
from utils import parse_bert_args


def get_tests_to_run():
    return [get_config("squad_base_inference.json", False, ["--tf-checkpoint",
                                                            str(get_bert_root_folder() / "data/bert_tf_5474/model.ckpt-5474")]),
            get_config("squad_large_inference.json", True)
            ]


def get_bert_root_folder():
    return Path(__file__).parent.parent.parent


def check_required_files(args):

    assert os.path.isfile(args.vocab_file)
    assert os.path.isfile(args.squad_evaluate_script)
    assert os.path.isdir(args.squad_results_dir)

    if args.input_files:
        for file_ in args.input_files:
            assert os.path.isfile(file_)

    if args.tf_checkpoint:
        assert os.path.isdir(Path(args.tf_checkpoint).parent)


def get_config(filename, synthetic=False, extra_args=None, uid=None):
    path = get_bert_root_folder() / "configs" / filename
    path = path.resolve()
    if uid is None:
        uid = path.stem
    return str(path), synthetic, extra_args, uid


def get_baseline_metrics(tf_checkpoint_path):
    """
    Scan the tf checkpoint folder.
    Then, read f1 and exact match from log file.
    """

    with os.scandir(Path(tf_checkpoint_path).parent) as files:
        log_file = ""
        for file_ in files:
            if file_.name.endswith('.log'):
                log_file = file_
                break

    assert(os.path.isfile(log_file))

    with open(log_file, 'r') as lf:
        lines = reversed(lf.readlines())

    for line in lines:
        if "f1" in line:
            line_json = json.loads(line)
            F1 = line_json['f1']
            exact_match = line_json['exact_match']
            break

    return F1, exact_match


def get_test_metrics(caplog):
    for record in reversed(caplog.records):
        if "F1" in record.message:
            message_split = record.message.split()
            F1 = float(message_split[2])
            exact_match = float(message_split[6])
            break

    return F1, exact_match


@pytest.mark.parametrize(
    "bert_config_file,synthetic,extra_args,uid",
    get_tests_to_run(),
    ids=list(map(lambda test: test[3], get_tests_to_run())))
def test_bert_inference_squad_regression(caplog, custom_ops, output_path,
                                         bert_config_file, synthetic,
                                         extra_args, uid, synthetic_steps=100):
    """
    Run an inference test of BERT, that will check performance against
    known metrics data.
    """

    # Build argument list, TODO remove extra flags when SDK fixed
    args_string = ["--config",
                   bert_config_file,
                   "--no-outlining",
                   "--no-drop-remainder",
                   "--no-model-save",
                   "--squeeze-model=false"
                   ]
    if synthetic:
        args_string.append("--synthetic")
        args_string += ["--epochs", str(synthetic_steps)]
        args_string += ["--aggregate-metrics-over-steps", str(synthetic_steps)]
    if extra_args is not None:
        args_string += extra_args
    args = parse_bert_args(args_string)

    # Early checks before running session
    os.makedirs(output_path, exist_ok=True)
    check_required_files(args)
    if args.tf_checkpoint:
        base_F1, base_exact_match = get_baseline_metrics(args.tf_checkpoint)
        caplog.set_level(logging.INFO)

    # Run inference session
    session, iteration = main(args)

    # Compare results with baseline if provided and destroy logs
    if args.tf_checkpoint:
        current_F1, current_exact_match = get_test_metrics(caplog)
        assert(np.isclose(current_exact_match, base_exact_match,
                          rtol=1e-2, atol=1e-1))
        assert(np.isclose(current_F1, base_F1,
                          rtol=1e-2, atol=1e-1))
        caplog.clear()
