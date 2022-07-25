# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse

from scripts.dataset_benchmark import estimate_ds_throughput
from tests.utils import get_app_root_dir
from utilities.argparser import add_arguments, combine_config_file_with_args
from utilities.options import Options


def test_benchmark_output():
    parser = argparse.ArgumentParser(description="Dataset benchmark test")
    test_dir = get_app_root_dir().joinpath("tests")
    args = add_arguments(parser).parse_args([f"{test_dir}/train_small_graph_sparse.json"])
    config = combine_config_file_with_args(args, Options)

    mean_tput, min_tput, max_tput, _ = estimate_ds_throughput(config)
    assert mean_tput > 0
    assert min_tput > 0
    assert max_tput > 0
