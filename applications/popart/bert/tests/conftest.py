# Copyright 2019 Graphcore Ltd.
import subprocess
from pathlib import Path
import pytest
import ctypes

custom_ops_loaded = False


def load_custom_ops():
    bert_dir = Path(__file__).parent.parent.resolve()
    so_path = str(bert_dir / "custom_ops.so")
    global custom_ops_loaded
    if not custom_ops_loaded:
        print("Building Custom Ops")
        subprocess.run(['make'], shell=True, cwd=str(bert_dir))
        ctypes.cdll.LoadLibrary(so_path)
        custom_ops_loaded = True
    return so_path


@pytest.fixture
def custom_ops():
    return load_custom_ops()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_config" in item.keywords and not config.getoption("--config-path"):
            item.add_marker(pytest.mark.skip(
                reason="Requires a config-graph path to run"))
        if "requires_chkpt" in item.keywords and not config.getoption("--chkpt-path"):
            item.add_marker(pytest.mark.skip(
                reason="Requires a chkpt-graph path to run"))
        if "requires_frozen" in item.keywords and not config.getoption("--frozen-path"):
            item.add_marker(pytest.mark.skip(
                reason="Requires a frozen-graph path to run"))


def pytest_addoption(parser):
    parser.addoption(
        "--config-path",
        action="store",
        help="Path to a file containing the BERT configuration parameters.")

    parser.addoption(
        "--chkpt-path",
        action="store",
        help="Path to the tensorflow checkpoint file.")

    parser.addoption(
        "--frozen-path",
        action="store",
        help="Path to the frozen graph file.")

    parser.addoption(
        "--chkpt-task",
        type="choice",
        choices=("PRETRAINING, SQUAD"),
        default="PRETRAINING",
        help="Checkpoint task - Pretraining or SQuAD.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_config: Skip if fixture config_path not provided")
    config.addinivalue_line(
        "markers", "requires_chkpt: Skip if fixture chkpt_path not provided")
    config.addinivalue_line(
        "markers", "requires_frozen: Skip if fixture frozen_path not provided")


@pytest.fixture
def config_path(request):
    return request.config.getoption("--config-path")


@pytest.fixture
def chkpt_path(request):
    return request.config.getoption("--chkpt-path")


@pytest.fixture
def chkpt_task(request):
    return request.config.getoption("--chkpt-task")


@pytest.fixture
def frozen_path(request):
    return request.config.getoption("--frozen-path")
