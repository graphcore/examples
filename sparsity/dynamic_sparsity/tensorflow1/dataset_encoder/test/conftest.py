# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import pytest


@pytest.fixture
def gpt2_repo_path(request):
    return request.config.getoption("--gpt2-repo-path")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpt2-repo-path") is not None:
        # --runslow given in cli: do not skip slow tests
        return
    skip_encoder = pytest.mark.skip(reason="need --gpt2-repo-path option to run encoder tests")
    for item in items:
        if "requires_encoder" in item.keywords:
            item.add_marker(skip_encoder)


def pytest_addoption(parser):
    parser.addoption("--gpt2-repo-path", action="store", default=None,
                     help="Path to the GPT-2 repository")
