# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest


# Below functions enable long tests to be skipped, unless a --long-test
# cli option is specified.
def pytest_addoption(parser):
    parser.addoption("--long-tests", action="store_true", default=False,
                     help="Run long tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--long-tests"):
        marker = pytest.mark.skip(reason="Skipping this long test, run with "
                                         "--long-tests option to enable it.")
        for item in items:
            if "long_test" in item.keywords:
                item.add_marker(marker)
