# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest


# Below functions enable long tests to be skipped, unless a --long-test
# cli option is specified.
def pytest_addoption(parser):
    parser.addoption("--long-tests", action="store_true", default=False, help="Run long tests")
    parser.addoption(
        "--ipus",
        type=int,
        metavar="N",
        help="only run tests that use N ipus.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--long-tests"):
        marker = pytest.mark.skip(reason="Skipping this long test, run with " "--long-tests option to enable it.")
        for item in items:
            if "long_test" in item.keywords:
                item.add_marker(marker)


def pytest_runtest_setup(item):
    ipus_needed = item.get_closest_marker(name="ipus")
    ipus_needed = ipus_needed.args[0] if ipus_needed else 0
    ipus_specified = item.config.getoption("ipus")
    if ipus_specified is not None and ipus_needed != ipus_specified:
        pytest.skip(f"Uses {ipus_needed} IPUs, not {ipus_specified}")
