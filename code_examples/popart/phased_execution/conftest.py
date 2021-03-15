# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import subprocess


def remote_buffers_available():
    output = subprocess.run(["gc-info -d 0 -i | grep 'remote buffers: 1'"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            shell=True,
                            check=False)
    return output.returncode == 0


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_remote_buffers" in item.keywords and not remote_buffers_available():
            item.add_marker(pytest.mark.skip(
                reason="Requires remote buffers to be enabled on this system."))
