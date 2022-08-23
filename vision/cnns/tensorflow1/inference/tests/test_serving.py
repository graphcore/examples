# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import pytest
import subprocess
import os


def test_efficientnet_serving():
    app_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), os.pardir)
    result = subprocess.run(['bash', 'get_images.sh'], cwd=app_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        pytest.fail(str(result.stdout.decode("utf-8")))

    test_params = ['python3', os.path.join(app_dir, 'send_request.py'), 'efficientnet-s', 'images',
                   '--batch-size', '8', '--model-batch-size', '16', '--port', '8502', '--num-threads', '32']
    result = subprocess.run(test_params, cwd=app_dir,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        pytest.fail(str(result.stdout.decode("utf-8")))
