# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper
from examples_tests.execute_once_per_fs import ExecuteOncePerFS
from model_export import export_model
import pytest
from triton_server.server_setup import *
from utils import download_images, get_models, get_cifar10_dataset, install_turbo_jpeg

test_configs = {"resnet50": "resnet50", "efficientnet-b0": "efficientnet-b0", "efficientnet-b4": "efficientnet-b4"}


@pytest.fixture(scope="session", autouse=True)
@ExecuteOncePerFS(
    lockfile=str(Path(__file__).parent.absolute()) + "/test_environment_ready.lock",
    file_list=[],
    timeout=120,
    retries=20,
)
def initialize_test_environment(request):
    """Get the data required for the tests."""
    get_cifar10_dataset()
    download_images()
    get_models()
    install_turbo_jpeg()

    # export models to popef files
    for model_name, config_name in test_configs.items():
        export_model(request, model_name, config_name)
