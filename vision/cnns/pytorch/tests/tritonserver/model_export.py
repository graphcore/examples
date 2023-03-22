# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper
import datasets
import models
import numpy as np
import os

# from pathlib import Path
import poptorch
import pytest
from test_utils import get_model_settings
from triton_server.server_setup import model_repo_opt
from triton_server.utilsTriton import GetModelPath


def export_model(request, model_name, yml_config):
    pargs = ("--config", yml_config)
    args, opts = get_model_settings(pargs)

    model_repo_path = GetModelPath(request.config, model_repo_opt)
    popef_file = model_repo_path + "/" + model_name + "/1/executable.popef"

    dataloader = datasets.get_data(args, opts, train=False, async_dataloader=True)

    model = models.get_model(
        args, datasets.datasets_info[args.data], pretrained=not args.random_weights, inference_mode=True
    )
    poptorch_save_model = poptorch.inferenceModel(model, opts)

    input_data = next(dataloader.__iter__())[0]

    # stage 1: calculate reference output
    ref_out = poptorch_save_model(input_data)

    # stage 2: compileAndExport
    poptorch_save_model.compileAndExport(popef_file, input_data, export_model=False)
    if not os.path.exists(popef_file):
        pytest.fail("Popef file: " + popef_file + " doesn't exist!")

    # stage 3: load executable
    poptorch_load_model = poptorch.inferenceModel(model, opts)
    poptorch_load_model.loadExecutable(popef_file)

    # stage 4: calculate output from loaded model
    load_out = poptorch_load_model(input_data)

    # stage 5: compare results
    if not np.allclose(ref_out.numpy(), load_out.numpy(), rtol=1e-5):
        pytest.fail("ref_out is different than load_out.")
