# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper

# import numpy as np
import os
import poptorch
import pytest
from triton_server.server_setup import model_repo_opt
from triton_server.utilsTriton import GetModelPath


def export_model(request, configure_bert_model, model_name):

    model_repo_path = GetModelPath(request.config, model_repo_opt)
    popef_file = model_repo_path + "/" + model_name + "/1/executable.popef"

    configure_bert_model.model_ipu.eval()
    poptorch_save_model = poptorch.inferenceModel(configure_bert_model.model_ipu, configure_bert_model.opts)

    input_data = next(configure_bert_model.val_dl.__iter__())

    # stage 1: compile model
    poptorch_save_model(**input_data)

    # stage 2: save model to popef file
    poptorch_save_model.save(popef_file, export_model=False, save_rng_state=True)
    if not os.path.exists(popef_file):
        pytest.fail("Popef file: " + popef_file + " doesn't exist!")
