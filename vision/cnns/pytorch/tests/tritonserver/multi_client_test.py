# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper
import copy
import datasets
import models
from multiprocessing import Pool
import os
import poptorch
import pytest
from triton_server.client import RequestType, task_one_client_one_model, task_one_client_one_model_buffered_data
from triton_server.utilsTriton import PoolLogExceptions
from tritonserver.conftest import test_configs, benchmark_opt
from test_utils import get_model_settings, log_performance_results, DataGeneratorWrapper

request_types_params = (RequestType.SYNC, RequestType.ASYNC)
processes = (1, 4, 8)


@pytest.mark.xdist_group(name="triton")
@pytest.mark.parametrize("number_of_processes", processes)
@pytest.mark.parametrize("request_type", request_types_params)
@pytest.mark.parametrize("model_name,yml_config", test_configs.items())
def test_single_model(request, triton_server, request_type, model_name, yml_config, number_of_processes):
    benchmark_only = request.config.getoption(benchmark_opt)

    if benchmark_only is False and number_of_processes > 1:
        pytest.skip("Multi client test is available only in benchmark mode.")

    pargs = ("--config", yml_config)
    args, opts = get_model_settings(pargs)

    if not benchmark_only:
        args.dataloader_worker = 2
        popef_file = triton_server.model_repo_path + "/" + model_name + "/1/executable.popef"
        if not os.path.exists(popef_file):
            pytest.fail("Popef file: " + popef_file + " doesn't exist!")

        model = models.get_model(
            args, datasets.datasets_info[args.data], pretrained=not args.random_weights, inference_mode=True
        )
        poptorch_ref_model = poptorch.inferenceModel(model, opts)
        poptorch_ref_model.loadExecutable(popef_file)
    else:
        poptorch_ref_model = None
        # intentionally reduce work size to single batch size when throughput and latency is measured
        args.device_iterations = 1
        opts.deviceIterations(args.device_iterations)

    dataloader = datasets.get_data(args, opts, train=False, async_dataloader=not benchmark_only)
    data_generator = DataGeneratorWrapper(dataloader)

    number_of_outputs = 1

    if benchmark_only:
        input_dataset = []
        result_dataset = []
        for input_data, ref_input in data_generator:
            input_dataset.append(input_data)

        pool = Pool(processes=number_of_processes)
        tasks = [
            pool.apply_async(
                PoolLogExceptions(task_one_client_one_model_buffered_data),
                args=(
                    client_id,
                    triton_server.url,
                    args,
                    model_name,
                    request_type,
                    copy.deepcopy(input_dataset),
                    result_dataset,
                    number_of_outputs,
                    benchmark_only,
                ),
            )
            for client_id in range(number_of_processes)
        ]
        results = [task.get()[0] for task in tasks]
        throughputs = [task.get()[1] for task in tasks]
        latencies = [task.get()[2] for task in tasks]
        result_data_type = tasks[0].get()[3]
        pool.close()
        pool.join()
        result = all(results)
    else:
        test_method = PoolLogExceptions(task_one_client_one_model)
        result, throughputs, latencies, result_data_type = test_method(
            0, triton_server.url, args, model_name, request_type, data_generator, number_of_outputs, poptorch_ref_model
        )
    log_performance_results(
        args, model_name, request_type, result_data_type, number_of_processes, throughputs, latencies
    )
    assert result
