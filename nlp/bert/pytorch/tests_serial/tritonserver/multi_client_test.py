# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper
from conftest import test_configs, benchmark_opt
import copy
from multiprocessing import Pool
import pytest
from test_utils import log_performance_results, DataGeneratorWrapper
from triton_server.client import RequestType, task_one_client_one_model, task_one_client_one_model_buffered_data
from triton_server.utilsTriton import PoolLogExceptions
from utils import logger

request_types_params = (RequestType.SYNC, RequestType.ASYNC)
processes = (1, 4, 8)


@pytest.mark.parametrize("number_of_processes", processes)
@pytest.mark.parametrize("request_type", request_types_params)
@pytest.mark.parametrize("model_name,yml_config", test_configs.items())
def test_single_model(
    request,
    triton_server,
    configure_bert_model,
    poptorch_ref_model,
    request_type,
    model_name,
    yml_config,
    number_of_processes,
):
    benchmark_only = request.config.getoption(benchmark_opt)
    if benchmark_only is False and number_of_processes > 1:
        pytest.skip("Multi client test is available only in benchmark mode.")

    logger("Validating...")
    input_names = ("input_ids", "attention_mask", "token_type_ids")
    number_of_outputs = 2

    data_generator = DataGeneratorWrapper(configure_bert_model.val_dl, input_names)

    input_dataset = []
    result_dataset = []

    for input_data, ref_input_data in data_generator:
        input_dataset.append(input_data)
        if not benchmark_only:
            result_dataset.append(poptorch_ref_model(*ref_input_data))

    pool = Pool(processes=number_of_processes)
    tasks = [
        pool.apply_async(
            PoolLogExceptions(task_one_client_one_model_buffered_data),
            args=(
                client_id,
                triton_server.url,
                configure_bert_model.config,
                model_name,
                request_type,
                copy.deepcopy(input_dataset),
                result_dataset,
                number_of_outputs,
                True,
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

    log_performance_results(model_name, request_type, result_data_type, number_of_processes, throughputs, latencies)

    if not benchmark_only:
        test_method = PoolLogExceptions(task_one_client_one_model)
        result_serial, throughputs, latencies, result_data_type = test_method(
            0,
            triton_server.url,
            configure_bert_model.config,
            model_name,
            request_type,
            data_generator,
            number_of_outputs,
            poptorch_ref_model,
        )

        result = result or result_serial
        log_performance_results(model_name, request_type, result_data_type, number_of_processes, throughputs, latencies)

    assert result
