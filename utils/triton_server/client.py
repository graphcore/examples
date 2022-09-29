# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import import_helper
from collections import namedtuple
from enum import Enum
from functools import partial
import logging
import numpy as np
import pytest
import time
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
from .utilsTriton import Timeout


def prepare_triton_client(url):
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=False)
    except Exception:  # pylint: disable=broad-except
        pytest.fail("Unexpected error during inference client creation")
    else:
        assert triton_client.is_server_ready()
    return triton_client


ClientInput = namedtuple('ClientInput', ['model_name', 'infer_input'])


class RequestType(Enum):
    SYNC = 1
    ASYNC = 2


class Client:
    def __init__(self, index, url, data_generator=None, reference_generator=None):
        self.triton_client = prepare_triton_client(url)
        self.data_generator = data_generator
        self.reference_generator = reference_generator
        self.result_data_type = None
        self.input_queue = []
        self.sample_sizes = []
        self.elapsed_times = []
        logging.info(f"Created Triton client with index {index}")

    def infer_data_item(self, data_item, need_numpy_conversion=False):
        if not isinstance(data_item, list):
            data_item = [data_item]

        inputs = []
        inp_size = 0
        for input_idx, (data) in enumerate(data_item):
            if need_numpy_conversion:
                data = data.numpy()
            inp_size = data.shape[0]
            name = "input_" + str(input_idx)
            infer_input = grpcclient.InferInput(name, data.shape,
                                                np_to_triton_dtype(data.dtype))
            infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)
        self.sample_sizes.append(inp_size)
        return inputs

    def infer_on_the_fly(self, request_type, model_name, number_of_outputs, report_performance=True):
        if self.data_generator is None:
            pytest.fail(
                "Cann't run test with data generated on the fly without data generator!")
        if request_type == RequestType.SYNC:
            return self.infer_requests_sync(model_name, number_of_outputs, report_performance)

        return self.infer_requests_async(model_name, number_of_outputs)

    def infer_buffered(self, request_type, model_name, input_data, number_of_outputs, report_performance=True):
        for data in input_data:
            inferred_input = self.infer_data_item(data)
            self.input_queue.append(ClientInput(model_name, inferred_input))

        if request_type == RequestType.SYNC:
            return self.infer_requests_sync_buffered(number_of_outputs, report_performance)

        return self.infer_requests_async_buffered(number_of_outputs)

    def infer_requests_sync(self, model_name, number_of_outputs, report_performance):
        cmp_res = True
        for data_index, (infer_item, ref_gen_item) in enumerate(self.data_generator):
            infer_input = self.infer_data_item(infer_item)
            outputs = [grpcclient.InferRequestedOutput(
                'output_' + str(out_idx)) for out_idx in range(number_of_outputs)]
            start_time = time.perf_counter()
            infer_result = self.triton_client.infer(
                model_name=model_name,
                inputs=infer_input,
                outputs=outputs)
            elapsed_time = time.perf_counter() - start_time
            self.elapsed_times.append(elapsed_time)
            if report_performance:
                logging.info(
                    f"Iteration: {data_index}: "
                    f"Throughput: {self.sample_sizes[data_index] / elapsed_time} "
                    f"samples/sec; Elapsed time: {elapsed_time} s, {1000*elapsed_time} "
                    f"ms, sample size: {self.sample_sizes[data_index]}")
            result = [infer_result.as_numpy(
                'output_' + str(out_idx)) for out_idx in range(number_of_outputs)]
            if self.reference_generator is not None:
                expected = self.reference_generator(*ref_gen_item)
                if isinstance(expected, list) or isinstance(expected, tuple):
                    for res, expB in zip(result, expected):
                        exp = expB.numpy()[:res.shape[0], :]
                        cmp_res = cmp_res and np.allclose(res, exp, rtol=1e-5)
                else:
                    exp = expected.numpy()[:result[0].shape[0], :]
                    cmp_res = cmp_res and np.allclose(
                        result[0], exp, rtol=1e-5)
            if not cmp_res:
                print("data error at index: ", data_index)
        self.result_data_type = result[0].dtype
        return cmp_res

    def infer_requests_sync_buffered(self, number_of_outputs, report_performance):
        results = []
        if report_performance:
            sample_size_idx = 0
        while len(self.input_queue) > 0:
            client_input = self.input_queue.pop(0)
            outputs = [grpcclient.InferRequestedOutput(
                'output_' + str(out_idx)) for out_idx in range(number_of_outputs)]
            start_time = time.perf_counter()
            infer_result = self.triton_client.infer(
                model_name=client_input.model_name,
                inputs=client_input.infer_input,
                outputs=outputs)
            elapsed_time = time.perf_counter() - start_time
            self.elapsed_times.append(elapsed_time)
            if report_performance:
                logging.info(
                    f"Iteration: {sample_size_idx}: "
                    f"throughput: {self.sample_sizes[sample_size_idx] / elapsed_time} "
                    f"samples/sec; Elapsed time: {elapsed_time} s, {1000*elapsed_time} "
                    f"ms, sample size: {self.sample_sizes[sample_size_idx]}")
                sample_size_idx = sample_size_idx + 1
            iteration_result = []
            for out_idx in range(number_of_outputs):
                iteration_result.append(
                    infer_result.as_numpy('output_' + str(out_idx)))
            results.append(iteration_result)
        return results

    def infer_requests_async(self, model_name, number_of_outputs):
        cmp_res = True
        results = []
        start_timestamps = []
        end_timestamps = []

        def callback(results, index, result, error):
            end_timestamps.insert(index, time.perf_counter())
            if error:
                results.insert(index, error)
            else:
                results.insert(index, [result.as_numpy(
                    'output_' + str(out_idx)) for out_idx in range(number_of_outputs)])

        requests_number = len(self.input_queue)

        for data_index, (data_item, _) in enumerate(self.data_generator):
            infer_input = self.infer_data_item(data_item)
            outputs = [grpcclient.InferRequestedOutput(
                'output_' + str(out_idx)) for out_idx in range(number_of_outputs)]
            start_timestamps.append(time.perf_counter())
            self.triton_client.async_infer(model_name=model_name,
                                           inputs=infer_input,
                                           callback=partial(
                                               callback, results, data_index),
                                           outputs=outputs)

        with Timeout(seconds=360, error_message="Async request timeout"):
            while len(results) < requests_number:
                time.sleep(0.00001)
        if self.reference_generator is not None:
            for resList, (_, ref_gen_item) in zip(results, self.data_generator):
                expected = self.reference_generator(*ref_gen_item)
                if isinstance(expected, list) or isinstance(expected, tuple):
                    for res, expB in zip(resList, expected):
                        exp = expB.numpy()[:res.shape[0], :]
                        cmp_res = cmp_res and np.allclose(res, exp, rtol=1e-5)
                else:
                    exp = expected.numpy()[:resList[0].shape[0], :]
                    cmp_res = cmp_res and np.allclose(
                        resList[0], exp, rtol=1e-5)
        self.result_data_type = results[0][0].dtype
        self.elapsed_times = [end - start for start,
                              end in zip(start_timestamps, end_timestamps)]
        self.input_queue.clear()
        return cmp_res

    def infer_requests_async_buffered(self, number_of_outputs):
        results = []
        start_timestamps = []
        end_timestamps = []

        def callback(results, index, result, error):
            end_timestamps.insert(index, time.perf_counter())
            if error:
                results.insert(index, error)
            else:
                outputs = []
                for out_idx in range(number_of_outputs):
                    outputs.append(result.as_numpy('output_' + str(out_idx)))
                results.insert(index, outputs)

        requests_number = len(self.input_queue)

        for index, (client_input) in enumerate(self.input_queue):
            outputs = [grpcclient.InferRequestedOutput(
                'output_' + str(out_idx)) for out_idx in range(number_of_outputs)]
            start_timestamps.append(time.perf_counter())
            self.triton_client.async_infer(model_name=client_input.model_name,
                                           inputs=client_input.infer_input,
                                           callback=partial(
                                               callback, results, index),
                                           outputs=outputs)

        with Timeout(seconds=360, error_message="Async request timeout"):
            while len(results) < requests_number:
                time.sleep(0.00001)
        self.elapsed_times = [end - start for start,
                              end in zip(start_timestamps, end_timestamps)]
        self.input_queue.clear()
        return results

    def get_perf_data(self, batch_size):
        # If possible, remove the first few throughput measurements to eliminate startup overhead.
        if len(self.elapsed_times) > 2:
            elapsed_times = self.elapsed_times[2:]
            sample_sizes = self.sample_sizes[2:]
        else:
            elapsed_times = self.elapsed_times
            sample_sizes = self.sample_sizes

        throughputs = [s / t for s, t in zip(sample_sizes, elapsed_times)]
        latencies_in_ms = [1000 * t * batch_size /
                           s for s, t in zip(sample_sizes, elapsed_times)]
        return throughputs, latencies_in_ms


def task_one_client_one_model(process_idx, url, model_cfg_args, model_name, request_type,
                              data_loader, number_of_outputs, reference_generator=None,
                              report_performance=True):
    infer_client = Client(process_idx, url, data_loader, reference_generator)

    results_check = infer_client.infer_on_the_fly(
        request_type, model_name, number_of_outputs, report_performance)

    throughputs, latencies_in_ms = infer_client.get_perf_data(
        model_cfg_args.micro_batch_size)

    return results_check, throughputs, latencies_in_ms, infer_client.result_data_type


def task_one_client_one_model_buffered_data(process_idx, url, model_cfg_args, model_name, request_type,
                                            input_data, expected_result, number_of_outputs,
                                            report_performance=False):
    infer_client = Client(process_idx, url)

    results = infer_client.infer_buffered(
        request_type, model_name, input_data, number_of_outputs, report_performance=report_performance)

    if expected_result is None or len(expected_result) == 0:
        cmp_res = True
    else:
        logging.info(f"Checking results for process with index: {process_idx}")
        cmp_res = len(results) == len(expected_result)
        for resList, expList in zip(results, expected_result):
            for res, expB in zip(resList, expList):
                exp = expB.numpy()[:res.shape[0], :]
                cmp_res = cmp_res and np.allclose(res, exp, rtol=1e-5)

    throughputs, latencies_in_ms = infer_client.get_perf_data(
        model_cfg_args.micro_batch_size)

    return cmp_res, throughputs, latencies_in_ms, results[0][0].dtype
