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
    def __init__(self, url, data_generator=None, reference_generator=None):
        self.triton_client = prepare_triton_client(url)
        self.data_generator = data_generator
        self.reference_generator = reference_generator
        self.result_data_type = None
        self.input_queue = []
        self.sample_sizes = []
        self.elapsed_times = []

    def enqueue_request_numpy(self, model_name, input_data, data_type="FP32"):
        inp_size = input_data.shape[0]
        self.sample_sizes.append(inp_size)
        infer_input = grpcclient.InferInput('input', input_data.shape,
                                            data_type)
        infer_input.set_data_from_numpy(input_data)
        self.input_queue.append(ClientInput(model_name, infer_input))

    def infer_data_item(self, data_item):
        data = data_item.numpy()
        inp_size = data.shape[0]
        self.sample_sizes.append(inp_size)
        infer_input = grpcclient.InferInput('input', data.shape,
                                            np_to_triton_dtype(data.dtype))
        infer_input.set_data_from_numpy(data)
        return infer_input

    def infer_on_the_fly(self, request_type, model_name, report_performance=True):
        if self.data_generator is None:
            pytest.fail(
                "Cann't run test with data generated on the fly without data generator!")
        if request_type == RequestType.SYNC:
            return self.infer_requests_sync(model_name, report_performance)

        return self.infer_requests_async(model_name)

    def infer_buffered(self, request_type, model_name, numpy_input_data, report_performance=True):
        for data in numpy_input_data:
            self.enqueue_request_numpy(
                model_name, data, np_to_triton_dtype(data.dtype))
        if request_type == RequestType.SYNC:
            return self.infer_requests_sync_buffered(report_performance)

        return self.infer_requests_async_buffered()

    def infer_requests_sync(self, model_name, report_performance):
        cmp_res = True
        for data_index, (data_item, _) in enumerate(self.data_generator):
            infer_input = self.infer_data_item(data_item)
            outputs = [grpcclient.InferRequestedOutput('output')]
            start_time = time.perf_counter()
            infer_result = self.triton_client.infer(
                model_name=model_name,
                inputs=[infer_input],
                outputs=outputs)
            elapsed_time = time.perf_counter() - start_time
            self.elapsed_times.append(elapsed_time)
            if report_performance:
                logging.info(
                    f"Throughput: {self.sample_sizes[data_index] / elapsed_time} "
                    f"imgs/sec; Elapsed time: {elapsed_time} s, {1000*elapsed_time} "
                    f"ms, sample size: {self.sample_sizes[data_index]}")
            result = infer_result.as_numpy('output')
            if self.reference_generator is not None:
                expected = self.reference_generator(data_item)
                exp = expected.numpy()[:result.shape[0], :]
                cmp_res = cmp_res and np.allclose(result, exp, rtol=1e-5)
        self.result_data_type = result.dtype
        return cmp_res

    def infer_requests_sync_buffered(self, report_performance):
        results = []
        if report_performance:
            sample_size_idx = 0
        while len(self.input_queue) > 0:
            client_input = self.input_queue.pop(0)
            outputs = [grpcclient.InferRequestedOutput('output')]
            start_time = time.perf_counter()
            infer_result = self.triton_client.infer(
                model_name=client_input.model_name,
                inputs=[client_input.infer_input],
                outputs=outputs)
            elapsed_time = time.perf_counter() - start_time
            self.elapsed_times.append(elapsed_time)
            if report_performance:
                logging.info(
                    f"Throughput: {self.sample_sizes[sample_size_idx] / elapsed_time} "
                    f"imgs/sec; Elapsed time: {elapsed_time} s, {1000*elapsed_time} "
                    f"ms, sample size: {self.sample_sizes[sample_size_idx]}")
                sample_size_idx = sample_size_idx + 1
            results.append(infer_result.as_numpy('output'))
        return results

    def infer_requests_async(self, model_name):
        cmp_res = True
        results = []
        start_timestamps = []
        end_timestamps = []

        def callback(results, index, result, error):
            end_timestamps.insert(index, time.perf_counter())
            if error:
                results.insert(index, error)
            else:
                results.insert(index, result.as_numpy('output'))

        requests_number = len(self.input_queue)

        for data_index, (data_item, _) in enumerate(self.data_generator):
            infer_input = self.infer_data_item(data_item)
            outputs = [grpcclient.InferRequestedOutput('output')]
            start_timestamps.append(time.perf_counter())
            self.triton_client.async_infer(model_name=model_name,
                                           inputs=[infer_input],
                                           callback=partial(
                                               callback, results, data_index),
                                           outputs=outputs)

        with Timeout(seconds=360, error_message="Async request timeout"):
            while len(results) < requests_number:
                time.sleep(0.00001)
        if self.reference_generator is not None:
            for result, (data_item, _) in zip(results, self.data_generator):
                expected = self.reference_generator(data_item)
                exp = expected.numpy()[:result.shape[0], :]
                cmp_res = cmp_res and np.allclose(result, exp, rtol=1e-5)
        self.result_data_type = results[0].dtype
        self.elapsed_times = [end - start for start,
                              end in zip(start_timestamps, end_timestamps)]
        self.input_queue.clear()
        return cmp_res

    def infer_requests_async_buffered(self):
        results = []
        start_timestamps = []
        end_timestamps = []

        def callback(results, index, result, error):
            end_timestamps.insert(index, time.perf_counter())
            if error:
                results.insert(index, error)
            else:
                results.insert(index, result.as_numpy('output'))

        requests_number = len(self.input_queue)

        for index, (client_input) in enumerate(self.input_queue):
            outputs = [grpcclient.InferRequestedOutput('output')]
            start_timestamps.append(time.perf_counter())
            self.triton_client.async_infer(model_name=client_input.model_name,
                                           inputs=[client_input.infer_input],
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
        # Remove the first few throughput measurements to eliminate startup overhead.
        elapsed_times = self.elapsed_times[2:]
        sample_sizes = self.sample_sizes[2:]

        throughputs = [s / t for s, t in zip(sample_sizes, elapsed_times)]
        latencies_in_ms = [1000 * t * batch_size /
                           s for s, t in zip(sample_sizes, elapsed_times)]
        return throughputs, latencies_in_ms


def task_one_client_one_model(url, model_cfg_args, model_name, request_type,
                              data_loader, reference_generator=None,
                              report_performance=True):
    infer_client = Client(url, data_loader, reference_generator)

    results = infer_client.infer_on_the_fly(
        request_type, model_name, report_performance)

    throughputs, latencies_in_ms = infer_client.get_perf_data(
        model_cfg_args.batch_size)

    return results, throughputs, latencies_in_ms, infer_client.result_data_type


def task_one_client_one_model_buffered_data(process_idx, url, model_cfg_args, model_name, request_type,
                                            input_data, expected_result,
                                            do_benchmark_only=False,
                                            report_performance=False):
    infer_client = Client(url)

    results = infer_client.infer_buffered(
        request_type, model_name, input_data, report_performance=report_performance)

    cmp_res = len(results) == len(expected_result) or do_benchmark_only
    if not do_benchmark_only:
        for res, expB in zip(results, expected_result):
            exp = expB.numpy()[:res.shape[0], :]
            cmp_res = cmp_res and np.allclose(res, exp, rtol=1e-5)

    throughputs, latencies_in_ms = infer_client.get_perf_data(
        model_cfg_args.batch_size)

    return cmp_res, throughputs, latencies_in_ms, results[0].dtype
