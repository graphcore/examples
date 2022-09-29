# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import atexit
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time

import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from batch_config import BatchConfig
from configuration import terminal_argparse
from datasets.dataset_factory import DatasetFactory
from eight_bit_transfer import EightBitTransfer
import precision


class PendingResult(object):

    def __init__(self):
        self.start = 0
        self.stop = 0
        self.result = None
        self.labels = None

    def future_callback(self, future_res):
        self.stop = time.time()
        self.result = future_res.result()


def findServerBin():

    server_dir = None

    if os.environ.get('POPLAR_SDK_PATH') is not None:
        server_dir = os.environ["POPLAR_SDK_PATH"]
        # POPLAR_SDK_PATH uses relative path, so to find serving binary
        # first we need to jump 4 levels up to public_examples_view folder
        server_dir = os.path.abspath(os.path.join(
            os.curdir, os.pardir, os.pardir, os.pardir, os.pardir, server_dir))
    elif os.environ.get('TF_POPLAR_BASE') is not None:
        server_dir = os.path.abspath(os.path.join(
            os.environ["TF_POPLAR_BASE"], os.pardir))
    else:
        sys.exit(
            f"Unable to find SDK location because TF_POPLAR_BASE or POPLAR_SDK_PATH env is not set,"
            " please use --serving-bin-path to point location of tensorflow server binary")

    for file in os.listdir(server_dir):
        if file.startswith("tensorflow_model_server-r2"):
            return os.path.join(server_dir, file)

    sys.exit(f"No serving binary found in {server_dir}")
    return None


def exportModelAndStartServer(tensorflow_server_path, model_config, grpc_port, batch_size, model_batch_size):

    tensorflow_server_path = tensorflow_server_path or findServerBin()
    if tensorflow_server_path is None:
        return None

    model_path = os.path.join(os.getcwd(), model_config)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    export_script_path = os.path.join(
        os.getcwd(), 'scripts', 'export_for_serving.py')
    result = subprocess.call(['python3', export_script_path,
                              '--config', model_config,
                              '--export-dir', os.path.join(model_path, '001'),
                              '--synthetic-data', 'host',
                              '--use-serving-api'], cwd=os.getcwd())
    if result != 0:
        return None

    serving_params = [tensorflow_server_path,
                      f'--model_base_path={model_path}',
                      f'--model_name={model_config}',
                      f'--port={grpc_port}']
    if model_batch_size != batch_size:
        config_dir = os.path.join(os.getcwd(), 'batch.conf')
        with open(config_dir, 'w') as f:
            f.write(
                'max_batch_size { value: ' + str(model_batch_size) + ' }\n')
            f.write('batch_timeout_micros { value: 100000 }\n')
            f.write('max_enqueued_batches { value: 1000000 }\n')
            f.write('num_batch_threads { value: 4 }\n')
            f.write('allowed_batch_sizes : ' +
                    str(model_batch_size))
        serving_params.append('--enable_batching=true')
        serving_params.append(f'--batching_parameters_file={config_dir}')

    return subprocess.Popen(serving_params)


def checkServerStatus(serving_address):
    channel = grpc.insecure_channel(serving_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'dummy_model'
    request.model_spec.signature_name = 'serving_default'
    try:
        stub.Predict(request, 2)
    except grpc.RpcError as e:
        if 'Servable not found for request: Latest(dummy_model)' == e.details():
            return True
        else:
            return False


def send_images(num_images, dataset_iter, batch_size, use_async, request, stub):
    req_results = []
    future_result = None
    for nr in range(int(np.ceil(num_images/batch_size))):

        data, labels = next(dataset_iter)
        raw_data = data.numpy()
        request.inputs['input'].CopyFrom(
            tf.make_tensor_proto(raw_data, shape=data.shape, dtype=data.dtype))

        pending_result = PendingResult()
        pending_result.labels = labels
        pending_result.start = time.time()
        if use_async:
            future_result = stub.Predict.future(request, 10.0)
            req_results.append(pending_result)
            future_result.add_done_callback(pending_result.future_callback)
        else:
            pending_result.result = stub.Predict(request, 10.0)
            pending_result.stop = time.time()
            req_results.append(pending_result)
    if use_async and future_result is not None:
        # Request queue works in FIFO mode, just wait for last one
        future_result.result()
    return req_results


def inference_process(process_index, barrier, hparams, serving_address):

    fp_precision = precision.Precision(hparams.precision)
    fp_precision.apply()

    # Get eight bit transfer object
    eight_bit_transfer = EightBitTransfer(
        fp_precision.compute_precision) if hparams.eight_bit_transfer else None

    batch_config = BatchConfig(micro_batch_size=hparams.request_batch_size,
                               num_replicas=1,
                               gradient_accumulation_count=1)

    # Get the validation dataset
    ds, _, hparams.pipeline_num_parallel = DatasetFactory.get_dataset(
        dataset_name=hparams.dataset,
        dataset_path=hparams.dataset_path,
        split='test',
        img_datatype=fp_precision.compute_precision,
        batch_config=batch_config,
        accelerator_side_preprocess=hparams.accelerator_side_preprocess,
        eight_bit_transfer=eight_bit_transfer,
        pipeline_num_parallel=hparams.pipeline_num_parallel,
        fused_preprocessing=hparams.fused_preprocessing,
        synthetic_data=hparams.synthetic_data)
    logging.debug(ds)

    dataset_iter = iter(ds.pipeline)

    channel = grpc.insecure_channel(serving_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = hparams.config
    request.model_spec.signature_name = 'serving_default'

    # Warmup
    send_images(hparams.request_batch_size*10, dataset_iter, hparams.request_batch_size,
                False, request, stub)

    # wait for other processes
    barrier.wait(10)

    results = send_images(hparams.num_images,  dataset_iter, hparams.request_batch_size,
                          False, request, stub)

    if hparams.verbose and process_index == 0:
        latency = []
        for res in results:
            duration = res.stop - res.start
            print(f"-- Latency for batch: {(duration*1000):2.2f} ms.")
            latency.append(duration*1000)
        print("Latency statistics")
        print("-------------------------------------------------------------------------------------------")
        print(
            f"Latencies - avg:{np.mean(latency)}, min: {np.min(latency)}, max: {np.max(latency)}, ")


def main(hparams):

    serving_address = f"{hparams.host}:{hparams.port}"

    exporter_pid = exportModelAndStartServer(
        hparams.serving_bin_path, hparams.config, hparams.port, hparams.request_batch_size, hparams.micro_batch_size)
    if exporter_pid is None:
        sys.exit("No server found")
    atexit.register(exporter_pid.kill)
    time.sleep(2)
    server_ready = False
    for _ in range(5):
        time.sleep(1)
        server_ready = checkServerStatus(serving_address)
        if server_ready:
            break
    if server_ready is False:
        sys.exit("Timeout: Unable to connect to the server in 5s")

    print("Spawn workers")
    barrier = multiprocessing.Barrier(hparams.num_threads+1)
    processes = []
    for r in range(hparams.num_threads):
        args = (r, barrier, hparams, serving_address)
        proc = multiprocessing.Process(target=inference_process, args=args)
        proc.start()
        processes.append(proc)
    print("Wait for workers")
    barrier.wait(10)
    start_all = time.time()
    print("Sending requests")

    for proc in processes:
        proc.join()

    end_all = time.time()
    print("All done")
    exec_time = (end_all - start_all)
    print("-------------------------------------------------------------------------------------------")
    print("Full time in ms:", exec_time * 1000)
    print(
        f"Processed num_images * num_threads: {hparams.num_images} * {hparams.num_threads} = {hparams.num_images*hparams.num_threads}")
    print(
        f"Average throughput: {((hparams.num_images*hparams.num_threads)/exec_time):2.4f} samples/sec")


if __name__ == '__main__':
    # configure logger
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Serving TF2 classification Models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help="Serving service host, example: 'localhost'")
    parser.add_argument('--port', type=int, default=8500,
                        help="Serving service host port")
    parser.add_argument('--batch-size', dest='request_batch_size', type=int, default=1,
                        help="Batch size for inference.")
    parser.add_argument('--num-threads', dest='num_threads', type=int, default=2,
                        help="Number of threads used for predict requests.")
    parser.add_argument('--num-images', dest='num_images', type=int, default=1000,
                        help="Number of images predicted by each thread.")
    parser.add_argument('--use-async', dest='use_async', action='store_true',
                        help="Use asynchoneus send recv.")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Print request latency from one of executing threads")
    parser.add_argument('--serving-bin-path', type=str, default='',
                        help="Path to TensorFlow serving binary file")
    hparams = terminal_argparse.handle_cmdline_arguments(parser)

    main(hparams)
