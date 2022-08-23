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

import atexit
import argparse
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
from tensorflow.keras.preprocessing import image
import yaml

from inference_networks import model_dict


class PendingResult(object):

    def __init__(self):
        self.start = 0
        self.stop = 0
        self.result = None
        self.labels = None

    def future_callback(self, future_res):
        self.stop = time.time()
        self.result = future_res.result()


def send_images(num_images, images, input_shape, dtype, use_async, request, stub):
    batch_count = len(images)
    batch_size = input_shape[0]
    req_results = []
    future_result = None
    for nr in range(int(np.ceil(num_images/batch_size))):

        batch, labels = images[nr % batch_count]
        request.inputs['in_img'].CopyFrom(
            tf.compat.v1.make_tensor_proto(batch, shape=input_shape, dtype=dtype))

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


def inference_process(process_index, barrier, model_arch, image_dir, server_address, batch_size, num_images, use_async, validate_output):
    # Select model architecture
    network_class = model_dict[model_arch]
    if model_arch == 'googlenet':
        model_arch = 'inceptionv1'
    config = os.path.join('configs', model_arch+'.yml')
    results = []

    # Model specific config
    with open(config) as file_stream:
        try:
            config_dict = yaml.safe_load(file_stream)
        except yaml.YAMLError as exc:
            tf.logging.error(exc)

    if 'dtype' not in config_dict:
        config_dict["dtype"] = 'float16'

    preprocess_input = network_class.preprocess_method()
    decode_predictions = network_class.decode_method()

    input_shape = [batch_size, config_dict["input_shape"][0],
                   config_dict["input_shape"][1], config_dict["input_shape"][2]]

    # Image preprocessing
    images = []
    batch = np.empty(input_shape, dtype=config_dict["dtype"])
    idx = 0
    batch_id = 0
    for label in os.listdir(image_dir):
        if label.endswith(".jpg") and not label.startswith("."):
            img_path = os.path.join(image_dir, label)
            img = image.load_img(img_path, target_size=(
                config_dict["input_shape"][0], config_dict["input_shape"][1]))
            raw_data = image.img_to_array(img)
            data = preprocess_input(raw_data)
            if idx % batch_size == 0:
                batch_id = 0
                batch = np.empty(input_shape, dtype=config_dict["dtype"])
                labels = []
            batch[batch_id] = data
            labels.append(label)
            batch_id += 1
            idx += 1
            if batch_id == batch_size:
                images.append((batch, labels))
            if idx >= num_images:
                break
    if batch_id != batch_size:
        images.append((batch, labels))

    channel = grpc.insecure_channel(server_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_arch
    request.model_spec.signature_name = 'serving_default'

    # Warmup
    send_images(batch_size*10, images, input_shape,
                config_dict["dtype"], False, request, stub)

    # wait for other processes
    barrier.wait(10)

    results = send_images(num_images, images, input_shape,
                          config_dict["dtype"], use_async, request, stub)

    if validate_output and process_index == 0:
        latency = []
        for res in results:
            predicted = res.result.outputs['output_0']
            predicted = tf.contrib.util.make_ndarray(predicted)
            predicted = np.reshape(predicted, (-1, predicted.shape[-1]))
            duration = res.stop - res.start
            print(f"-- Latency for batch: {(duration*1000):2.2f} ms.")
            latency.append(duration*1000)
            if model_arch in ("inceptionv1", "efficientnet-s", "efficientnet-m", "efficientnet-l"):
                # These models encode background in 0th index.
                decoded_predictios = decode_predictions(
                    predicted[:, 1:], top=3)
            else:
                decoded_predictios = decode_predictions(predicted, top=3)
            for idx, label in enumerate(res.labels):
                print('Actual: ', label)
                print('Predicted: ', decoded_predictios[idx])

        print("Latency statistics")
        print("-------------------------------------------------------------------------------------------")
        print(
            f"Average latency={np.mean(latency)}, Minimal latency={np.min(latency)}, Maximal latency={np.max(latency)}")


def findServerBin():

    server_dir = ""

    if os.environ.get('POPLAR_SDK_PATH') is not None:
        server_dir = os.environ["POPLAR_SDK_PATH"]
        # POPLAR_SDK_PATH uses relative patch, so to find serving binary
        # first we need to jump 5 levels up to public_examples_view folder
        server_dir = os.path.abspath(os.path.join(
            os.curdir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, server_dir))
    elif os.environ.get('TF_POPLAR_BASE') is not None:
        server_dir = os.path.abspath(os.path.join(
            os.environ["TF_POPLAR_BASE"], os.pardir))
    else:
        return ""

    if not os.path.exists(server_dir):
        raise AssertionError(server_dir)
    print(f"Searching for serving binary in: {server_dir}")
    for file in os.listdir(server_dir):
        if file.startswith("tensorflow_model_server-r1"):
            server_bin = os.path.join(server_dir, file)
            print(f"Serving bin found: {server_bin}")
            return server_bin
    return ""


def exportModelAndStartServer(model_config, grpc_port):
    tensorflow_server_path = findServerBin()
    if tensorflow_server_path == "":
        return None

    model_path = os.path.join(os.getcwd(), model_config['model_arch'])
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    export_script_path = os.path.join(
        os.getcwd(), 'export_for_serving.py')

    export_params = ['python3', export_script_path,
                     model_config['model_arch'],
                     '--batch-size', str(model_config['model_batch_size']),
                     ]

    result = subprocess.call(export_params)
    if result != 0:
        return None

    serving_params = [tensorflow_server_path,
                      f'--model_base_path={model_path}',
                      f'--model_name={model_config["model_arch"]}',
                      f'--port={grpc_port}']
    if model_config['model_batch_size'] != model_config['batch_size']:
        config_dir = os.path.join(os.getcwd(), 'batch.conf')
        with open(config_dir, 'w') as f:
            f.write(
                'max_batch_size { value: ' + str(model_config['model_batch_size']) + ' }\n')
            f.write('batch_timeout_micros { value: 100000 }\n')
            f.write('max_enqueued_batches { value: 1000000 }\n')
            f.write('num_batch_threads { value: 4 }\n')
            f.write('allowed_batch_sizes : ' +
                    str(model_config['model_batch_size']))
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


def main(model_arch, image_dir, host, port, batch_size, num_threads,
         num_images, model_batch_size, use_async, verbose):

    model_config = {
        'model_arch': model_arch,
        'model_batch_size': model_batch_size,
        'batch_size': batch_size
    }

    serving_address = f"{host}:{port}"

    exporter_pid = exportModelAndStartServer(model_config, port)
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
    barrier = multiprocessing.Barrier(num_threads+1)
    processes = []
    for r in range(num_threads):
        args = (r, barrier, model_arch, image_dir, serving_address,
                batch_size, num_images, use_async, verbose)
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
        f"Processed num_images * num_threads: {num_images} * {num_threads} = {num_images*num_threads}")
    print(f"Average img/s: {((num_images*num_threads)/exec_time):2.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Serving request testing.")
    parser.add_argument('model_arch', type=str.lower,
                        choices=["googlenet", "inceptionv1", "mobilenet", "mobilenetv2",
                                 "inceptionv3", "resnet50", "densenet121", "xception", "efficientnet-s",
                                 "efficientnet-m", "efficientnet-l"],
                        help="Type of image classification model.")
    parser.add_argument('image_dir', type=str, default="", nargs='?',
                        help="Path to directory of images to run inference on.")
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help="Serving service host, example: 'localhost'")
    parser.add_argument('--port', type=int, default=8500,
                        help="Serving service host port")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                        help="Batch size for inference.")
    parser.add_argument('--num-threads', dest='num_threads', type=int, default=2,
                        help="Number of threads used for predict requests.")
    parser.add_argument('--num-images', dest='num_images', type=int, default=1000,
                        help="Number of images predicted by each thread.")
    parser.add_argument('--model-batch-size', dest='model_batch_size', type=int, default=0,
                        help="Number of images predicted by each thread.")
    parser.add_argument('--use-async', dest='use_async', action='store_true',
                        help="Use asynchoneus send recv.")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Print predictions and latency from one of executing threads")

    args = parser.parse_args()

    if args.model_batch_size == 0:
        args.model_batch_size = args.batch_size

    main(args.model_arch, args.image_dir, args.host, args.port, args.batch_size,
         args.num_threads, args.num_images, args.model_batch_size, args.use_async, args.verbose)
