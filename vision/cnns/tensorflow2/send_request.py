# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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


from typing import Callable, Tuple
import glob
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

from configuration import terminal_argparse
from pathlib import Path
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


def findServerBin() -> str:
    """Find location of TF2 Serving binary
    Returns:
        TF2 Serving binary path
    """

    server_dir = None

    if os.environ.get("TF_SERVING_PATH") is not None:
        server_dir = os.path.abspath(os.environ["TF_SERVING_PATH"])
    elif os.environ.get("POPLAR_SDK_PATH") is not None:
        server_dir = os.environ["POPLAR_SDK_PATH"]
        # POPLAR_SDK_PATH uses relative path, so to find serving binary
        # first we need to jump 4 levels up to public_examples_view folder
        server_dir = os.path.abspath(os.path.join(os.curdir, os.pardir, os.pardir, os.pardir, os.pardir, server_dir))
    elif os.environ.get("TF_POPLAR_BASE") is not None:
        server_dir = os.path.abspath(os.path.join(os.environ["TF_POPLAR_BASE"], os.pardir))
    elif os.environ.get("POPLAR_SDK_ENABLED") is not None:
        server_dir = os.path.abspath(os.path.join(os.environ["POPLAR_SDK_ENABLED"], os.pardir))
    else:
        sys.exit(
            f"Unable to find SDK location because TF_POPLAR_BASE, POPLAR_SDK_PATH, POPLAR_SDK_ENABLED or TF_SERVING_PATH"
            " env is not set, please use --serving-bin-path to point location of tensorflow server binary"
        )

    for file in os.listdir(server_dir):
        if file.startswith("tensorflow_model_server-r2"):
            return os.path.join(server_dir, file)

    sys.exit(f"No serving binary found in {server_dir}")
    return None


def exportModelAndStartServer(
    tensorflow_server_path: str, model_config: str, grpc_port: int, batch_size: int, model_batch_size: int, pytest: bool
):
    """Export TF2 model to PopEF format and start serving process
    Args:
        tensorflow_server_path:
                    Patch to TF2 serving binary
        model_config:
                    Path to the model config file
        grpc_port:
                    Port used for grpc connection
        batch_size:
                    Number of images sent in single predict request
        model_batch_size:
                     Number of images expected at model input in single predict request
    Returns:
        Serving process ID.
    """

    if tensorflow_server_path is None:
        tensorflow_server_path = findServerBin()
    if tensorflow_server_path is None:
        return None

    # Export model to PopEF format using external python script
    model_path = os.path.join(os.getcwd(), model_config)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    export_script_path = os.path.join(os.getcwd(), "scripts", "export_for_serving.py")
    result = subprocess.call(
        [
            "python3",
            export_script_path,
            "--config",
            model_config,
            "--export-dir",
            os.path.join(model_path, "001"),
            "--synthetic-data",
            "host",
            "--pytest",
            str(pytest),
            "--use-serving-api",
        ],
        cwd=os.getcwd(),
    )
    if result != 0:
        return None

    serving_params = [
        tensorflow_server_path,
        f"--model_base_path={model_path}",
        f"--model_name={model_config}",
        f"--port={grpc_port}",
    ]
    # For cases where request batch does not match batch size expected
    # by model, special config file is required. Serving will zero pad
    # data form client to match expected size.
    if model_batch_size != batch_size:
        config_dir = os.path.join(os.getcwd(), "batch.conf")
        with open(config_dir, "w") as f:
            f.write("max_batch_size { value: " + str(model_batch_size) + " }\n")
            f.write("batch_timeout_micros { value: 500 }\n")
            f.write("max_enqueued_batches { value:  1000 }\n")
            f.write("num_batch_threads { value: 4 }\n")
            f.write("allowed_batch_sizes : " + str(model_batch_size))
        serving_params.append("--enable_batching=true")
        serving_params.append(f"--batching_parameters_file={config_dir}")

    return subprocess.Popen(serving_params)


def checkServerStatus(serving_address) -> bool:
    """Check if serving is active"""
    channel = grpc.insecure_channel(serving_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "dummy_model"
    request.model_spec.signature_name = "serving_default"
    try:
        stub.Predict(request, 10)
    except grpc.RpcError as e:
        if "Servable not found for request: Latest(dummy_model)" == e.details():
            return True
        else:
            return False


def send_images(num_images: int, dataset_iter, batch_size: int, use_async: bool, config: str, stub, timeout: int):
    """Sends given number of images to TF2 Server for inference
    Args:
        num_images:
            Number of images to be sent (will be rounded up to batch size)
        dataset_iter:
            TF2 dataset iterator
        batch_size:
            Number of images sent in single predict request
        use_async:
            If True client will not wait for server response after sending
            image batch
        config:
            Name of configuration used when model was compiled, serves as model id
        stub:
            GRPC TF2 Serving object
        timeout:
            Maximal wait time for single request response
    Returns:
        List of result predictions
    """
    req_results = []
    future_result = None
    for nr in range(int(np.ceil(num_images / batch_size))):

        data, labels = next(dataset_iter)
        raw_data = data.numpy()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = config
        request.model_spec.signature_name = "serving_default"
        request.inputs["input"].CopyFrom(tf.make_tensor_proto(raw_data, shape=data.shape, dtype=data.dtype))

        pending_result = PendingResult()
        pending_result.labels = labels
        pending_result.start = time.time()
        if use_async:
            future_result = stub.Predict.future(request, timeout)
            req_results.append(pending_result)
            future_result.add_done_callback(pending_result.future_callback)
        else:
            pending_result.result = stub.Predict(request, timeout)
            pending_result.stop = time.time()
            req_results.append(pending_result)
    if use_async and future_result is not None:
        # Request queue works in FIFO mode, just wait for last one
        future_result.result()
    return req_results


def get_preprocessing_fn(img_datatype) -> Callable:
    def processing_fn(raw_record):
        return parse_imagenet_record(raw_record, img_datatype)

    return processing_fn


def preprocess_image(image_buffer, output_height: int, output_width: int, num_channels: int):
    """Decode, resize and crop imagenet image to format expected by model
    Returns:
        Reformated image
    """

    # Decode jpeg
    decoded_image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method="INTEGER_FAST")
    shape = tf.shape(input=decoded_image)

    # Resize image so that smaller dim matches target
    input_height = tf.cast(shape[0], tf.float32)
    input_width = tf.cast(shape[1], tf.float32)
    resize_min = tf.cast(int(output_height * float(256) / float(224)), tf.float32)
    smaller_dim = tf.minimum(input_height, input_width)
    scale_ratio = resize_min / smaller_dim
    scaled_height = tf.cast(input_height * scale_ratio, tf.int32)
    scaled_width = tf.cast(input_width * scale_ratio, tf.int32)
    resized_image = tf.compat.v1.image.resize(
        decoded_image, [scaled_height, scaled_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False
    )

    # Crop image
    crop_top = (scaled_height - output_height) // 2
    crop_left = (scaled_width - output_width) // 2
    image = tf.slice(resized_image, [crop_top, crop_left, 0], [output_height, output_width, -1])

    image.set_shape([output_height, output_width, num_channels])

    return image


def parse_imagenet_record(raw_record, dtype):
    """Parse raw_record data to extract image and label
    Returns:
        raw_image and label
    """
    feature_map = {
        "image/encoded": tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/class/label": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    }
    features = tf.io.parse_single_example(serialized=raw_record, features=feature_map)
    label = tf.cast(features["image/class/label"], dtype=tf.int32)

    image = preprocess_image(
        image_buffer=features["image/encoded"], output_height=224, output_width=224, num_channels=3
    )
    image = tf.cast(image, dtype)

    # Subtract one so that labels are in [0, 1000), and cast to int32 for
    # Keras model.
    label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1, dtype=tf.int32)
    return image, label


def get_imagenet_dataset(dataset_path: str) -> tf.data.Dataset:
    """Create dataset object form files in dataset_path
    Returns:
        Dataset object
    """

    if not os.path.exists(dataset_path):
        raise NameError(f"Directory {dataset_path} does not exist")

    tfrecord_prefix = "validation"

    filenames = glob.glob1(dataset_path, f"{tfrecord_prefix}*")

    filenames = list(map(lambda filename: os.path.join(dataset_path, filename), filenames))

    return tf.data.Dataset.from_tensor_slices(filenames)


def get_synthetic_dataset(image_shape: Tuple, eight_bit_transfer: bool):
    """Create generic dataset object
     Args:
        image_shape:
            Image params (height, width, num_channels)
        eight_bit_transfer:
            Use uint8
    Returns:
        Dataset object
    """

    images = tf.random.truncated_normal(image_shape, dtype=tf.float32, mean=127, stddev=60)
    if eight_bit_transfer:
        images = tf.cast(images, tf.uint8)

    labels = tf.random.uniform([], minval=0, maxval=1000, dtype=tf.float32)

    return tf.data.Dataset.from_tensors((images, labels))


def inference_process(process_index: int, barrier, hparams, serving_address):
    """Process function of serving client
    Args:
       process_index:
           Id of client
       barrier:
           Barrier used to synchronize client's
       hparams:
           Cmd line params
       serving_address:
           Address of serving service
    """

    fp_precision = precision.Precision(hparams.precision)
    fp_precision.apply()

    # Use uint8 for I/O data transfers between client and server
    if hparams.eight_bit_transfer:
        img_datatype = tf.uint8
    else:
        img_datatype = fp_precision.compute_precision

    ds = None

    # Use random generated data
    if hparams.synthetic_data == "host":
        ds = get_synthetic_dataset(image_shape=(224, 224, 3), eight_bit_transfer=hparams.eight_bit_transfer)
        ds = ds.cache()
        ds = ds.repeat()
    # Use data form dataset dir
    elif hparams.dataset == "imagenet":
        # Get the validation dataset
        preprocessing_fn = get_preprocessing_fn(img_datatype)
        ds = get_imagenet_dataset(hparams.dataset_path)
        ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=4, block_length=4, num_parallel_calls=4, deterministic=False
        )
        ds = ds.map(preprocessing_fn, num_parallel_calls=hparams.request_batch_size)

    else:
        sys.exit("Dataset source not compatible.")

    ds = ds.batch(batch_size=hparams.request_batch_size, drop_remainder=True, num_parallel_calls=1, deterministic=False)
    # Cache loaded images
    ds = ds.cache()
    # Make dataset infinite
    ds = ds.repeat()
    # Prefetch first batch
    ds.prefetch(hparams.request_batch_size)

    dataset_iter = iter(ds)
    # channel used for GRPC communication
    channel = grpc.insecure_channel(serving_address)
    # Object used for prediction requests
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Warmup
    send_images(hparams.request_batch_size, dataset_iter, hparams.request_batch_size, False, hparams.config, stub, 1000)

    # wait for other processes
    barrier.wait(600)

    results = send_images(
        hparams.num_images, dataset_iter, hparams.request_batch_size, hparams.use_async, hparams.config, stub, 10
    )

    if hparams.verbose and process_index == 0:
        latency = []
        for res in results:
            duration = res.stop - res.start
            print(f"-- Latency for batch: {(duration*1000):2.2f} ms.")
            latency.append(duration * 1000)
        print("Latency statistics")
        print("-------------------------------------------------------------------------------------------")
        print(f"Latencies - avg:{np.mean(latency)}, min: {np.min(latency)}, max: {np.max(latency)}, ")


def main(hparams):

    serving_address = f"{hparams.host}:{hparams.port}"

    # Change dataset path to local downloaded dir if this is called from Pytest
    if hparams.pytest:
        hparams.dataset_path = str(Path(__file__).absolute().parent)

    exporter_pid = exportModelAndStartServer(
        hparams.serving_bin_path,
        hparams.config,
        hparams.port,
        hparams.request_batch_size,
        hparams.micro_batch_size,
        hparams.pytest,
    )

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
    barrier = multiprocessing.Barrier(hparams.num_threads + 1)
    processes = []
    for r in range(hparams.num_threads):
        args = (r, barrier, hparams, serving_address)
        proc = multiprocessing.Process(target=inference_process, args=args)
        proc.start()
        processes.append(proc)
    print("Wait for workers")
    barrier.wait(600)
    start_all = time.time()
    print("Sending requests")

    for proc in processes:
        proc.join()

    end_all = time.time()
    print("All done")
    exec_time = end_all - start_all
    print("-------------------------------------------------------------------------------------------")
    print("Full time in ms:", exec_time * 1000)
    print(
        f"Processed num_images * num_threads: {hparams.num_images} * {hparams.num_threads} = {hparams.num_images*hparams.num_threads}"
    )
    print(f"Average throughput: {((hparams.num_images*hparams.num_threads)/exec_time):2.4f} samples/sec")


if __name__ == "__main__":
    # configure logger
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Serving TF2 classification Models", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Serving service host, example: 'localhost'")
    parser.add_argument("--port", type=int, default=8500, help="Serving service host port")
    parser.add_argument(
        "--batch-size", dest="request_batch_size", type=int, default=1, help="Batch size for inference."
    )
    parser.add_argument(
        "--num-threads", dest="num_threads", type=int, default=2, help="Number of threads used for predict requests."
    )
    parser.add_argument(
        "--num-images", dest="num_images", type=int, default=1000, help="Number of images predicted by each thread."
    )
    parser.add_argument("--use-async", dest="use_async", action="store_true", help="Use asynchoneus send recv.")
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Print request latency from one of executing threads"
    )
    parser.add_argument(
        "--pytest", dest="pytest", action="store_true", help="Whether or not this run is called from Pytest or not"
    )
    parser.add_argument("--serving-bin-path", type=str, default=None, help="Path to TensorFlow serving binary file")
    hparams = terminal_argparse.handle_cmdline_arguments(parser)

    main(hparams)
