# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
import cv2
import numpy as np
import json
import os
import subprocess
import re
import time
import datetime
from multiprocessing import Pool
import math


def value_stats(values):
    max = values.max().astype(float)
    mean = np.mean(values, axis=0).astype(float)
    return max, mean


def deterministic_uv_samples(image_shape, dtype):
    height = image_shape[0]
    width = image_shape[1]
    _, uv = make_image_grid(width, height)
    return uv.astype(dtype)


def stochastic_uv_samples(rng, sample_count, image_shape, dtype):
    u = rng.uniform(low=0.0, high=1.0, size=[sample_count, 1]).astype(dtype)
    v = rng.uniform(low=0.0, high=1.0, size=[sample_count, 1]).astype(dtype)
    return np.concatenate([u, v], axis=1)


def decode_samples(image_shape, uv, values, params):
    if uv.shape[0] != values.shape[0]:
        raise ValueError(f"Size mismatch between uv coords: {uv.shape} and values: {values.shape}")
    decoded_values = values * params["max"]

    transfer_function = params["transfer_function"]
    if transfer_function not in ["linear", "log"]:
        raise ValueError(f"Unsupported transfer function: {transfer_function}")

    if transfer_function == "log":
        decoded_values += np.array(params["mean"]) - params["eps"]
        decoded_values = np.exp(decoded_values)
    else:
        decoded_values += np.array(params["mean"])

    output = np.zeros(shape=image_shape, dtype=np.float32)
    i = 0
    for r, c in uv:
        output[r][c] = decoded_values[i]
        i += 1
    return output


def bilinear_interpolate(img, coords):
    x = coords[:, 1]
    y = coords[:, 0]

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = np.expand_dims(wa, 1)
    wb = np.expand_dims(wb, 1)
    wc = np.expand_dims(wc, 1)
    wd = np.expand_dims(wd, 1)

    wa = np.repeat(wa, Ia.shape[1], axis=1)
    wb = np.repeat(wb, Ib.shape[1], axis=1)
    wc = np.repeat(wc, Ic.shape[1], axis=1)
    wd = np.repeat(wd, Id.shape[1], axis=1)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def encode_samples(image, uv_coords, transfer_function, debug_filename=""):
    width = image.shape[1]
    height = image.shape[0]
    remap_coords = (uv_coords * [height - 1, width - 1]).astype(np.float32)
    pixel_coords = np.rint(remap_coords).astype(np.int32)

    image = image.astype(np.float32)
    max_value = image.max()
    # If the image is high dynamic range we use log
    # tone-mapping to lower the dynamic range for training:
    eps = 0.00000001
    if transfer_function == "log":
        image = np.log(image + eps)
        max_value = image.max()

    mean_value = np.mean(np.mean(image, axis=0), axis=0)
    image -= mean_value
    image *= 1.0 / max_value

    # Get bi-linear filtered samples:
    train_values = bilinear_interpolate(image, remap_coords).astype(np.float32)

    # Create a JSON serialiseable dict:
    encode_params = {
        "mean": list(mean_value.astype(float)),
        "max": max_value.astype(float),
        "log_tone_map": transfer_function == "log",  # Set for backwards compatibility
        "transfer_function": transfer_function,
        "eps": eps,
    }

    # Optionally reconstruct an image from the input samples (useful for debugging):
    if debug_filename:
        cv2.imwrite(debug_filename, decode_samples(image.shape, pixel_coords, train_values, encode_params))

    return train_values, encode_params


def make_image_grid(width, height):
    u_coords, v_coords = np.mgrid[0.0:height, 0.0:width]
    pixel_coords = np.array([u_coords.flatten(), v_coords.flatten()]).transpose()
    u_coords /= height
    v_coords /= width
    uv_coords = np.array([u_coords.flatten(), v_coords.flatten()]).transpose()
    return pixel_coords, uv_coords


def make_train_data(train_uv, train_values, args):
    factor = args.gradient_accumulation_count * args.replicas
    dataset_batch_size = args.batch_size // factor
    effective_batch_size = dataset_batch_size * factor
    print(f"Effective batch-size: {effective_batch_size} " f"Dataset batch-size: {dataset_batch_size} ")
    train_steps = train_uv.shape[0] // effective_batch_size

    # Steps per exec must be multiple of the number of replicas:
    steps_per_exec = factor
    step_ratio = train_steps // steps_per_exec
    # Need many steps per exec for efficiency:
    if step_ratio > 1:
        steps_per_exec *= step_ratio

    # Steps per epoch must be multiple of steps per exec:
    steps_per_epoch = (train_steps // steps_per_exec) * steps_per_exec

    ds = tf.data.Dataset.from_tensor_slices((train_uv, train_values)).batch(dataset_batch_size, drop_remainder=True)
    print(f"Training dataset: {ds}")
    print(f"Training steps-per-epoch: {steps_per_epoch} " f"Training steps-per-execution: {steps_per_exec}")
    if steps_per_epoch == 0:
        raise RuntimeError(
            "Training steps == 0: make sure you have enough training samples given the batch size and exec steps."
        )
    return ds, steps_per_epoch, steps_per_exec


def make_prediction_dataset(width, height, batch_size, embedding_dimension, embedding_sigma):
    pixel_coords, uv_coords = make_image_grid(width, height)
    if embedding_dimension > 0:
        uv_coords = uv_positional_encode(uv_coords, embedding_dimension, embedding_sigma)
    ds = tf.data.Dataset.from_tensor_slices(uv_coords).batch(batch_size, drop_remainder=True)
    print(f"Prediction dataset: {ds}")
    return ds, pixel_coords


def sincos(coeffs, uv, idx):
  results = []
  i = idx
  for px, py in uv:
    # Order of components doesn't matter as they will be fed to a fully
    # connected layer so we can concatenate in a more efficient order:
    pos = np.concatenate([coeffs * px, coeffs * py])
    results.append([i, np.concatenate([np.sin(pos), np.cos(pos)])])
    i += 1
  return results


# This is the position encoding the original NERF paper.
def uv_positional_encode(uv, dimension, sigma):
    print(f"UV input samples shape: {uv.shape}")
    powers = np.arange(0.0, dimension, 1.0)
    coeffs = np.power([sigma], powers)
    uv2 = 2 * (uv - 1.0)
    encoded = np.empty([uv.shape[0], 4 * dimension], dtype=uv.dtype)
    print(f"UV position encoded shape: {encoded.shape}")

    # Encoding can be slow so use a process pool:
    processes = 40
    async_results = []
    with Pool(processes) as p:
      chunk_size = math.ceil(uv.shape[0] / processes)
      chunks = math.ceil(uv.shape[0] / chunk_size)
      for i in range(0, chunks):
        start = i * chunk_size
        end = start + chunk_size
        if end > uv.shape[0]:
          end = uv.shape[0]
        uv_slice = uv[start:end, :]
        result = p.apply_async(sincos, [coeffs, uv_slice, start])
        async_results.append(result)

      for a in async_results:
        rows = a.get(timeout=200)
        for r in rows:
          idx = r[0]
          vec = r[1]
          encoded[idx] = vec

    return encoded


def save_metadata(file_name, name, args, shape, encode_params, embedding_dim, embedding_sigma, model_path):
    nif_params = {
        "name": name,
        "train_command": args,
        "original_image_shape": list(shape),
        "embedding_dimension": embedding_dim,
        "embedding_sigma": embedding_sigma,
        "keras_model": model_path,
        "encode_params": encode_params,
    }
    with open(file_name, "w") as file:
        json.dump(nif_params, file, indent=2, sort_keys=True)


def load_metadata(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)
    return data


def metadata_path_from_keras_model_path(model_path):
    return os.path.join(model_path, "assets.extra", "nif_metadata.txt")


def find_output_value(text, regex):
    items = re.findall(regex, text, re.MULTILINE)
    try:
        match = items[-1]
        value = float(match.split()[-1])
    except Exception:
        print(f"Could not find regex '{regex}' in output:\n{text}")
        value = None
    return value


def color_space_to_bgr_matrix(color_space: str):
    """
    Return the matrix that transforms from the specified colour-space to RGB
    or None if the space is RGB.

    NOTE: Storage order for RGB is BGR for OpenCV compatibility.
    """
    if color_space not in ["rgb", "yuv", "ycocg"]:
        raise ValueError(f"Unsupported color space: {color_space}")
    # From YUV color space:
    if color_space == "yuv":
        return np.array([[1, 2.032, 0], [1, -0.395, -0.581], [1, 0, 1.140]])  # B  # G  # R
    # From YCoCg color space:
    if color_space == "ycocg":
        return np.array([[1, -1, -1], [1, 0, 1], [1, -1, 1]])  # B  # G  # R
    return None


class EvalCallback(tf.keras.callbacks.Callback):
    """
    A Keras callback which logs training info and optionally launches a
    separate evaluation process to run in parallel with training.
    """

    def __init__(self, original_file, model_path, period, compute_psnr, no_ipu, log_dir=None):
        super().__init__()
        self.input_file = original_file
        self.model_path = model_path
        self.period = period
        self.eval_process = None
        self.compute_psnr = compute_psnr
        _, file_extension = os.path.splitext(self.input_file)
        self.tmp_output = os.path.join(self.model_path, "tmp_eval_image" + file_extension)
        self.eval_args = [
            "python3",
            "predict_nif.py",
            "--model",
            self.model_path,
            "--output",
            self.tmp_output,
            "--original",
            self.input_file,
        ]
        if no_ipu:
            self.eval_args.append("--no-ipu")
        if log_dir:
            self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "eval"))
            img = self.load_image_for_tensorboard(self.input_file)
            with self.summary_writer.as_default():
                tf.summary.image("Original Image", img, step=0)
        self.epoch_start_time = None

    def load_image_for_tensorboard(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0)

    def wait_until_finished(self, epoch):
        if self.eval_process:
            print(f"Waiting for previous eval process (pid: {self.eval_process.pid}) to finish...")
            wait_start = time.time()
            try:
                self.eval_process.wait(timeout=90)
                (
                    stdout,
                    stderr,
                ) = self.eval_process.communicate(timeout=30)
                wait_time = time.time() - wait_start
                print(f"Previous eval process successful (waited {wait_time:.2f} seconds).")
                # Read the evaluation metric back (currently peak-signal-to-noise-ratio):
                text = stdout.decode("utf-8")

                psnr_rgb = find_output_value(text, "PSNR RGB:.*$")
                psnr_l = find_output_value(text, "PSNR L:.*$")
                psnr_ab = find_output_value(text, "PSNR AB:.*$")
                print(f"RGB PSNR RGB at epoch {epoch}: {psnr_rgb}")
                print(f"LUMINAL PSNR at epoch {epoch}: {psnr_l}")
                print(f"CHROMATIC PSNR at epoch {epoch}: {psnr_ab}")

                # Log the result if successful:
                if self.summary_writer and psnr_rgb is not None:
                    img = self.load_image_for_tensorboard(self.tmp_output)
                    with self.summary_writer.as_default():
                        tf.summary.image("Reconstructed Image", img, step=epoch)
                        tf.summary.scalar("PSNR RGB", psnr_rgb, step=epoch)
                        tf.summary.scalar("PSNR L", psnr_l, step=epoch)
                        tf.summary.scalar("PSNR AB", psnr_ab, step=epoch)

            except subprocess.TimeoutExpired:
                print(f"Killing previous eval process for taking too long (pid: {self.eval_process.pid}).")
                self.eval_process.kill()
                (
                    stdout,
                    stderr,
                ) = self.eval_process.communicate()
            return stdout, stderr
        return None, None

    def cleanup(self, epoch):
        stdout, stderr = self.wait_until_finished(epoch - self.period)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Completed epoch {epoch} in {epoch_time:.2f} seconds. Loss: {logs['loss']}")
        if epoch % self.period == 0:
            out, err = self.wait_until_finished(epoch - self.period)
            if self.compute_psnr:
                # Set env so we use executable caching for eval process:
                eval_env = os.environ.copy()
                cache_path = os.path.join(self.model_path, "poplar_cachedir")
                tf_poplar_flags = f"--executable_cache_path={cache_path} "
                if "TF_POPLAR_FLAGS" in eval_env:
                    tf_poplar_flags += eval_env["TF_POPLAR_FLAGS"]
                eval_env["TF_POPLAR_FLAGS"] = tf_poplar_flags
                self.eval_process = subprocess.Popen(
                    self.eval_args, env=eval_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                print(
                    f"{datetime.datetime.now().isoformat()}: Launched eval process (pid: {self.eval_process.pid}) at epoch {epoch}"
                )
