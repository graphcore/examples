# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow import keras
import strategy_utils as su
import argparse
import cv2
import numpy as np
import nif
import os
from ipu_tensorflow_addons.keras.optimizers import AdamIpuOptimizer
from skimage import color, metrics

if tf.__version__[0] != "2":
    raise ImportError("TensorFlow 2 is required")


def parse_args():
    parser = argparse.ArgumentParser("Neural Image Field (NIF) Generator")
    parser.add_argument("--output", type=str, default="mlp_samples.png", help="Output image file name.")
    parser.add_argument("--model", type=str, default="./saved_model/", help="Input path to load a trained NIF model.")
    parser.add_argument("--width", type=int, default=0, help="Width of generated image.")
    parser.add_argument("--height", type=int, default=0, help="Height of generated image.")
    parser.add_argument(
        "--original",
        type=str,
        default="",
        help="If an original reference image is specified then an error "
        "metric will be computed between it and the reconstruction.",
    )
    parser.add_argument("--no-ipu", action="store_true", help="Set this flag to disable IPU specific code paths.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    metadata_path = nif.metadata_path_from_keras_model_path(args.model)
    metadata = nif.load_metadata(metadata_path)
    print(f"NIF info: {metadata}")

    img_shape = metadata["original_image_shape"]
    embedding_dimension = metadata["embedding_dimension"]
    embedding_sigma = metadata["embedding_sigma"]
    encode_params = metadata["encode_params"]

    if args.width == 0 or args.height == 0:
        width = img_shape[1]
        height = img_shape[0]
    else:
        width = args.width
        height = args.height
        img_shape[1] = width
        img_shape[0] = height

    strategy = su.get_predict_strategy(args.no_ipu)

    with strategy.scope():
        # Upstream Keras format does not support mixed precision optimiser slots so we load the H5 inference model:
        h5_model_path = os.path.join(args.model, "assets.extra", "converted.hdf5")
        model = keras.models.load_model(h5_model_path, custom_objects={"AdamIpuOptimizer": AdamIpuOptimizer})
        model.summary()

        # Reconstruct the entire image from the trained NIF:
        output_sample_count = width * height
        prediction_batch_size = max(width, height)
        prediction_batches = output_sample_count // prediction_batch_size
        print(f"Required samples: {output_sample_count} and batches: {prediction_batches}")
        eval_ds, pixel_coords = nif.make_prediction_dataset(
            width, height, prediction_batch_size, embedding_dimension, embedding_sigma
        )
        result = model.predict(x=eval_ds, batch_size=prediction_batch_size, steps=prediction_batches)
        reconstructed = nif.decode_samples(
            img_shape, pixel_coords[0:output_sample_count].astype(np.int32), result, encode_params
        )
        cv2.imwrite(args.output, reconstructed)
        print(f"Saved image.")

        if args.original:
            bgr_img = cv2.imread(args.original, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Compute PSNR on RGB directly:
            rgb_max = np.max(rgb_img)
            rgb_mse = metrics.mean_squared_error(rgb_img, reconstructed)
            psnr = 10 * np.log10((rgb_max / rgb_mse) * rgb_max)
            print(f"PSNR RGB: {psnr}")

            # Also compute PSNR separately for luminance and chrominance using the Lab colour space. This colour space
            # is perceptually linear in theory. (See "Efficient High Dynamic Range Texture Compression", Roimela et. al. 2008)
            lab_img = color.rgb2lab(rgb_img)
            lab_reconstructed = color.rgb2lab(reconstructed)

            # Luminance PSNR:
            lum_img = lab_img[:, :, 0]
            lum_max = np.max(lum_img)
            lum_recon = lab_reconstructed[:, :, 0]
            lum_mse = metrics.mean_squared_error(lum_img, lum_recon)
            luminal_psnr = 10 * np.log10((lum_max / lum_mse) * lum_max)
            print(f"PSNR L: {luminal_psnr}")

            # Chrominance PSNR:
            ab_img = lab_img[:, :, 1:]
            ab_recon = lab_reconstructed[:, :, 1:]
            ab_max = np.max(ab_img)
            ab_mse = metrics.mean_squared_error(ab_img, ab_recon)
            chromatic_psnr = 10 * np.log10((ab_max / ab_mse) * ab_max)
            print(f"PSNR AB: {chromatic_psnr}")
