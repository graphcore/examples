# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow import keras
import strategy_utils as su
import argparse
import cv2
import numpy as np
import nif


if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required")


def parse_args():
    parser = argparse.ArgumentParser("Neural Image Field (NIF) Generator")
    parser.add_argument("--output", type=str, default="mlp_samples.png", help="Output image file name.")
    parser.add_argument("--model", type=str, default="./saved_model/", help="Input path to load a trained NIF model.")
    parser.add_argument("--width", type=int, default=0, help="Width of generated image.")
    parser.add_argument("--height", type=int, default=0, help="Height of generated image.")
    parser.add_argument("--original", type=str, default="",
                        help="If an original reference image is specified then an error "
                        "metric will be computed between it and the reconstruction.")
    parser.add_argument("--no-ipu", action="store_true", help="Set this flag to disable IPU specific code paths.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
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
        model = keras.models.load_model(args.model)
        model.summary()

        # Reconstruct the entire image from the trained NIF:
        output_sample_count = width * height
        prediction_batch_size = max(width, height)
        prediction_batches = output_sample_count // prediction_batch_size
        print(f"Required samples: {output_sample_count} and batches: {prediction_batches}")
        eval_ds, pixel_coords = nif.make_prediction_dataset(width, height, prediction_batch_size, embedding_dimension, embedding_sigma)
        result = model.predict(x=eval_ds, batch_size=prediction_batch_size, steps=prediction_batches)
        reconstructed = nif.decode_samples(img_shape, pixel_coords[0:output_sample_count].astype(np.int32), result, encode_params)
        cv2.imwrite(args.output, reconstructed)
        print(f"Saved image.")

        if args.original:
            img = cv2.imread(args.original, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
            psnr = cv2.PSNR(img, reconstructed)
            print(f"PSNR: {psnr}")
