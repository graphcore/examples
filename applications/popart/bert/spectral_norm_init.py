# Copyright 2019 Graphcore Ltd.

'''
This script loads an onnx Bert checkpoint and
apply spectral normalization to initializers.
'''

import numpy as np
import onnx
from timeit import default_timer as timer
import argparse


def load_model_and_initializers_from_onnx(model_path):
    initializers = {}
    model = onnx.load_model(model_path)
    for weight in model.graph.initializer:
        if weight.data_type == onnx.TensorProto.FLOAT16:
            int_data = np.asarray(weight.int32_data, np.int32)
            np_weight = int_data.view(dtype=np.float16).reshape(weight.dims)
        else:
            np_weight = onnx.numpy_helper.to_array(weight)
        initializers[weight.name] = np_weight
    return model, initializers


def spectral_normalization(float_16_weight):
    weight_float_32 = float_16_weight.astype("float32")
    s = np.linalg.svd(weight_float_32, full_matrices=False, compute_uv=False)
    spectral_norm = np.max(s)

    print(f"Spectral norm --> {spectral_norm}")

    weight_float_32 = weight_float_32 * 1.0 / spectral_norm
    return weight_float_32.astype("float16")


def spectral_normalization_QKV(weight):

    splitted_weights = np.hsplit(weight, 3)
    splitted_weights_normalized = [spectral_normalization(
        weight) for weight in splitted_weights]
    return np.concatenate(splitted_weights_normalized, axis=1)


if __name__ == "__main__":

    # Timer
    start = timer()

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='onnx model file to load')
    parser.add_argument('-o', '--output', help=' onnx model file to save')
    args = parser.parse_args()

    # Get initializers and apply spectral norm to each of them
    if not args.input.split('.')[1] == "onnx":
        print("input model file should have onnx extension, aborting.")
        exit()
    else:
        print(f"processing model {args.input}")
        model, initializers = load_model_and_initializers_from_onnx(args.input)

    for initializer in initializers.keys():

        print(f" \nGot tensor --> {initializer}")
        print(f" \n shape = {initializers[initializer].shape}")

        if initializers[initializer].ndim < 2:
            print(f"ndim < 2, skipping")

        elif "QKV" in initializer:
            initializers[initializer] = spectral_normalization_QKV(
                initializers[initializer])

        else:
            initializers[initializer] = spectral_normalization(
                initializers[initializer])

    # Save model
    if args.output:
        onnx.save(model, args.output)
    else:
        onnx.save(model, "out_spectral.onnx")

    # Duration
    stop = timer()
    print(f"\nProgram took {stop - start} seconds\n")
