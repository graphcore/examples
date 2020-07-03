# Copyright 2019 Graphcore Ltd.
import torch
import torch.onnx
import urllib.request
import pretrainedmodels
import os
import onnx
import argparse

"""
Downloads the model in Pytorch format and converts to ONNX.
Creates copies with different (micro) batch size dimensions.
"""


def get_model(opts):

    path = "{}/{}/".format(opts.model_path, opts.model_name)
    log_path = "{}/".format(opts.log_path)
    filename = "model.onnx"

    if not os.path.exists(path):
        print("Creating models directory")
        os.makedirs(path)

    if not os.path.exists(log_path):
        print("Creating logs directory")
        os.makedirs(log_path)

    dataset = "imagenet"

    if opts.url:
        # Monkey patch an alternate URL into pretrained models package
        pretrainedmodels.models.resnext.pretrained_settings[
            opts.model_name
        ][dataset]["url"] = opts.url
        print(f"Download URL set to {opts.url}")

    pretrained_model_base_path = opts.pretrained_model_path

    if not pretrained_model_base_path:
        # Get the model. If it doesn't exist it will be downloaded
        if not os.path.isfile(path + filename):
            print(f"Downloading model to {path + filename}")
        pretrained_model_path = path + filename

        # Create the right input shape
        dummy_input = torch.randn(1, 3, 224, 224)
        model = pretrainedmodels.__dict__[opts.model_name](
            num_classes=1000, pretrained=dataset)
        torch.onnx.export(model, dummy_input, pretrained_model_path)
    else:
        pretrained_model_path = os.path.join(
            pretrained_model_base_path,
            opts.model_name,
            filename
        )

    onnx_model = onnx.load(pretrained_model_path)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = opts.micro_batch_size
    print(
        f"Converting model to batch size {opts.micro_batch_size} and saving to {path + 'model_' + str(opts.micro_batch_size) + '.onnx'}")
    onnx.save(onnx_model, path + f"model_{opts.micro_batch_size}.onnx")


parser = argparse.ArgumentParser()
parser.add_argument("--micro-batch-size", type=int, default=1, help="""Batch size per device.
    Larger batches can be run with this model by launching the app with resnext_inference_launch.py and passing in a value > 1 for num_ipus """)
parser.add_argument("--model-name", type=str, default='resnext101_32x4d',
                    help="pretrained model name, according to `pretrainedmodels` Python package")
parser.add_argument(
    "--model-path", type=str, default="models",
    help=(
        "If set, the model will be saved to this"
        " specfic path, instead of models/"
    )
)
parser.add_argument(
    "--log-path", type=str, default="logs",
    help=(
        "If set, the logs will be saved to this"
        " specfic path, instead of logs/"
    )
)
parser.add_argument(
    "--url", type=str, default=None,
    help="If set, uses an alternate url to get the pretrained model from."
)
parser.add_argument(
    "--pretrained-model-path", type=str, default=None,
    help=(
        "If set, the pretrained model will attempt to be loaded from here,"
        " instead of downloaded."
    )
)

# set up directory
model_name = 'resnext101_32x4d'
filename = "model.onnx"


if __name__ == "__main__":
    opts = parser.parse_args()
    get_model(opts)
