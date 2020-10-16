# Copyright 2020 Graphcore Ltd.
import os
import argparse
import torch
from tqdm import tqdm
import poptorch
import logging
from data import get_data, datasets_info
from train_utils import accuracy
from dataloader import AsyncDataLoader
import sys
sys.path.append('..')
import models  # noqa: E402



# Set up logging
logging.basicConfig(format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


def test(inference_model, test_data, opts):
    nr_batches = len(test_data)
    bar = tqdm(test_data, total=nr_batches)
    sum_acc = 0.0
    with torch.no_grad():
        for idx, (data, labels) in enumerate(bar):
            if opts.precision == "half":
                data = data.half()
            data = data.contiguous()
            output = inference_model(data)
            output = output.float()
            sum_acc += accuracy(output, labels)
            aggregated_accuracy = sum_acc/(idx+1)
            bar.set_description("Accuracy:{:0.2f}%".format(aggregated_accuracy))
    print("Accuracy on test set: {:0.2f}%".format(sum_acc / len(test_data)))


def load_checkpoint_weights(inference_model, file_path):
    checkpoint = torch.load(file_path)
    opts = checkpoint['opts']
    logging.info("Restore the {0} model to epoch {1} on {2} dataset(Train loss:{3}, train accuracy:{4}%)".format(opts.model, checkpoint["epoch"], opts.data, checkpoint["loss"], checkpoint["train_accuracy"]))
    inference_model.model.load_state_dict(checkpoint['model_state_dict'])
    inference_model.copyWeightsToDevice()


def validate_checkpoints(checkpoint_list, test_data=None):
    checkpoint = torch.load(checkpoint_list[0])
    opts = checkpoint['opts']

    model_opts = poptorch.Options().deviceIterations(opts.device_iteration)
    model_opts.replicationFactor(opts.replicas)
    model_opts.anchorMode(poptorch.AnchorMode.All)

    if test_data is None:
        logging.info("Loading the data")
        train_data, test_data = get_data(opts, model_opts)
        if isinstance(train_data, AsyncDataLoader):
            train_data.stop_data_fetch()

    logging.info("Create model")
    model = models.get_model(opts, datasets_info[opts.data], pretrained=False)
    model.eval()
    inference_model = poptorch.inferenceModel(model, model_opts)

    for checkpoint in checkpoint_list:
        load_checkpoint_weights(inference_model, checkpoint)
        test(inference_model, test_data, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run validation on a given checkpoint')
    parser.add_argument('--checkpoint-path', help="The path of the checkpoint file", required=True)
    args = parser.parse_args()
    if os.path.isdir(args.checkpoint_path):
        checkpoint_files = [os.path.join(args.checkpoint_path, file_name) for file_name in os.listdir(args.checkpoint_path)]
    else:
        checkpoint_files = [args.checkpoint_path]

    validate_checkpoints(checkpoint_files)
