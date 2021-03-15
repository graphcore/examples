# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import argparse
import torch
from tqdm import tqdm
import poptorch
import logging
import sys
sys.path.append('..')
import models
import utils
import datasets


def test(inference_model, test_data, opts):
    nr_batches = len(test_data)
    bar = tqdm(test_data, total=nr_batches)
    sum_acc = 0.0

    with torch.no_grad():
        for idx, (input_data, labels) in enumerate(bar):
            output = inference_model(input_data)
            output = output.float()
            sum_acc += utils.accuracy(output, labels)
            aggregated_accuracy = sum_acc/(idx+1)
            bar.set_description(f"Accuracy:{aggregated_accuracy:0.2f}%")
    acc = sum_acc / nr_batches
    logging.info(f"Accuracy on test set: {acc:0.2f}%")
    return acc


def load_checkpoint_weights(inference_model, file_path):
    checkpoint = torch.load(file_path)
    opts = checkpoint['opts']
    logging.info(f"Restore the {opts.model} model to epoch {checkpoint['epoch']} on {opts.data} dataset(Train loss:{checkpoint['loss']}, train accuracy:{checkpoint['train_accuracy']}%)")
    inference_model.model.load_state_dict(checkpoint['model_state_dict'])
    inference_model.copyWeightsToDevice()


def create_validation_opts(opts):
    model_opts = poptorch.Options().deviceIterations(max(opts.device_iterations, 1+len(opts.pipeline_splits)))
    model_opts.replicationFactor(opts.replicas)
    model_opts.anchorMode(poptorch.AnchorMode.All)
    return model_opts


def validate_checkpoints(checkpoint_list, test_data=None):
    checkpoint = torch.load(checkpoint_list[0])
    opts = checkpoint['opts']
    utils.Logger.setup_logging_folder(opts)

    # make sure the order is ascending
    def ckpt_key(ckpt):
        return int(ckpt.split('_')[-1].split('.')[0])
    try:
        checkpoint_list = sorted(checkpoint_list, key=ckpt_key)
    except:
        logging.warn("Checkpoint names are changed, which may cause inconsistent order in evaluation.")

    model_opts = create_validation_opts(opts)

    if test_data is None:
        logging.info("Loading the data")
        test_data = datasets.get_data(opts, model_opts, train=False, async_dataloader=True)

    logging.info("Create model")
    model = models.get_model(opts, datasets.datasets_info[opts.data], pretrained=False)
    model.eval()
    inference_model = poptorch.inferenceModel(model, model_opts)

    for checkpoint in checkpoint_list:
        load_checkpoint_weights(inference_model, checkpoint)
        acc = test(inference_model, test_data, opts)
        epoch_nr = torch.load(checkpoint)["epoch"]
        result_dict = {"validation_epoch": epoch_nr,
                       "validation_iteration": opts.logs_per_epoch * epoch_nr,
                       "validation_accuracy": acc}
        utils.Logger.log_validate_results(result_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run validation on a given checkpoint')
    parser.add_argument('--checkpoint-path', help="Path of checkpoint file or folder", required=True)
    args = parser.parse_args()
    if os.path.isdir(args.checkpoint_path):
        checkpoint_files = [os.path.join(args.checkpoint_path, file_name) for file_name in os.listdir(args.checkpoint_path) if file_name.endswith(".pt")]
    else:
        checkpoint_files = [args.checkpoint_path]
    validate_checkpoints(checkpoint_files)
